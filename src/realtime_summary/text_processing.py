from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import monotonic
from typing import Any, Protocol, cast

from google.genai import types

from .diagnostics import log_event, log_exception
from .state import CompletedTurn, SummaryState


GEMINI_MODEL = "gemini-3.1-flash-lite"


class InvalidSummaryResponse(ValueError):
    """概要応答がOBSへ表示できる構造ではない。"""


@dataclass(frozen=True, slots=True)
class SummaryResult:
    context_memory: str
    display_lines: tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class GeneratedText:
    text: str
    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True, slots=True)
class GeneratedSummary:
    summary: SummaryResult
    input_tokens: int | None
    output_tokens: int | None


class GeminiTextProcessor:
    def __init__(self, client: Any) -> None:
        self._models = client.aio.models

    async def translate(self, transcript: str) -> GeneratedText:
        response = await self._models.generate_content(
            model=GEMINI_MODEL,
            contents=transcript,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "日本語の配信字幕を、意味を保った簡潔で自然な英語字幕へ翻訳する。"
                    "説明や引用符を付けず、翻訳だけを返す。"
                ),
                temperature=0.2,
                max_output_tokens=256,
            ),
        )
        return _generated_text(response)

    async def summarize(
        self,
        *,
        context_memory: str,
        transcripts: tuple[str, ...],
    ) -> GeneratedSummary:
        input_payload = json.dumps(
            {
                "previous_context": context_memory,
                "new_transcripts": transcripts,
            },
            ensure_ascii=False,
        )
        response = await self._models.generate_content(
            model=GEMINI_MODEL,
            contents=input_payload,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "日本語の一人語り配信を要約する。過去の重要な文脈と現在の話題を"
                    "保ち、古くなった話題は外す。context_memoryは次回へ渡す短い状態、"
                    "display_linesはOBSへ表示する簡潔な日本語を必ず3行で返す。"
                ),
                temperature=0.2,
                max_output_tokens=512,
                response_mime_type="application/json",
                response_json_schema={
                    "type": "object",
                    "properties": {
                        "context_memory": {"type": "string"},
                        "display_lines": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 3,
                        },
                    },
                    "required": ["context_memory", "display_lines"],
                    "additionalProperties": False,
                },
            ),
        )
        generated = _generated_text(response)
        return GeneratedSummary(
            summary=parse_summary_response(generated.text),
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
        )


class TextProcessor(Protocol):
    async def translate(self, transcript: str) -> GeneratedText: ...

    async def summarize(
        self,
        *,
        context_memory: str,
        transcripts: tuple[str, ...],
    ) -> GeneratedSummary: ...


TextCallback = Callable[[str], Awaitable[None] | None]


class TextProcessingPipeline:
    def __init__(
        self,
        *,
        processor: TextProcessor,
        summary_state: SummaryState,
        on_english: TextCallback,
        on_summary: TextCallback,
        logger: logging.Logger,
    ) -> None:
        self._processor = processor
        self._summary_state = summary_state
        self._on_english = on_english
        self._on_summary = on_summary
        self._logger = logger
        self._latest_english_sequence = -1
        self._tasks: set[asyncio.Task[None]] = set()

    def submit_turn(self, turn: CompletedTurn) -> asyncio.Task[None]:
        self._summary_state.add_transcript(turn.transcript)
        task = asyncio.create_task(self._translate(turn))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def update_summary_once(self) -> None:
        request = self._summary_state.begin_update()
        if request is None:
            return

        started = monotonic()
        operation_id = f"summary-{int(started * 1000)}"
        input_chars = len(request.context_memory) + sum(
            len(transcript) for transcript in request.transcripts
        )
        log_event(
            self._logger,
            logging.INFO,
            subsystem="gemini",
            event="summary_started",
            operation_id=operation_id,
            input_chars=input_chars,
            buffer_cycles=self._summary_state.pending_cycles,
            dropped_cycles=self._summary_state.dropped_cycles,
        )
        try:
            generated = await self._processor.summarize(
                context_memory=request.context_memory,
                transcripts=request.transcripts,
            )
            summary = generated.summary
            await _invoke_text(self._on_summary, "\n".join(summary.display_lines))
            self._summary_state.succeed_update(
                context_memory=summary.context_memory,
                display_lines=summary.display_lines,
            )
            log_event(
                self._logger,
                logging.INFO,
                subsystem="gemini",
                event="summary_completed",
                operation_id=operation_id,
                duration_ms=round((monotonic() - started) * 1000, 1),
                input_chars=input_chars,
                output_chars=sum(len(line) for line in summary.display_lines),
                input_tokens=generated.input_tokens,
                output_tokens=generated.output_tokens,
                buffer_cycles=self._summary_state.pending_cycles,
                dropped_cycles=self._summary_state.dropped_cycles,
            )
        except asyncio.CancelledError:
            self._summary_state.fail_update()
            raise
        except Exception as exc:
            self._summary_state.fail_update()
            log_exception(
                self._logger,
                subsystem="gemini",
                event="summary_failed",
                exc=exc,
                operation_id=operation_id,
                duration_ms=round((monotonic() - started) * 1000, 1),
                input_chars=input_chars,
                buffer_cycles=self._summary_state.pending_cycles,
                dropped_cycles=self._summary_state.dropped_cycles,
            )

    async def run_summary_loop(self, interval_seconds: float) -> None:
        while True:
            await asyncio.sleep(interval_seconds)
            await self.update_summary_once()

    async def close(self) -> None:
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _translate(self, turn: CompletedTurn) -> None:
        started = monotonic()
        operation_id = f"translation-{turn.sequence}"
        log_event(
            self._logger,
            logging.INFO,
            subsystem="gemini",
            event="translation_started",
            operation_id=operation_id,
            turn_sequence=turn.sequence,
            input_chars=len(turn.transcript),
        )
        try:
            generated = await self._processor.translate(turn.transcript)
            if generated.text and turn.sequence > self._latest_english_sequence:
                self._latest_english_sequence = turn.sequence
                await _invoke_text(self._on_english, generated.text)
            log_event(
                self._logger,
                logging.INFO,
                subsystem="gemini",
                event="translation_completed",
                operation_id=operation_id,
                turn_sequence=turn.sequence,
                duration_ms=round((monotonic() - started) * 1000, 1),
                input_chars=len(turn.transcript),
                output_chars=len(generated.text),
                input_tokens=generated.input_tokens,
                output_tokens=generated.output_tokens,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_exception(
                self._logger,
                subsystem="gemini",
                event="translation_failed",
                exc=exc,
                operation_id=operation_id,
                turn_sequence=turn.sequence,
                duration_ms=round((monotonic() - started) * 1000, 1),
                input_chars=len(turn.transcript),
            )


def parse_summary_response(text: str) -> SummaryResult:
    try:
        payload = json.loads(text)
        context_memory = payload["context_memory"]
        raw_lines = payload["display_lines"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise InvalidSummaryResponse("Summary response is not valid JSON") from exc

    if not isinstance(context_memory, str) or not isinstance(raw_lines, list):
        raise InvalidSummaryResponse("Summary response has invalid field types")
    if len(raw_lines) != 3 or not all(isinstance(line, str) for line in raw_lines):
        raise InvalidSummaryResponse("Summary must contain exactly three lines")

    lines = tuple(line.strip() for line in raw_lines)
    if any(not line or "\n" in line or "\r" in line for line in lines):
        raise InvalidSummaryResponse("Summary lines must be nonempty single lines")
    return SummaryResult(
        context_memory.strip(),
        cast(tuple[str, str, str], lines),
    )


def _generated_text(response: Any) -> GeneratedText:
    text = str(response.text or "").strip()
    usage = getattr(response, "usage_metadata", None)
    return GeneratedText(
        text=text,
        input_tokens=getattr(usage, "prompt_token_count", None),
        output_tokens=getattr(usage, "candidates_token_count", None),
    )


async def _invoke_text(callback: TextCallback, text: str) -> None:
    result = callback(text)
    if inspect.isawaitable(result):
        await result
