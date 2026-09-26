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

from .config import DEFAULT_SUMMARY_MAX_CHARS, MIN_SUMMARY_MAX_CHARS
from .diagnostics import log_event, log_exception
from .state import CompletedTurn, SummaryState


GEMINI_MODEL = "gemini-3.1-flash-lite"
SUMMARY_TRUNCATION_SUFFIX = "(…)"


class InvalidSummaryResponse(ValueError):
    """概要応答がOBSへ表示できる構造ではない。"""


@dataclass(frozen=True, slots=True)
class SummaryResult:
    display_lines: tuple[str, str, str]
    model_line_chars: tuple[int, int, int] | None = None


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
    def __init__(
        self, client: Any, *, summary_max_chars: int = DEFAULT_SUMMARY_MAX_CHARS
    ) -> None:
        if summary_max_chars < MIN_SUMMARY_MAX_CHARS:
            raise ValueError("summary_max_chars must be at least 4")
        self._models = client.aio.models
        self._summary_max_chars = summary_max_chars

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
        transcripts: tuple[str, ...],
    ) -> GeneratedSummary:
        target_chars = min(
            round(self._summary_max_chars * 0.8), self._summary_max_chars - 2
        )
        input_payload = json.dumps(
            {"recent_transcripts": transcripts},
            ensure_ascii=False,
        )
        response = await self._models.generate_content(
            model=GEMINI_MODEL,
            contents=input_payload,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "日本語の一人語り配信を要約する。recent_transcriptsは"
                    "直近5分間に確定した発言を古い順に並べたもの。"
                    "この発言に含まれる内容だけを使い、"
                    "発言にない配信内容、予定、募集、視聴者への呼びかけを加えない。"
                    "例えば「今日はピアノを弾きます」だけなら、演奏することだけを"
                    "1行に書き、リクエスト募集は書かない。"
                    "過去に終えた行は過去形、今している行は進行中の形、"
                    "これからする行は予定や意向を示す形で述語まで書く。"
                    "発言の時点と時制を変えず、条件付きの予定を断定しない。"
                    "後の発言で話題や予定が変わった場合は"
                    "新しい発言を優先する。display_linesはOBSへ表示する日本語で、"
                    "見出しや単語の羅列ではなく、発言した対象、動作、順序が"
                    "伝わる文にする。まず同じ対象について述べた内容を"
                    "1つの話題にまとめ、話題ごとに1行で最大3行にする。"
                    "例えば果物としてリンゴ、バナナ、ミカンを挙げた場合は"
                    "種類を1行に列挙し、果物の行を増やさない。"
                    "同じ話題の列挙が1行の上限内に収まる場合は、"
                    "挙げられた種類を省かず1行にまとめる。"
                    "列挙された固有の種類は「など」「等」で省略せず、"
                    "説明語を短くして全種類を先に残す。"
                    "同じ話題を分割して3行を埋めない。"
                    f"各行の最大文字数は{self._summary_max_chars - 2}文字とし、"
                    "必ずこの上限以内に収める。文字数はUTF-8のバイト数や"
                    "英単語数ではなく、漢字、ひらがな、カタカナ、英数字、"
                    "記号をそれぞれ1文字として数える。"
                    f"情報が十分な行は{target_chars}文字前後を目指す。"
                    "短い要点だけに圧縮せず、発言された時点、対象、動作に加え、"
                    "方法または理由などの関連情報を一つ程度含める。"
                    "目安を超える細部は重要度に応じて省く。"
                    "時制と具体的な内容を残す。情報が足りない行は無理に伸ばさない。"
                    "文字数や行数を満たすために内容を水増し・反復しない。"
                    f"返答前に各行を確認し、{self._summary_max_chars - 2}文字を"
                    "超えた行は、事実と時制を保って短く書き直す。"
                    "配列は必ず3要素とする。表示する話題が2件なら3行目は空文字、"
                    "1件なら2行目と3行目は空文字にする。"
                ),
                temperature=0.0,
                max_output_tokens=512,
                response_mime_type="application/json",
                response_json_schema={
                    "type": "object",
                    "properties": {
                        "display_lines": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 3,
                        },
                    },
                    "required": ["display_lines"],
                    "additionalProperties": False,
                },
            ),
        )
        generated = _generated_text(response)
        return GeneratedSummary(
            summary=parse_summary_response(
                generated.text, max_chars=self._summary_max_chars
            ),
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
        )


class TextProcessor(Protocol):
    async def translate(self, transcript: str) -> GeneratedText: ...

    async def summarize(
        self,
        *,
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
        summary_interval_seconds: float,
    ) -> None:
        if summary_interval_seconds <= 0:
            raise ValueError("summary_interval_seconds must be positive")
        self._processor = processor
        self._summary_state = summary_state
        self._on_english = on_english
        self._on_summary = on_summary
        self._logger = logger
        self._summary_interval_seconds = summary_interval_seconds
        self._last_summary_completed_at: float | None = None
        self._summary_task: asyncio.Task[None] | None = None
        self._latest_english_sequence = -1
        self._tasks: set[asyncio.Task[None]] = set()

    def submit_turn(self, turn: CompletedTurn) -> asyncio.Task[None]:
        included = self._summary_state.add_transcript(
            turn.transcript, completed_at=turn.completed_at
        )
        task = asyncio.create_task(self._translate(turn))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        if included:
            self._maybe_start_summary()
        return task

    def _maybe_start_summary(self) -> None:
        if self._summary_task is not None and not self._summary_task.done():
            return
        if (
            self._last_summary_completed_at is not None
            and monotonic() - self._last_summary_completed_at
            < self._summary_interval_seconds
        ):
            return

        task = asyncio.create_task(self.update_summary_once())
        self._summary_task = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def update_summary_once(self) -> None:
        request = self._summary_state.begin_update()
        if request is None:
            return

        started = monotonic()
        operation_id = f"summary-{int(started * 1000)}"
        input_chars = sum(len(transcript) for transcript in request.transcripts)
        log_event(
            self._logger,
            logging.INFO,
            subsystem="gemini",
            event="summary_started",
            operation_id=operation_id,
            input_chars=input_chars,
            transcript_count=len(request.transcripts),
        )
        try:
            generated = await self._processor.summarize(
                transcripts=request.transcripts,
            )
            summary = generated.summary
            await _invoke_text(
                self._on_summary,
                "\n".join(line for line in summary.display_lines if line),
            )
            self._summary_state.succeed_update(display_lines=summary.display_lines)
            self._last_summary_completed_at = monotonic()
            log_event(
                self._logger,
                logging.INFO,
                subsystem="gemini",
                event="summary_completed",
                operation_id=operation_id,
                duration_ms=round((monotonic() - started) * 1000, 1),
                input_chars=input_chars,
                output_chars=sum(len(line) for line in summary.display_lines),
                model_line_chars=summary.model_line_chars,
                summary_line_chars=[len(line) for line in summary.display_lines],
                input_tokens=generated.input_tokens,
                output_tokens=generated.output_tokens,
                transcript_count=len(request.transcripts),
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
                transcript_count=len(request.transcripts),
            )

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


def parse_summary_response(
    text: str, *, max_chars: int = DEFAULT_SUMMARY_MAX_CHARS
) -> SummaryResult:
    if max_chars < MIN_SUMMARY_MAX_CHARS:
        raise ValueError("max_chars must be at least 4")
    try:
        payload = json.loads(text)
        raw_lines = payload["display_lines"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise InvalidSummaryResponse("Summary response is not valid JSON") from exc

    if not isinstance(raw_lines, list):
        raise InvalidSummaryResponse("Summary response has invalid field types")
    if len(raw_lines) != 3 or not all(isinstance(line, str) for line in raw_lines):
        raise InvalidSummaryResponse("Summary must contain exactly three lines")

    lines = tuple(line.strip() for line in raw_lines)
    if any("\n" in line or "\r" in line for line in lines):
        raise InvalidSummaryResponse("Summary lines must each fit on one line")
    if not lines[0] or any(
        not line and any(lines[index + 1 :]) for index, line in enumerate(lines)
    ):
        raise InvalidSummaryResponse("Summary lines must be nonempty before empty lines")
    display_lines = tuple(
        line[: max_chars - len(SUMMARY_TRUNCATION_SUFFIX)]
        + SUMMARY_TRUNCATION_SUFFIX
        if len(line) > max_chars
        else line
        for line in lines
    )
    return SummaryResult(
        cast(tuple[str, str, str], display_lines),
        cast(tuple[int, int, int], tuple(len(line) for line in lines)),
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
