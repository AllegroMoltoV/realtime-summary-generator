from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from time import monotonic
from typing import Any

from websockets.asyncio.client import connect

from .audio import AudioChunkQueue
from .diagnostics import log_event
from .state import CompletedTurn, TurnTracker


CaptionCallback = Callable[[str], Awaitable[None] | None]
CompletedCallback = Callable[[CompletedTurn], Awaitable[None] | None]
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime?intent=transcription"


class RealtimeApiError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(f"OpenAI Realtime API error: {code}")
        self.code = code


def build_audio_append(pcm: bytes) -> dict[str, str]:
    return {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(pcm).decode("ascii"),
    }


def build_session_update() -> dict[str, Any]:
    return {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {
                        "model": "gpt-transcribe",
                        "prompt": "日本語の一人語り。固有名詞、数字、英単語を正確に転写する。",
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                    },
                }
            },
        },
    }


class TranscriptionEventRouter:
    def __init__(
        self,
        *,
        on_caption: CaptionCallback,
        on_completed: CompletedCallback,
    ) -> None:
        self._on_caption = on_caption
        self._on_completed = on_completed
        self._turns = TurnTracker()
        self._partials: dict[str, str] = {}
        self._latest_displayed_sequence = -1

    async def handle(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "input_audio_buffer.committed":
            self._turns.commit(
                str(event["item_id"]),
                previous_item_id=event.get("previous_item_id"),
            )
            return

        if event_type == "conversation.item.input_audio_transcription.delta":
            item_id = str(event["item_id"])
            sequence = self._turns.sequence_for(item_id)
            if sequence is None or sequence < self._latest_displayed_sequence:
                return
            partial = self._partials.get(item_id, "") + str(event.get("delta", ""))
            self._partials[item_id] = partial
            self._latest_displayed_sequence = sequence
            await _invoke(self._on_caption, partial)
            return

        if event_type == "conversation.item.input_audio_transcription.completed":
            item_id = str(event["item_id"])
            released = self._turns.complete(
                item_id,
                str(event.get("transcript", "")).strip(),
            )
            self._partials.pop(item_id, None)
            await self._deliver(released)
            return

        if event_type == "conversation.item.input_audio_transcription.failed":
            item_id = str(event["item_id"])
            self._partials.pop(item_id, None)
            await self._deliver(self._turns.fail(item_id))

    async def _deliver(self, turns: list[CompletedTurn]) -> None:
        for turn in turns:
            if not turn.transcript:
                continue
            self._latest_displayed_sequence = max(
                self._latest_displayed_sequence, turn.sequence
            )
            await _invoke(self._on_caption, turn.transcript)
            await _invoke(self._on_completed, turn)


class RealtimeTranscriber:
    def __init__(
        self,
        *,
        api_key: str,
        router: TranscriptionEventRouter,
        logger: logging.Logger,
    ) -> None:
        self._api_key = api_key
        self._router = router
        self._logger = logger
        self._committed_at: dict[str, float] = {}
        self._first_delta_seen: set[str] = set()

    async def run(self, chunks: AudioChunkQueue) -> None:
        log_event(
            self._logger,
            logging.INFO,
            subsystem="openai",
            event="connecting",
        )
        async with connect(
            OPENAI_REALTIME_URL,
            additional_headers={"Authorization": f"Bearer {self._api_key}"},
            open_timeout=10,
            close_timeout=5,
            max_size=4 * 1024 * 1024,
        ) as websocket:
            log_event(
                self._logger,
                logging.INFO,
                subsystem="openai",
                event="connected",
            )
            await websocket.send(json.dumps(build_session_update(), ensure_ascii=False))
            sender = asyncio.create_task(self._send_audio(websocket, chunks))
            try:
                async for message in websocket:
                    event = json.loads(message)
                    self._log_server_event(event)
                    error = event.get("error") if event.get("type") == "error" else None
                    if isinstance(error, dict):
                        raise RealtimeApiError(
                            str(error.get("code") or error.get("type") or "unknown")
                        )
                    await self._router.handle(event)
            finally:
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
                log_event(
                    self._logger,
                    logging.INFO,
                    subsystem="openai",
                    event="disconnected",
                )

    async def _send_audio(self, websocket: Any, chunks: AudioChunkQueue) -> None:
        sent = 0
        while True:
            chunk = await chunks.get()
            await websocket.send(json.dumps(build_audio_append(chunk)))
            sent += 1
            if sent % 100 == 0:
                log_event(
                    self._logger,
                    logging.DEBUG,
                    subsystem="audio",
                    event="queue_status",
                    queue_depth=chunks.qsize(),
                    audio_chunks_dropped=chunks.dropped_chunks,
                )

    def _log_server_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", "unknown"))
        item_id = event.get("item_id")
        now = monotonic()
        fields: dict[str, object] = {}
        if item_id is not None:
            fields["item_id"] = str(item_id)
        if event.get("previous_item_id") is not None:
            fields["previous_item_id"] = str(event["previous_item_id"])

        if event_type == "input_audio_buffer.committed" and item_id is not None:
            self._committed_at[str(item_id)] = now
        elif (
            event_type == "conversation.item.input_audio_transcription.delta"
            and item_id is not None
            and str(item_id) not in self._first_delta_seen
        ):
            self._first_delta_seen.add(str(item_id))
            started = self._committed_at.get(str(item_id))
            if started is not None:
                fields["duration_ms"] = round((now - started) * 1000, 1)
        elif (
            event_type
            in {
                "conversation.item.input_audio_transcription.completed",
                "conversation.item.input_audio_transcription.failed",
            }
            and item_id is not None
        ):
            started = self._committed_at.pop(str(item_id), None)
            self._first_delta_seen.discard(str(item_id))
            if started is not None:
                fields["duration_ms"] = round((now - started) * 1000, 1)

        if event_type in {
            "session.created",
            "session.updated",
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
            "input_audio_buffer.committed",
            "conversation.item.input_audio_transcription.delta",
            "conversation.item.input_audio_transcription.completed",
            "conversation.item.input_audio_transcription.failed",
            "error",
        }:
            log_event(
                self._logger,
                logging.DEBUG,
                subsystem="openai",
                event=event_type,
                **fields,
            )


async def _invoke(callback: Callable[[Any], Any], value: Any) -> None:
    result = callback(value)
    if inspect.isawaitable(result):
        await result
