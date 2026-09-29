from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import monotonic
from typing import Any

from websockets.asyncio.client import connect, process_exception
from websockets.exceptions import ConnectionClosed

from .audio import AudioChunkQueue
from .diagnostics import log_event, log_exception
from .state import CompletedTurn, TurnTracker


CaptionCallback = Callable[[str], Awaitable[None] | None]
CompletedCallback = Callable[[CompletedTurn], Awaitable[None] | None]
ConnectFactory = Callable[..., Any]
Sleep = Callable[[float], Awaitable[None]]
Jitter = Callable[[], float]
Clock = Callable[[], float]
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
PING_INTERVAL_SECONDS = 20
PING_TIMEOUT_SECONDS = 20


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_retries: int = 5
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 16.0
    stable_connection_seconds: float = 60.0

    def delay_seconds(self, attempt: int, *, jitter: float) -> float:
        ceiling = min(
            self.max_delay_seconds,
            self.base_delay_seconds * (2 ** (attempt - 1)),
        )
        return ceiling * (0.5 + (0.5 * jitter))


class RealtimeApiError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(f"OpenAI Realtime API error: {code}")
        self.code = code


class RealtimeConnectionEnded(ConnectionError):
    def __init__(self, code: int | None, reason: str | None) -> None:
        detail = f"code={code}, reason={reason or 'none'}"
        super().__init__(f"OpenAI Realtime connection ended unexpectedly ({detail})")
        self.code = code
        self.reason = reason


def is_retryable_connection_error(exc: Exception) -> bool:
    return isinstance(exc, (ConnectionClosed, RealtimeConnectionEnded)) or (
        process_exception(exc) is None
    )


def connection_close_fields(exc: Exception) -> dict[str, object]:
    if not isinstance(exc, (ConnectionClosed, RealtimeConnectionEnded)):
        return {}
    received = exc.rcvd if isinstance(exc, ConnectionClosed) else None
    sent = exc.sent if isinstance(exc, ConnectionClosed) else None
    if isinstance(exc, RealtimeConnectionEnded):
        received_code = exc.code
        received_reason = exc.reason
    else:
        received_code = getattr(received, "code", None)
        received_reason = getattr(received, "reason", None)
    return {
        "error_code": received_code if received_code is not None else 1006,
        "received_close_code": received_code,
        "received_close_reason": received_reason,
        "sent_close_code": getattr(sent, "code", None),
        "sent_close_reason": getattr(sent, "reason", None),
    }


def build_audio_append(pcm: bytes) -> dict[str, str]:
    return {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(pcm).decode("ascii"),
    }


def build_session_update(*, keywords: tuple[str, ...] = ()) -> dict[str, Any]:
    transcription: dict[str, Any] = {
        "model": "gpt-transcribe",
        "prompt": "日本語の一人語り。固有名詞、数字、英単語を正確に転写する。",
    }
    if keywords:
        transcription["keywords"] = list(keywords)
    return {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": transcription,
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

    def reset_session(self) -> tuple[int, int]:
        pending_items = self._turns.pending_count
        partial_captions = len(self._partials)
        self._turns = TurnTracker()
        self._partials.clear()
        self._latest_displayed_sequence = -1
        return pending_items, partial_captions

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
        keywords: tuple[str, ...] = (),
        connect_factory: ConnectFactory = connect,
        sleep: Sleep = asyncio.sleep,
        jitter: Jitter = random.random,
        clock: Clock = monotonic,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._api_key = api_key
        self._router = router
        self._logger = logger
        self._keywords = keywords
        self._connect_factory = connect_factory
        self._sleep = sleep
        self._jitter = jitter
        self._clock = clock
        self._retry_policy = retry_policy or RetryPolicy()
        self._committed_at: dict[str, float] = {}
        self._first_delta_seen: set[str] = set()

    async def run(self, chunks: AudioChunkQueue) -> None:
        retry_count = 0
        while True:
            attempt_started = self._clock()
            try:
                await self._run_session(chunks, retry_count=retry_count)
                return
            except Exception as exc:
                if not is_retryable_connection_error(exc):
                    raise
                connection_duration = self._clock() - attempt_started
                if (
                    connection_duration
                    >= self._retry_policy.stable_connection_seconds
                ):
                    retry_count = 0
                log_exception(
                    self._logger,
                    subsystem="openai",
                    event="connection_lost",
                    exc=exc,
                    retry_count=retry_count,
                    duration_ms=round(connection_duration * 1000, 1),
                    **connection_close_fields(exc),
                )
                if retry_count >= self._retry_policy.max_retries:
                    log_event(
                        self._logger,
                        logging.ERROR,
                        subsystem="openai",
                        event="reconnect_exhausted",
                        exception_type=type(exc).__name__,
                        retry_count=retry_count,
                        max_retries=self._retry_policy.max_retries,
                        **connection_close_fields(exc),
                    )
                    raise
                retry_count += 1
                discarded = chunks.discard_pending()
                pending_items, partial_captions = self._router.reset_session()
                timing_items = len(
                    self._committed_at.keys() | self._first_delta_seen
                )
                self._committed_at.clear()
                self._first_delta_seen.clear()
                delay = self._retry_policy.delay_seconds(
                    retry_count,
                    jitter=self._jitter(),
                )
                log_event(
                    self._logger,
                    logging.WARNING,
                    subsystem="openai",
                    event="reconnect_scheduled",
                    retry_count=retry_count,
                    retry_delay_seconds=round(delay, 3),
                    audio_chunks_discarded=discarded,
                    transcription_items_discarded=pending_items,
                    partial_captions_discarded=partial_captions,
                    timing_items_discarded=timing_items,
                )
                await self._sleep(delay)

    async def _run_session(
        self,
        chunks: AudioChunkQueue,
        *,
        retry_count: int,
    ) -> None:
        log_event(
            self._logger,
            logging.INFO,
            subsystem="openai",
            event="connecting",
            retry_count=retry_count,
            max_retries=self._retry_policy.max_retries,
            retry_base_delay_seconds=self._retry_policy.base_delay_seconds,
            retry_max_delay_seconds=self._retry_policy.max_delay_seconds,
            stable_connection_seconds=self._retry_policy.stable_connection_seconds,
            ping_interval_seconds=PING_INTERVAL_SECONDS,
            ping_timeout_seconds=PING_TIMEOUT_SECONDS,
        )
        async with self._connect_factory(
            OPENAI_REALTIME_URL,
            additional_headers={"Authorization": f"Bearer {self._api_key}"},
            open_timeout=10,
            close_timeout=5,
            max_size=4 * 1024 * 1024,
            ping_interval=PING_INTERVAL_SECONDS,
            ping_timeout=PING_TIMEOUT_SECONDS,
        ) as websocket:
            log_event(
                self._logger,
                logging.INFO,
                subsystem="openai",
                event="connected",
                retry_count=retry_count,
            )
            await websocket.send(
                json.dumps(
                    build_session_update(keywords=self._keywords), ensure_ascii=False
                )
            )
            sender = asyncio.create_task(self._send_audio(websocket, chunks))
            receiver = asyncio.create_task(self._receive_events(websocket))
            try:
                done, _pending = await asyncio.wait(
                    {sender, receiver},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    exception = task.exception()
                    if exception is not None:
                        raise exception
                raise RealtimeConnectionEnded(
                    getattr(websocket, "close_code", None),
                    getattr(websocket, "close_reason", None),
                )
            finally:
                sender.cancel()
                receiver.cancel()
                await asyncio.gather(sender, receiver, return_exceptions=True)
                log_event(
                    self._logger,
                    logging.INFO,
                    subsystem="openai",
                    event="disconnected",
                    retry_count=retry_count,
                )

    async def _receive_events(self, websocket: Any) -> None:
        async for message in websocket:
            event = json.loads(message)
            self._log_server_event(event)
            error = event.get("error") if event.get("type") == "error" else None
            if isinstance(error, dict):
                raise RealtimeApiError(
                    str(error.get("code") or error.get("type") or "unknown")
                )
            await self._router.handle(event)

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
