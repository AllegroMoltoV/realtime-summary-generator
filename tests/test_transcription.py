from __future__ import annotations

import asyncio
import json
from unittest.mock import Mock

import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from realtime_summary.audio import AudioChunkQueue
from realtime_summary.diagnostics import close_diagnostics, configure_diagnostics
from realtime_summary.state import CompletedTurn
from realtime_summary.transcription import (
    OPENAI_REALTIME_URL,
    RealtimeApiError,
    RealtimeConnectionEnded,
    RealtimeTranscriber,
    RetryPolicy,
    TranscriptionEventRouter,
    build_audio_append,
    build_session_update,
)


class _FakeWebSocket:
    def __init__(self, events: list[str | BaseException]) -> None:
        self._events = events

    async def __aenter__(self) -> _FakeWebSocket:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def send(self, _message: str) -> None:
        return None

    def __aiter__(self) -> _FakeWebSocket:
        return self

    async def __anext__(self) -> str:
        if not self._events:
            raise StopAsyncIteration
        event = self._events.pop(0)
        if isinstance(event, BaseException):
            raise event
        return event


def _router() -> TranscriptionEventRouter:
    return TranscriptionEventRouter(
        on_caption=lambda _text: None,
        on_completed=lambda _turn: None,
    )


def _logger() -> Mock:
    logger = Mock()
    logger.name = "realtime_summary.test"
    return logger


def _read_records(path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_realtime_url_selects_a_transcription_session() -> None:
    assert OPENAI_REALTIME_URL == (
        "wss://api.openai.com/v1/realtime?intent=transcription"
    )


def test_retry_policy_uses_capped_exponential_backoff_with_jitter() -> None:
    policy = RetryPolicy(
        max_retries=5,
        base_delay_seconds=1.0,
        max_delay_seconds=8.0,
    )

    assert policy.delay_seconds(1, jitter=0.0) == 0.5
    assert policy.delay_seconds(2, jitter=1.0) == 2.0
    assert policy.delay_seconds(5, jitter=1.0) == 8.0


@pytest.mark.asyncio
async def test_router_delivers_completed_transcripts_in_commit_order() -> None:
    displayed: list[str] = []
    completed: list[CompletedTurn] = []
    router = TranscriptionEventRouter(
        on_caption=displayed.append,
        on_completed=completed.append,
    )

    await router.handle(
        {
            "type": "input_audio_buffer.committed",
            "item_id": "item-a",
            "previous_item_id": None,
        }
    )
    await router.handle(
        {
            "type": "input_audio_buffer.committed",
            "item_id": "item-b",
            "previous_item_id": "item-a",
        }
    )
    await router.handle(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item-b",
            "transcript": "二番目",
        }
    )
    await router.handle(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item-a",
            "transcript": "最初",
        }
    )

    assert completed == [
        CompletedTurn(0, "item-a", "最初", 0.0),
        CompletedTurn(1, "item-b", "二番目", 0.0),
    ]
    assert displayed[-1] == "二番目"


@pytest.mark.asyncio
async def test_router_accumulates_partial_caption_deltas() -> None:
    displayed: list[str] = []
    router = TranscriptionEventRouter(
        on_caption=displayed.append,
        on_completed=lambda _turn: None,
    )
    await router.handle(
        {
            "type": "input_audio_buffer.committed",
            "item_id": "item-a",
            "previous_item_id": None,
        }
    )

    await router.handle(
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "item-a",
            "delta": "リアル",
        }
    )
    await router.handle(
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "item-a",
            "delta": "タイム",
        }
    )

    assert displayed == ["リアル", "リアルタイム"]


@pytest.mark.asyncio
async def test_failed_turn_does_not_block_the_following_turn() -> None:
    completed: list[CompletedTurn] = []
    router = TranscriptionEventRouter(
        on_caption=lambda _text: None,
        on_completed=completed.append,
    )
    await router.handle(
        {
            "type": "input_audio_buffer.committed",
            "item_id": "item-a",
            "previous_item_id": None,
        }
    )
    await router.handle(
        {
            "type": "input_audio_buffer.committed",
            "item_id": "item-b",
            "previous_item_id": "item-a",
        }
    )
    await router.handle(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item-b",
            "transcript": "後続の発話",
        }
    )

    await router.handle(
        {
            "type": "conversation.item.input_audio_transcription.failed",
            "item_id": "item-a",
        }
    )

    assert completed == [CompletedTurn(1, "item-b", "後続の発話", 0.0)]


def test_session_uses_pcm24k_transcription_and_server_vad() -> None:
    event = build_session_update()

    assert event["type"] == "session.update"
    audio_input = event["session"]["audio"]["input"]
    assert event["session"]["type"] == "transcription"
    assert audio_input["format"] == {"type": "audio/pcm", "rate": 24000}
    assert audio_input["transcription"]["model"] == "gpt-transcribe"
    assert audio_input["turn_detection"] == {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 500,
    }


def test_audio_append_encodes_pcm_without_writing_a_file() -> None:
    assert build_audio_append(b"\x00\x01\x02") == {
        "type": "input_audio_buffer.append",
        "audio": "AAEC",
    }


@pytest.mark.asyncio
async def test_transcriber_reconnects_after_an_abnormal_close() -> None:
    connections = [
        _FakeWebSocket([ConnectionClosedError(None, None)]),
        _FakeWebSocket(
            [json.dumps({"type": "error", "error": {"code": "invalid_request"}})]
        ),
    ]
    connect_calls = 0

    def connect_factory(*_args: object, **_kwargs: object) -> _FakeWebSocket:
        nonlocal connect_calls
        connection = connections[connect_calls]
        connect_calls += 1
        return connection

    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    transcriber = RealtimeTranscriber(
        api_key="test-key",
        router=_router(),
        logger=_logger(),
        connect_factory=connect_factory,
        sleep=sleep,
        jitter=lambda: 0.0,
    )

    with pytest.raises(RealtimeApiError, match="invalid_request"):
        await transcriber.run(AudioChunkQueue())

    assert connect_calls == 2
    assert delays == [0.5]


@pytest.mark.asyncio
async def test_reconnect_starts_with_fresh_turn_order_state() -> None:
    completed: list[CompletedTurn] = []
    connections = [
        _FakeWebSocket(
            [
                json.dumps(
                    {
                        "type": "input_audio_buffer.committed",
                        "item_id": "old-item",
                        "previous_item_id": None,
                    }
                ),
                ConnectionClosedError(None, None),
            ]
        ),
        _FakeWebSocket(
            [
                json.dumps(
                    {
                        "type": "input_audio_buffer.committed",
                        "item_id": "new-item",
                        "previous_item_id": None,
                    }
                ),
                json.dumps(
                    {
                        "type": (
                            "conversation.item.input_audio_transcription.completed"
                        ),
                        "item_id": "new-item",
                        "transcript": "再接続後",
                    }
                ),
                json.dumps(
                    {"type": "error", "error": {"code": "invalid_request"}}
                ),
            ]
        ),
    ]

    def connect_factory(*_args: object, **_kwargs: object) -> _FakeWebSocket:
        return connections.pop(0)

    async def sleep(_delay: float) -> None:
        return None

    router = TranscriptionEventRouter(
        on_caption=lambda _text: None,
        on_completed=completed.append,
    )
    transcriber = RealtimeTranscriber(
        api_key="test-key",
        router=router,
        logger=_logger(),
        connect_factory=connect_factory,
        sleep=sleep,
    )

    with pytest.raises(RealtimeApiError, match="invalid_request"):
        await transcriber.run(AudioChunkQueue())

    assert completed == [CompletedTurn(0, "new-item", "再接続後", 0.0)]


@pytest.mark.asyncio
async def test_transcriber_limits_retries_after_clean_remote_closes() -> None:
    connect_calls = 0

    def connect_factory(*_args: object, **_kwargs: object) -> _FakeWebSocket:
        nonlocal connect_calls
        connect_calls += 1
        return _FakeWebSocket([])

    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    transcriber = RealtimeTranscriber(
        api_key="test-key",
        router=_router(),
        logger=_logger(),
        connect_factory=connect_factory,
        sleep=sleep,
        jitter=lambda: 0.0,
        retry_policy=RetryPolicy(max_retries=2),
    )

    with pytest.raises(RealtimeConnectionEnded):
        await transcriber.run(AudioChunkQueue())

    assert connect_calls == 3
    assert delays == [0.5, 1.0]


@pytest.mark.asyncio
async def test_transcriber_retries_a_transient_connection_failure() -> None:
    connect_calls = 0

    def connect_factory(*_args: object, **_kwargs: object) -> _FakeWebSocket:
        nonlocal connect_calls
        connect_calls += 1
        if connect_calls == 1:
            raise OSError("temporary network failure")
        return _FakeWebSocket(
            [json.dumps({"type": "error", "error": {"code": "invalid_request"}})]
        )

    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    transcriber = RealtimeTranscriber(
        api_key="test-key",
        router=_router(),
        logger=_logger(),
        connect_factory=connect_factory,
        sleep=sleep,
        jitter=lambda: 0.0,
    )

    with pytest.raises(RealtimeApiError, match="invalid_request"):
        await transcriber.run(AudioChunkQueue())

    assert connect_calls == 2
    assert delays == [0.5]


@pytest.mark.asyncio
async def test_transcriber_logs_websocket_close_details(tmp_path) -> None:
    connections = [
        _FakeWebSocket(
            [
                json.dumps(
                    {
                        "type": "input_audio_buffer.committed",
                        "item_id": "pending-item",
                        "previous_item_id": None,
                    }
                ),
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "item_id": "pending-item",
                        "delta": "途中",
                    }
                ),
                ConnectionClosedError(None, Close(1011, "keepalive ping timeout")),
            ]
        ),
        _FakeWebSocket(
            [json.dumps({"type": "error", "error": {"code": "invalid_request"}})]
        ),
    ]

    def connect_factory(*_args: object, **_kwargs: object) -> _FakeWebSocket:
        return connections.pop(0)

    async def sleep(_delay: float) -> None:
        return None

    logger = configure_diagnostics(tmp_path, run_id="run-close", level="DEBUG")
    transcriber = RealtimeTranscriber(
        api_key="test-key",
        router=_router(),
        logger=logger,
        connect_factory=connect_factory,
        sleep=sleep,
        jitter=lambda: 0.0,
    )

    with pytest.raises(RealtimeApiError):
        await transcriber.run(AudioChunkQueue())
    close_diagnostics(logger)

    records = _read_records(tmp_path / "realtime-summary.jsonl")
    failure = next(record for record in records if record["event"] == "connection_lost")
    assert failure["exception_type"] == "ConnectionClosedError"
    assert failure["error_code"] == 1006
    assert failure["received_close_code"] is None
    assert failure["received_close_reason"] is None
    assert failure["sent_close_code"] == 1011
    assert failure["sent_close_reason"] == "keepalive ping timeout"
    assert failure["duration_ms"] >= 0
    connecting = [record for record in records if record["event"] == "connecting"]
    connected = [record for record in records if record["event"] == "connected"]
    assert [record["retry_count"] for record in connecting] == [0, 1]
    assert [record["retry_count"] for record in connected] == [0, 1]
    assert connecting[0]["ping_interval_seconds"] == 20
    assert connecting[0]["ping_timeout_seconds"] == 20
    assert connecting[0]["max_retries"] == 5
    assert connecting[0]["retry_base_delay_seconds"] == 1.0
    assert connecting[0]["retry_max_delay_seconds"] == 16.0
    assert connecting[0]["stable_connection_seconds"] == 60.0
    scheduled = next(
        record for record in records if record["event"] == "reconnect_scheduled"
    )
    assert scheduled["transcription_items_discarded"] == 1
    assert scheduled["partial_captions_discarded"] == 1
    assert scheduled["timing_items_discarded"] == 1


@pytest.mark.asyncio
async def test_transcriber_logs_discarded_audio_and_retry_exhaustion(tmp_path) -> None:
    def connect_factory(*_args: object, **_kwargs: object) -> _FakeWebSocket:
        raise OSError("network unavailable")

    async def sleep(_delay: float) -> None:
        return None

    chunks = AudioChunkQueue()
    chunks.put_nowait(b"stale-1")
    chunks.put_nowait(b"stale-2")
    logger = configure_diagnostics(tmp_path, run_id="run-exhausted", level="DEBUG")
    transcriber = RealtimeTranscriber(
        api_key="test-key",
        router=_router(),
        logger=logger,
        connect_factory=connect_factory,
        sleep=sleep,
        jitter=lambda: 0.0,
        retry_policy=RetryPolicy(max_retries=1),
    )

    with pytest.raises(OSError, match="network unavailable"):
        await transcriber.run(chunks)
    close_diagnostics(logger)

    records = _read_records(tmp_path / "realtime-summary.jsonl")
    scheduled = next(
        record for record in records if record["event"] == "reconnect_scheduled"
    )
    assert scheduled["retry_count"] == 1
    assert scheduled["retry_delay_seconds"] == 0.5
    assert scheduled["audio_chunks_discarded"] == 2
    exhausted = next(
        record for record in records if record["event"] == "reconnect_exhausted"
    )
    assert exhausted["retry_count"] == 1
    assert exhausted["max_retries"] == 1


@pytest.mark.asyncio
async def test_stable_connection_resets_the_consecutive_retry_limit() -> None:
    connections = [
        _FakeWebSocket([ConnectionClosedError(None, None)]),
        _FakeWebSocket([ConnectionClosedError(None, None)]),
        _FakeWebSocket(
            [json.dumps({"type": "error", "error": {"code": "invalid_request"}})]
        ),
    ]

    def connect_factory(*_args: object, **_kwargs: object) -> _FakeWebSocket:
        return connections.pop(0)

    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    clock_values = iter([0.0, 1.0, 2.0, 63.0, 64.0])
    transcriber = RealtimeTranscriber(
        api_key="test-key",
        router=_router(),
        logger=_logger(),
        connect_factory=connect_factory,
        sleep=sleep,
        jitter=lambda: 0.0,
        clock=lambda: next(clock_values),
        retry_policy=RetryPolicy(
            max_retries=1,
            stable_connection_seconds=60.0,
        ),
    )

    with pytest.raises(RealtimeApiError):
        await transcriber.run(AudioChunkQueue())

    assert delays == [0.5, 0.5]


@pytest.mark.asyncio
async def test_audio_sender_failure_triggers_reconnection() -> None:
    class SendFailingWebSocket(_FakeWebSocket):
        def __init__(self) -> None:
            super().__init__([])
            self._send_count = 0

        async def send(self, _message: str) -> None:
            self._send_count += 1
            if self._send_count > 1:
                raise ConnectionClosedError(None, None)

        async def __anext__(self) -> str:
            await asyncio.Future()
            raise AssertionError("unreachable")

    connections = [
        SendFailingWebSocket(),
        _FakeWebSocket(
            [json.dumps({"type": "error", "error": {"code": "invalid_request"}})]
        ),
    ]

    def connect_factory(*_args: object, **_kwargs: object) -> _FakeWebSocket:
        return connections.pop(0)

    async def sleep(_delay: float) -> None:
        return None

    chunks = AudioChunkQueue()
    chunks.put_nowait(b"audio")
    transcriber = RealtimeTranscriber(
        api_key="test-key",
        router=_router(),
        logger=_logger(),
        connect_factory=connect_factory,
        sleep=sleep,
        jitter=lambda: 0.0,
    )

    with pytest.raises(RealtimeApiError):
        await asyncio.wait_for(transcriber.run(chunks), timeout=0.2)

    assert not connections


@pytest.mark.asyncio
async def test_cancellation_is_not_retried() -> None:
    connect_calls = 0

    def connect_factory(*_args: object, **_kwargs: object) -> _FakeWebSocket:
        nonlocal connect_calls
        connect_calls += 1
        raise asyncio.CancelledError

    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    transcriber = RealtimeTranscriber(
        api_key="test-key",
        router=_router(),
        logger=_logger(),
        connect_factory=connect_factory,
        sleep=sleep,
    )

    with pytest.raises(asyncio.CancelledError):
        await transcriber.run(AudioChunkQueue())

    assert connect_calls == 1
    assert delays == []
