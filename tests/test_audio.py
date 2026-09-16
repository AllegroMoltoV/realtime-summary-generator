import asyncio
from unittest.mock import Mock

import pytest

from realtime_summary.audio import AudioChunkQueue, MicrophoneInput


@pytest.mark.asyncio
async def test_full_audio_queue_drops_oldest_chunk() -> None:
    chunks = AudioChunkQueue(max_chunks=2)
    chunks.put_nowait(b"oldest")
    chunks.put_nowait(b"middle")

    chunks.put_nowait(b"newest")

    assert chunks.dropped_chunks == 1
    assert await chunks.get() == b"middle"
    assert await chunks.get() == b"newest"


def test_discard_pending_audio_drops_every_queued_chunk() -> None:
    chunks = AudioChunkQueue(max_chunks=3)
    chunks.put_nowait(b"oldest")
    chunks.put_nowait(b"middle")
    chunks.put_nowait(b"newest")

    discarded = chunks.discard_pending()

    assert discarded == 3
    assert chunks.qsize() == 0
    assert chunks.dropped_chunks == 3


def test_microphone_uses_configured_input_device(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = Mock()
    raw_input_stream = Mock(return_value=stream)
    query_devices = Mock(return_value={"name": "配信用マイク"})
    logger = Mock()
    logger.name = "realtime_summary.test"
    monkeypatch.setattr("realtime_summary.audio.sd.RawInputStream", raw_input_stream)
    monkeypatch.setattr("realtime_summary.audio.sd.query_devices", query_devices)
    microphone = MicrophoneInput(
        AudioChunkQueue(),
        logger=logger,
        device=12,
    )

    microphone.start(asyncio.new_event_loop())

    query_devices.assert_called_once_with(12, kind="input")
    assert raw_input_stream.call_args.kwargs["device"] == 12
