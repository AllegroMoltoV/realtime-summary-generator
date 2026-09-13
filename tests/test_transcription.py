from __future__ import annotations

import pytest

from realtime_summary.state import CompletedTurn
from realtime_summary.transcription import (
    OPENAI_REALTIME_URL,
    TranscriptionEventRouter,
    build_audio_append,
    build_session_update,
)


def test_realtime_url_selects_a_transcription_session() -> None:
    assert OPENAI_REALTIME_URL == (
        "wss://api.openai.com/v1/realtime?intent=transcription"
    )


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
        CompletedTurn(0, "item-a", "最初"),
        CompletedTurn(1, "item-b", "二番目"),
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

    assert completed == [CompletedTurn(1, "item-b", "後続の発話")]


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
