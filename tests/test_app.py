import json
from unittest.mock import AsyncMock, Mock

import pytest

from realtime_summary import app
from realtime_summary.config import Settings


def test_summary_cli_accepts_defaults_and_custom_values() -> None:
    defaults = app.parse_cli_args([])
    custom = app.parse_cli_args(
        ["--summary-max-chars", "48", "--summary-interval-seconds", "2.5"]
    )

    assert defaults.summary_max_chars == 30
    assert defaults.summary_interval_seconds == 10.0
    assert custom.summary_max_chars == 48
    assert custom.summary_interval_seconds == 2.5


@pytest.mark.parametrize("value", ["3", "0", "-1", "abc", "4.5"])
def test_summary_max_chars_cli_rejects_invalid_limit(value: str) -> None:
    with pytest.raises(SystemExit) as captured:
        app.parse_cli_args(["--summary-max-chars", value])
    assert captured.value.code == 2


@pytest.mark.parametrize("value", ["0", "-1", "abc", "nan", "inf", "-inf"])
def test_summary_interval_cli_rejects_invalid_seconds(value: str) -> None:
    with pytest.raises(SystemExit) as captured:
        app.parse_cli_args(["--summary-interval-seconds", value])
    assert captured.value.code == 2


def test_main_passes_cli_values_to_runtime(monkeypatch) -> None:
    settings = Settings(
        openai_api_key="openai-secret",
        gemini_api_key="gemini-secret",
        obs_websocket_password="obs-secret",
    )
    received: list[tuple[int, float]] = []

    async def fake_run(config: Settings) -> None:
        received.append((config.summary_max_chars, config.summary_interval_seconds))

    monkeypatch.setattr(
        app.sys,
        "argv",
        [
            "realtime-summary",
            "--summary-max-chars",
            "40",
            "--summary-interval-seconds",
            "12.5",
        ],
    )
    monkeypatch.setattr(app.Settings, "from_environment", classmethod(lambda _cls: settings))
    monkeypatch.setattr(app, "run", fake_run)

    assert app.main() == 0
    assert received == [(40, 12.5)]


@pytest.mark.asyncio
async def test_failed_startup_logs_error_stop_status(tmp_path, monkeypatch) -> None:
    def reject_connection(**_kwargs):
        raise ConnectionRefusedError("接続エラー本文")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app.obs, "ReqClient", reject_connection)
    settings = Settings(
        openai_api_key="openai-secret",
        gemini_api_key="gemini-secret",
        obs_websocket_password="obs-secret",
    )

    with pytest.raises(ConnectionRefusedError):
        await app.run(settings)

    records = [
        json.loads(line)
        for line in (tmp_path / ".logs" / "realtime-summary.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    serialized = json.dumps(records, ensure_ascii=False)

    assert records[-1]["event"] == "stopped"
    assert records[-1]["status"] == "error"
    assert records[0]["summary_max_chars"] == 30
    assert records[0]["summary_interval_seconds"] == 10.0
    assert "接続エラー本文" not in serialized
    assert "openai-secret" not in serialized
    assert "gemini-secret" not in serialized
    assert "obs-secret" not in serialized


@pytest.mark.asyncio
async def test_run_passes_transcription_keywords_without_logging_them(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app.obs, "ReqClient", Mock())
    monkeypatch.setattr(app, "ObsTextOutput", Mock())
    client = Mock()
    client.aio.aclose = AsyncMock()
    monkeypatch.setattr(app.genai, "Client", Mock(return_value=client))
    pipeline = Mock()
    pipeline.close = AsyncMock()
    monkeypatch.setattr(app, "TextProcessingPipeline", Mock(return_value=pipeline))
    monkeypatch.setattr(app, "MicrophoneInput", Mock())
    transcriber = Mock()
    transcriber.run = AsyncMock()
    factory = Mock(return_value=transcriber)
    monkeypatch.setattr(app, "RealtimeTranscriber", factory)
    settings = Settings(
        openai_api_key="openai-secret",
        gemini_api_key="gemini-secret",
        obs_websocket_password="obs-secret",
        transcription_keywords=("架空ブランド", "配信用語"),
    )

    await app.run(settings)

    assert factory.call_args.kwargs["keywords"] == settings.transcription_keywords
    transcriber.run.assert_awaited_once()
    log = (tmp_path / ".logs" / "realtime-summary.jsonl").read_text(encoding="utf-8")
    assert all(keyword not in log for keyword in settings.transcription_keywords)
