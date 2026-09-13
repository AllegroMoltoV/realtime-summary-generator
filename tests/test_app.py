import json

import pytest

from realtime_summary import app
from realtime_summary.config import Settings


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
    assert "接続エラー本文" not in serialized
    assert "openai-secret" not in serialized
    assert "gemini-secret" not in serialized
    assert "obs-secret" not in serialized
