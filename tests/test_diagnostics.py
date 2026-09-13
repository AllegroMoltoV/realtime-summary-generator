from __future__ import annotations

import json
import logging

from realtime_summary.diagnostics import configure_diagnostics, log_event, log_exception


def test_diagnostic_event_drops_content_and_credentials(tmp_path) -> None:
    logger = configure_diagnostics(tmp_path, run_id="run-test", level="DEBUG")

    log_event(
        logger,
        logging.INFO,
        subsystem="gemini",
        event="translation_completed",
        operation_id="translation-7",
        duration_ms=125.5,
        input_chars=42,
        output_chars=18,
        transcript="ログへ残してはいけない本文",
        api_key="secret-key",
    )
    for handler in logger.handlers:
        handler.flush()

    record = json.loads((tmp_path / "realtime-summary.jsonl").read_text(encoding="utf-8"))
    serialized = json.dumps(record, ensure_ascii=False)

    assert record["run_id"] == "run-test"
    assert record["subsystem"] == "gemini"
    assert record["event"] == "translation_completed"
    assert record["operation_id"] == "translation-7"
    assert record["duration_ms"] == 125.5
    assert "ログへ残してはいけない本文" not in serialized
    assert "secret-key" not in serialized
    assert "transcript" not in record
    assert "api_key" not in record


def test_exception_log_keeps_stack_but_drops_exception_message(tmp_path) -> None:
    logger = configure_diagnostics(tmp_path, run_id="run-error", level="DEBUG")
    try:
        raise RuntimeError("字幕本文とsecret-keyを含む外部API応答")
    except RuntimeError as exc:
        log_exception(logger, subsystem="openai", event="request_failed", exc=exc)
    for handler in logger.handlers:
        handler.flush()

    record = json.loads((tmp_path / "realtime-summary.jsonl").read_text(encoding="utf-8"))
    serialized = json.dumps(record, ensure_ascii=False)

    assert record["exception_type"] == "RuntimeError"
    assert record["stack"]
    assert "字幕本文" not in serialized
    assert "secret-key" not in serialized


def test_connection_metadata_is_preserved_without_password(tmp_path) -> None:
    logger = configure_diagnostics(tmp_path, run_id="run-obs", level="DEBUG")

    log_event(
        logger,
        logging.INFO,
        subsystem="obs",
        event="connecting",
        host="127.0.0.1",
        port=4455,
        password="secret-password",
    )
    for handler in logger.handlers:
        handler.flush()

    record = json.loads((tmp_path / "realtime-summary.jsonl").read_text(encoding="utf-8"))
    serialized = json.dumps(record, ensure_ascii=False)

    assert record["host"] == "127.0.0.1"
    assert record["port"] == 4455
    assert "secret-password" not in serialized
    assert "password" not in record
