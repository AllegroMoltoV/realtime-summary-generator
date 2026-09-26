from __future__ import annotations

import json
import logging
import traceback
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


_ALLOWED_FIELDS = frozenset(
    {
        "app_version",
        "audio_chunks_discarded",
        "audio_chunks_dropped",
        "close_code",
        "device_name",
        "duration_ms",
        "error_code",
        "exception_type",
        "host",
        "input_chars",
        "input_format",
        "input_tokens",
        "item_id",
        "max_retries",
        "model_line_chars",
        "operation_id",
        "output_chars",
        "output_tokens",
        "overflow",
        "partial_captions_discarded",
        "ping_interval_seconds",
        "ping_timeout_seconds",
        "port",
        "previous_item_id",
        "python_version",
        "queue_depth",
        "request_id",
        "received_close_code",
        "received_close_reason",
        "retry_base_delay_seconds",
        "retry_delay_seconds",
        "retry_max_delay_seconds",
        "retry_count",
        "sent_close_code",
        "sent_close_reason",
        "source_name",
        "stack",
        "stable_connection_seconds",
        "status",
        "summary_interval_seconds",
        "summary_line_chars",
        "summary_max_chars",
        "timing_items_discarded",
        "transcript_count",
        "transcription_items_discarded",
        "turn_sequence",
    }
)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "run_id": record.run_id,
            "level": record.levelname,
            "subsystem": record.subsystem,
            "event": record.event_name,
        }
        payload.update(record.event_fields)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_diagnostics(
    log_dir: Path,
    *,
    run_id: str,
    level: str = "INFO",
) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"realtime_summary.{run_id}")
    for old_handler in logger.handlers:
        old_handler.close()
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(level.upper())

    handler = RotatingFileHandler(
        log_dir / "realtime-summary.jsonl",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    return logger


def close_diagnostics(logger: logging.Logger) -> None:
    for handler in tuple(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def log_event(
    logger: logging.Logger,
    level: int,
    *,
    subsystem: str,
    event: str,
    **fields: object,
) -> None:
    safe_fields = {key: value for key, value in fields.items() if key in _ALLOWED_FIELDS}
    logger.log(
        level,
        event,
        extra={
            "run_id": logger.name.rsplit(".", maxsplit=1)[-1],
            "subsystem": subsystem,
            "event_name": event,
            "event_fields": safe_fields,
        },
    )


def log_exception(
    logger: logging.Logger,
    *,
    subsystem: str,
    event: str,
    exc: BaseException,
    **fields: object,
) -> None:
    stack = [
        {
            "file": frame.filename,
            "line": frame.lineno,
            "function": frame.name,
        }
        for frame in traceback.extract_tb(exc.__traceback__)
    ]
    error_code = fields.pop("error_code", None)
    if error_code is None:
        error_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    log_event(
        logger,
        logging.ERROR,
        subsystem=subsystem,
        event=event,
        exception_type=type(exc).__name__,
        stack=stack,
        error_code=error_code,
        request_id=request_id,
        **fields,
    )
