from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values


class ConfigurationError(ValueError):
    """実行に必要な設定が不足している。"""


DEFAULT_SUMMARY_MAX_CHARS = 30
MIN_SUMMARY_MAX_CHARS = 4
DEFAULT_SUMMARY_INTERVAL_SECONDS = 10.0


@dataclass(frozen=True, slots=True)
class Settings:
    openai_api_key: str = field(repr=False)
    gemini_api_key: str = field(repr=False)
    obs_websocket_password: str = field(repr=False)
    log_level: str = "INFO"
    obs_host: str = "127.0.0.1"
    obs_port: int = 4455
    summary_interval_seconds: float = DEFAULT_SUMMARY_INTERVAL_SECONDS
    summary_window_seconds: float = 300.0
    summary_max_chars: int = DEFAULT_SUMMARY_MAX_CHARS
    audio_input_device: int | None = None

    @classmethod
    def from_environment(cls) -> Settings:
        values = {
            name: value
            for name, value in dotenv_values(Path(".env")).items()
            if value is not None
        }
        values.update(os.environ)
        return cls.from_mapping(values)

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> Settings:
        required = (
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "OBS_WEBSOCKET_PASSWORD",
        )
        missing = [name for name in required if not values.get(name)]
        if missing:
            raise ConfigurationError(
                "Missing environment variables: " + ", ".join(missing)
            )

        log_level = values.get("LOG_LEVEL", "INFO").upper()
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ConfigurationError(f"Unsupported LOG_LEVEL: {log_level}")

        raw_audio_input_device = values.get("AUDIO_INPUT_DEVICE")
        try:
            audio_input_device = (
                int(raw_audio_input_device) if raw_audio_input_device else None
            )
        except ValueError as exc:
            raise ConfigurationError(
                "AUDIO_INPUT_DEVICE must be a numeric device ID"
            ) from exc

        return cls(
            openai_api_key=values["OPENAI_API_KEY"],
            gemini_api_key=values["GEMINI_API_KEY"],
            obs_websocket_password=values["OBS_WEBSOCKET_PASSWORD"],
            log_level=log_level,
            audio_input_device=audio_input_device,
        )
