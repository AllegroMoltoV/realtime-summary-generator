import pytest

from realtime_summary.config import ConfigurationError, Settings


def test_dotenv_is_loaded_without_overriding_existing_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "environment-openai")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OBS_WEBSOCKET_PASSWORD", raising=False)
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=file-openai\n"
        "GEMINI_API_KEY=file-gemini\n"
        "OBS_WEBSOCKET_PASSWORD=file-obs\n",
        encoding="utf-8",
    )

    settings = Settings.from_environment()

    assert settings.openai_api_key == "environment-openai"
    assert settings.gemini_api_key == "file-gemini"
    assert settings.obs_websocket_password == "file-obs"


def test_missing_credentials_are_reported_without_values() -> None:
    with pytest.raises(ConfigurationError) as captured:
        Settings.from_mapping({"OPENAI_API_KEY": "present"})

    message = str(captured.value)
    assert "GEMINI_API_KEY" in message
    assert "OBS_WEBSOCKET_PASSWORD" in message
    assert "present" not in message


def test_audio_input_device_is_loaded_from_environment() -> None:
    settings = Settings.from_mapping(
        {
            "OPENAI_API_KEY": "openai-secret",
            "GEMINI_API_KEY": "gemini-secret",
            "OBS_WEBSOCKET_PASSWORD": "obs-secret",
            "AUDIO_INPUT_DEVICE": "12",
        }
    )

    assert settings.audio_input_device == 12


def test_audio_input_device_rejects_non_numeric_value() -> None:
    with pytest.raises(ConfigurationError, match="AUDIO_INPUT_DEVICE"):
        Settings.from_mapping(
            {
                "OPENAI_API_KEY": "openai-secret",
                "GEMINI_API_KEY": "gemini-secret",
                "OBS_WEBSOCKET_PASSWORD": "obs-secret",
                "AUDIO_INPUT_DEVICE": "配信用マイク",
            }
        )
