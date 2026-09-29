from traceback import format_exception

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
    monkeypatch.delenv("OPENAI_TRANSCRIPTION_KEYWORDS", raising=False)
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=file-openai\n"
        "GEMINI_API_KEY=file-gemini\n"
        "OBS_WEBSOCKET_PASSWORD=file-obs\n"
        "OPENAI_TRANSCRIPTION_KEYWORDS='[\"架空ブランド\",\"商品A,B\"]'\n",
        encoding="utf-8",
    )

    settings = Settings.from_environment()

    assert settings.openai_api_key == "environment-openai"
    assert settings.gemini_api_key == "file-gemini"
    assert settings.obs_websocket_password == "file-obs"
    assert settings.transcription_keywords == ("架空ブランド", "商品A,B")
    assert "架空ブランド" not in repr(settings)

    monkeypatch.setenv("OPENAI_TRANSCRIPTION_KEYWORDS", '["環境変数の語句"]')
    assert Settings.from_environment().transcription_keywords == ("環境変数の語句",)
    monkeypatch.setenv("OPENAI_TRANSCRIPTION_KEYWORDS", "")
    assert Settings.from_environment().transcription_keywords == ()


@pytest.mark.parametrize("value", [None, "", "  ", "[]"])
def test_transcription_keywords_are_optional(value: str | None) -> None:
    values = {
        "OPENAI_API_KEY": "openai-secret",
        "GEMINI_API_KEY": "gemini-secret",
        "OBS_WEBSOCKET_PASSWORD": "obs-secret",
    }
    if value is not None:
        values["OPENAI_TRANSCRIPTION_KEYWORDS"] = value

    assert Settings.from_mapping(values).transcription_keywords == ()


@pytest.mark.parametrize(
    "value",
    [
        "非公開語句",
        '"非公開語句"',
        '{"keyword":"非公開語句"}',
        '["非公開語句",123]',
        '["非公開語句",null]',
        '["非公開語句",""]',
        '["非公開語句","  "]',
        '["非公開語句<"]',
        '["非公開語句>"]',
        '["非公開語句\\r"]',
        '["非公開語句\\n"]',
    ],
)
def test_invalid_transcription_keywords_do_not_disclose_values(value: str) -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_TRANSCRIPTION_KEYWORDS") as error:
        Settings.from_mapping(
            {
                "OPENAI_API_KEY": "openai-secret",
                "GEMINI_API_KEY": "gemini-secret",
                "OBS_WEBSOCKET_PASSWORD": "obs-secret",
                "OPENAI_TRANSCRIPTION_KEYWORDS": value,
            }
        )

    assert "非公開語句" not in str(error.value)
    assert "非公開語句" not in "".join(format_exception(error.value))


def test_transcription_keywords_trim_surrounding_whitespace() -> None:
    settings = Settings.from_mapping(
        {
            "OPENAI_API_KEY": "openai-secret",
            "GEMINI_API_KEY": "gemini-secret",
            "OBS_WEBSOCKET_PASSWORD": "obs-secret",
            "OPENAI_TRANSCRIPTION_KEYWORDS": '[" 架空ブランド ","two words"]',
        }
    )

    assert settings.transcription_keywords == ("架空ブランド", "two words")


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
