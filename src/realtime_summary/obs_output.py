from __future__ import annotations

from typing import Any, Protocol


JAPANESE_CAPTION = "字幕（日本語）"
ENGLISH_CAPTION = "字幕（英語）"
JAPANESE_SUMMARY = "概要（日本語）"
TEXT_SOURCES = (JAPANESE_CAPTION, ENGLISH_CAPTION, JAPANESE_SUMMARY)


class ObsConfigurationError(RuntimeError):
    """OBSのテキストソース設定が実行要件を満たしていない。"""


class ObsClient(Protocol):
    def get_input_settings(self, name: str) -> Any: ...

    def set_input_settings(
        self,
        name: str,
        settings: dict[str, str],
        overlay: bool,
    ) -> object: ...


class ObsTextOutput:
    def __init__(self, client: ObsClient) -> None:
        self._client = client

    def set_text(self, source_name: str, text: str) -> None:
        if source_name not in TEXT_SOURCES:
            raise ValueError(f"Unknown OBS text source: {source_name}")
        self._client.set_input_settings(source_name, {"text": text}, True)

    def preflight(self) -> None:
        for source_name in TEXT_SOURCES:
            try:
                response = self._client.get_input_settings(source_name)
            except Exception as exc:
                raise ObsConfigurationError(
                    f"OBS Text (GDI+) source was not found: {source_name}"
                ) from exc

            input_kind = str(response.input_kind)
            if not input_kind.startswith("text_gdiplus"):
                raise ObsConfigurationError(
                    f"OBS source is not Text (GDI+): {source_name} ({input_kind})"
                )
            if bool(response.input_settings.get("read_from_file", False)):
                raise ObsConfigurationError(
                    f"Disable Read from file for OBS source: {source_name}"
                )

    def clear(self) -> None:
        for source_name in TEXT_SOURCES:
            self.set_text(source_name, "")
