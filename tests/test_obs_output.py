from types import SimpleNamespace

import pytest

from realtime_summary.obs_output import (
    JAPANESE_CAPTION,
    ObsConfigurationError,
    ObsTextOutput,
)


class FakeObsClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str], bool]] = []

    def set_input_settings(
        self,
        name: str,
        settings: dict[str, str],
        overlay: bool,
    ) -> None:
        self.calls.append((name, settings, overlay))

    def get_input_settings(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(
            input_kind="text_gdiplus_v2",
            input_settings={"read_from_file": name == JAPANESE_CAPTION},
        )


def test_set_text_overlays_only_the_text_setting() -> None:
    client = FakeObsClient()
    output = ObsTextOutput(client)

    output.set_text(JAPANESE_CAPTION, "表示する字幕")

    assert client.calls == [
        (JAPANESE_CAPTION, {"text": "表示する字幕"}, True),
    ]


def test_preflight_rejects_read_from_file_sources() -> None:
    output = ObsTextOutput(FakeObsClient())

    with pytest.raises(ObsConfigurationError, match="Read from file"):
        output.preflight()
