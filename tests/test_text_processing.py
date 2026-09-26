import asyncio
import json
from time import monotonic
from types import SimpleNamespace

import pytest

from realtime_summary.diagnostics import configure_diagnostics
from realtime_summary.state import CompletedTurn, SummaryState
from realtime_summary.text_processing import (
    GeneratedSummary,
    GeneratedText,
    GeminiTextProcessor,
    InvalidSummaryResponse,
    SummaryResult,
    TextProcessingPipeline,
    parse_summary_response,
)


def test_summary_response_requires_three_array_entries() -> None:
    result = parse_summary_response(
        '{"display_lines":["目的を説明中","字幕方式を比較","OBSへ表示"]}'
    )

    assert result.display_lines == (
        "目的を説明中",
        "字幕方式を比較",
        "OBSへ表示",
    )

    with pytest.raises(InvalidSummaryResponse):
        parse_summary_response(
            '{"display_lines":["一行だけ"]}'
        )


def test_summary_response_allows_unused_trailing_lines() -> None:
    result = parse_summary_response(
        '{"display_lines":["今日はピアノを弾く", "", ""]}'
    )

    assert result.display_lines == ("今日はピアノを弾く", "", "")

    with pytest.raises(InvalidSummaryResponse):
        parse_summary_response(
            '{"display_lines":["", "", ""]}'
        )

    with pytest.raises(InvalidSummaryResponse):
        parse_summary_response(
            '{"display_lines":["話題", "", "補足"]}'
        )


def test_summary_response_limits_japanese_characters_and_marks_overflow() -> None:
    result = parse_summary_response(
        '{"display_lines":["日本語文字列追加", "日本語文字列", ""]}',
        max_chars=6,
    )

    assert result.display_lines == ("日本語(…)", "日本語文字列", "")
    assert result.model_line_chars == (8, 6, 0)
    assert all(len(line) <= 6 for line in result.display_lines)
    assert parse_summary_response(
        '{"display_lines":["日本語の説明", "", ""]}', max_chars=4
    ).display_lines[0] == "日(…)"


class FakeModels:
    async def generate_content(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            text="  Concise English caption.  ",
            usage_metadata=SimpleNamespace(
                prompt_token_count=12,
                candidates_token_count=5,
            ),
        )


@pytest.mark.asyncio
async def test_translation_returns_text_and_usage_counts() -> None:
    models = FakeModels()
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    processor = GeminiTextProcessor(client)

    result = await processor.translate("日本語の字幕")

    assert result.text == "Concise English caption."
    assert result.input_tokens == 12
    assert result.output_tokens == 5
    assert models.kwargs["model"] == "gemini-3.1-flash-lite"
    assert models.kwargs["config"].temperature == 0.2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_chars", "model_limit", "target_chars"),
    [(4, 2, 2), (8, 6, 6), (30, 28, 24), (40, 38, 32)],
)
async def test_summary_returns_validated_structure(
    max_chars: int, model_limit: int, target_chars: int
) -> None:
    models = FakeModels()

    async def generate_summary(**kwargs):
        models.kwargs = kwargs
        return SimpleNamespace(
            text='{"display_lines":["目的を説明中","字幕方式を比較","OBSへ表示"]}',
            usage_metadata=SimpleNamespace(
                prompt_token_count=30,
                candidates_token_count=20,
            ),
        )

    models.generate_content = generate_summary
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    processor = GeminiTextProcessor(client, summary_max_chars=max_chars)

    result = await processor.summarize(
        transcripts=("字幕方式を比較する", "OBSへ表示する"),
    )

    assert all(len(line) <= max_chars for line in result.summary.display_lines)
    assert json.loads(models.kwargs["contents"]) == {
        "recent_transcripts": ["字幕方式を比較する", "OBSへ表示する"]
    }
    instruction = models.kwargs["config"].system_instruction
    assert models.kwargs["config"].temperature == 0.0
    assert f"各行の最大文字数は{model_limit}文字" in instruction
    assert "必ずこの上限以内" in instruction
    assert f"情報が十分な行は{target_chars}文字前後を目指す" in instruction
    assert "方法または理由などの関連情報を一つ程度含める" in instruction
    assert "条件付きの予定を断定しない" in instruction
    assert "過去に終えた行は過去形" in instruction
    assert "挙げられた種類を省かず1行にまとめる" in instruction
    assert "説明語を短くして全種類を先に残す" in instruction
    assert "同じ話題を分割して3行を埋めない" in instruction
    assert "表示する話題が2件なら3行目は空文字" in instruction
    assert "漢字、ひらがな、カタカナ、英数字、記号をそれぞれ1文字" in instruction
    assert "1行30文字前後" not in instruction
    assert result.input_tokens == 30
    assert result.output_tokens == 20


@pytest.mark.asyncio
async def test_gemini_summary_truncates_overlong_line_before_return() -> None:
    models = FakeModels()

    async def generate_summary(**kwargs):
        models.kwargs = kwargs
        return SimpleNamespace(
            text='{"display_lines":["日本語文字列追加", "", ""]}',
            usage_metadata=None,
        )

    models.generate_content = generate_summary
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    processor = GeminiTextProcessor(client, summary_max_chars=6)

    result = await processor.summarize(transcripts=("日本語文字列追加",))

    assert result.summary.display_lines == ("日本語(…)", "", "")
    assert result.summary.model_line_chars == (8, 0, 0)


@pytest.mark.asyncio
async def test_older_translation_cannot_overwrite_newer_caption(tmp_path) -> None:
    release_first = asyncio.Event()

    class Processor:
        async def translate(self, transcript: str) -> GeneratedText:
            if transcript == "first":
                await release_first.wait()
            return GeneratedText(f"en-{transcript}", 1, 1)

        async def summarize(self, **_kwargs) -> GeneratedSummary:
            return GeneratedSummary(SummaryResult(("話題", "", "")), 1, 1)

    displayed: list[str] = []
    logger = configure_diagnostics(tmp_path, run_id="pipeline", level="DEBUG")
    pipeline = TextProcessingPipeline(
        processor=Processor(),
        summary_state=SummaryState(window_seconds=300.0),
        on_english=displayed.append,
        on_summary=lambda _text: None,
        logger=logger,
        summary_interval_seconds=20.0,
    )

    first = pipeline.submit_turn(CompletedTurn(0, "a", "first", monotonic()))
    second = pipeline.submit_turn(CompletedTurn(1, "b", "second", monotonic()))
    await second
    release_first.set()
    await first

    assert displayed == ["en-second"]


@pytest.mark.asyncio
async def test_translation_logs_start_before_completion(tmp_path) -> None:
    class Processor:
        async def translate(self, transcript: str) -> GeneratedText:
            return GeneratedText("translated", 1, 1)

        async def summarize(self, **_kwargs) -> GeneratedSummary:
            return GeneratedSummary(SummaryResult(("話題", "", "")), 1, 1)

    logger = configure_diagnostics(tmp_path, run_id="logging", level="DEBUG")
    pipeline = TextProcessingPipeline(
        processor=Processor(),
        summary_state=SummaryState(window_seconds=300.0),
        on_english=lambda _text: None,
        on_summary=lambda _text: None,
        logger=logger,
        summary_interval_seconds=20.0,
    )

    await pipeline.submit_turn(CompletedTurn(0, "item", "発話本文", monotonic()))
    for handler in logger.handlers:
        handler.flush()
    records = [
        json.loads(line)
        for line in (tmp_path / "realtime-summary.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert [
        record["event"]
        for record in records
        if record["event"].startswith("translation_")
    ] == [
        "translation_started",
        "translation_completed",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("interval_seconds", [10.0, 20.0])
async def test_summary_runs_on_qualifying_completed_turns(
    tmp_path, monkeypatch, interval_seconds: float
) -> None:
    clock = [0.0]
    monkeypatch.setattr("realtime_summary.text_processing.monotonic", lambda: clock[0])
    release_first = asyncio.Event()
    summarized: list[tuple[str, ...]] = []

    class Processor:
        async def translate(self, _transcript: str) -> GeneratedText:
            return GeneratedText("translation", 1, 1)

        async def summarize(self, *, transcripts: tuple[str, ...]) -> GeneratedSummary:
            summarized.append(transcripts)
            if len(summarized) == 1:
                await release_first.wait()
            return GeneratedSummary(SummaryResult(("話題", "", "")), 1, 1)

    logger = configure_diagnostics(tmp_path, run_id="summary-schedule", level="INFO")
    pipeline = TextProcessingPipeline(
        processor=Processor(),
        summary_state=SummaryState(
            window_seconds=300.0, clock=lambda: clock[0]
        ),
        on_english=lambda _text: None,
        on_summary=lambda _text: None,
        logger=logger,
        summary_interval_seconds=interval_seconds,
    )

    pipeline.submit_turn(CompletedTurn(0, "a", "first", clock[0]))
    first_summary = pipeline._summary_task
    assert first_summary is not None
    await asyncio.sleep(0)
    assert summarized == [("first",)]

    clock[0] = 25.0
    pipeline.submit_turn(CompletedTurn(1, "b", "second", clock[0]))
    assert pipeline._summary_task is first_summary
    release_first.set()
    await first_summary

    clock[0] = 25.0 + interval_seconds - 1.0
    pipeline.submit_turn(CompletedTurn(2, "c", "third", clock[0]))
    assert pipeline._summary_task is first_summary

    clock[0] = 25.0 + interval_seconds
    pipeline.submit_turn(CompletedTurn(3, "d", "fourth", clock[0]))
    second_summary = pipeline._summary_task
    assert second_summary is not None and second_summary is not first_summary
    await second_summary

    assert summarized == [("first",), ("first", "second", "third", "fourth")]
    await pipeline.close()


@pytest.mark.asyncio
async def test_summary_logs_start_before_completion(tmp_path) -> None:
    class Processor:
        async def summarize(self, **_kwargs) -> GeneratedSummary:
            return GeneratedSummary(
                SummaryResult(("一行目", "二行目", "三行目"), (9, 3, 3)),
                2,
                3,
            )

    state = SummaryState(window_seconds=300.0)
    state.add_transcript("発話本文", completed_at=monotonic())
    logger = configure_diagnostics(tmp_path, run_id="summary-logging", level="DEBUG")
    pipeline = TextProcessingPipeline(
        processor=Processor(),
        summary_state=state,
        on_english=lambda _text: None,
        on_summary=lambda _text: None,
        logger=logger,
        summary_interval_seconds=20.0,
    )

    await pipeline.update_summary_once()
    for handler in logger.handlers:
        handler.flush()
    records = [
        json.loads(line)
        for line in (tmp_path / "realtime-summary.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert [record["event"] for record in records] == [
        "summary_started",
        "summary_completed",
    ]
    assert records[1]["summary_line_chars"] == [3, 3, 3]
    assert records[1]["model_line_chars"] == [9, 3, 3]


@pytest.mark.asyncio
async def test_summary_displays_only_supported_lines(tmp_path) -> None:
    class Processor:
        async def summarize(self, **_kwargs) -> GeneratedSummary:
            return GeneratedSummary(
                SummaryResult(("今日はピアノを弾く", "", "")),
                2,
                3,
            )

    state = SummaryState(window_seconds=300.0)
    state.add_transcript("今日はピアノを弾きます", completed_at=monotonic())
    displayed: list[str] = []
    logger = configure_diagnostics(tmp_path, run_id="sparse-summary", level="INFO")
    pipeline = TextProcessingPipeline(
        processor=Processor(),
        summary_state=state,
        on_english=lambda _text: None,
        on_summary=displayed.append,
        logger=logger,
        summary_interval_seconds=20.0,
    )

    await pipeline.update_summary_once()

    assert displayed == ["今日はピアノを弾く"]
    assert state.display_lines == ("今日はピアノを弾く", "", "")


@pytest.mark.asyncio
async def test_summary_log_includes_window_transcript_count(tmp_path) -> None:
    class Processor:
        async def summarize(self, **_kwargs) -> GeneratedSummary:
            return GeneratedSummary(
                SummaryResult(("一行目", "二行目", "三行目")),
                2,
                3,
            )

    state = SummaryState(window_seconds=300.0)
    completed_at = monotonic()
    state.add_transcript("前の発話", completed_at=completed_at)
    state.add_transcript("新しい発話", completed_at=completed_at)
    logger = configure_diagnostics(tmp_path, run_id="window-count", level="DEBUG")
    pipeline = TextProcessingPipeline(
        processor=Processor(),
        summary_state=state,
        on_english=lambda _text: None,
        on_summary=lambda _text: None,
        logger=logger,
        summary_interval_seconds=20.0,
    )

    await pipeline.update_summary_once()
    for handler in logger.handlers:
        handler.flush()
    records = [
        json.loads(line)
        for line in (tmp_path / "realtime-summary.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert records[0]["event"] == "summary_started"
    assert records[0]["transcript_count"] == 2
