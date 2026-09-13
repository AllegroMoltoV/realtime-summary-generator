import asyncio
import json
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


def test_summary_response_requires_three_nonempty_lines() -> None:
    result = parse_summary_response(
        '{"context_memory":"配信の目的",'
        '"display_lines":["目的を説明中","字幕方式を比較","OBSへ表示"]}'
    )

    assert result.context_memory == "配信の目的"
    assert result.display_lines == (
        "目的を説明中",
        "字幕方式を比較",
        "OBSへ表示",
    )

    with pytest.raises(InvalidSummaryResponse):
        parse_summary_response(
            '{"context_memory":"状態","display_lines":["一行だけ"]}'
        )


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


@pytest.mark.asyncio
async def test_summary_returns_validated_structure() -> None:
    models = FakeModels()

    async def generate_summary(**kwargs):
        models.kwargs = kwargs
        return SimpleNamespace(
            text=(
                '{"context_memory":"配信の目的と方式",'
                '"display_lines":["目的を説明中","字幕方式を比較","OBSへ表示"]}'
            ),
            usage_metadata=SimpleNamespace(
                prompt_token_count=30,
                candidates_token_count=20,
            ),
        )

    models.generate_content = generate_summary
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    processor = GeminiTextProcessor(client)

    result = await processor.summarize(
        context_memory="配信の目的",
        transcripts=("字幕方式を比較する", "OBSへ表示する"),
    )

    assert result.summary.display_lines[2] == "OBSへ表示"
    assert result.input_tokens == 30
    assert result.output_tokens == 20


@pytest.mark.asyncio
async def test_older_translation_cannot_overwrite_newer_caption(tmp_path) -> None:
    release_first = asyncio.Event()

    class Processor:
        async def translate(self, transcript: str) -> GeneratedText:
            if transcript == "first":
                await release_first.wait()
            return GeneratedText(f"en-{transcript}", 1, 1)

    displayed: list[str] = []
    logger = configure_diagnostics(tmp_path, run_id="pipeline", level="DEBUG")
    pipeline = TextProcessingPipeline(
        processor=Processor(),
        summary_state=SummaryState(max_cycles=3),
        on_english=displayed.append,
        on_summary=lambda _text: None,
        logger=logger,
    )

    first = pipeline.submit_turn(CompletedTurn(0, "a", "first"))
    second = pipeline.submit_turn(CompletedTurn(1, "b", "second"))
    await second
    release_first.set()
    await first

    assert displayed == ["en-second"]


@pytest.mark.asyncio
async def test_translation_logs_start_before_completion(tmp_path) -> None:
    class Processor:
        async def translate(self, transcript: str) -> GeneratedText:
            return GeneratedText("translated", 1, 1)

    logger = configure_diagnostics(tmp_path, run_id="logging", level="DEBUG")
    pipeline = TextProcessingPipeline(
        processor=Processor(),
        summary_state=SummaryState(max_cycles=3),
        on_english=lambda _text: None,
        on_summary=lambda _text: None,
        logger=logger,
    )

    await pipeline.submit_turn(CompletedTurn(0, "item", "発話本文"))
    for handler in logger.handlers:
        handler.flush()
    records = [
        json.loads(line)
        for line in (tmp_path / "realtime-summary.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert [record["event"] for record in records] == [
        "translation_started",
        "translation_completed",
    ]


@pytest.mark.asyncio
async def test_summary_logs_start_before_completion(tmp_path) -> None:
    class Processor:
        async def summarize(self, **_kwargs) -> GeneratedSummary:
            return GeneratedSummary(
                SummaryResult("context", ("一行目", "二行目", "三行目")),
                2,
                3,
            )

    state = SummaryState(max_cycles=3)
    state.add_transcript("発話本文")
    logger = configure_diagnostics(tmp_path, run_id="summary-logging", level="DEBUG")
    pipeline = TextProcessingPipeline(
        processor=Processor(),
        summary_state=state,
        on_english=lambda _text: None,
        on_summary=lambda _text: None,
        logger=logger,
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


@pytest.mark.asyncio
async def test_summary_log_includes_dropped_cycle_count(tmp_path) -> None:
    class Processor:
        async def summarize(self, **_kwargs) -> GeneratedSummary:
            return GeneratedSummary(
                SummaryResult("context", ("一行目", "二行目", "三行目")),
                2,
                3,
            )

    state = SummaryState(max_cycles=1)
    state.add_transcript("失敗した周期")
    assert state.begin_update() is not None
    state.add_transcript("新しい周期")
    state.fail_update()
    logger = configure_diagnostics(tmp_path, run_id="dropped-cycles", level="DEBUG")
    pipeline = TextProcessingPipeline(
        processor=Processor(),
        summary_state=state,
        on_english=lambda _text: None,
        on_summary=lambda _text: None,
        logger=logger,
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
    assert records[0]["dropped_cycles"] == 1
