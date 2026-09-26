from __future__ import annotations

import argparse
import asyncio
import logging
import platform
import sys
from dataclasses import replace
from math import isfinite
from pathlib import Path
from time import monotonic
from uuid import uuid4

import obsws_python as obs
from google import genai

from . import __version__
from .audio import AudioChunkQueue, MicrophoneInput
from .config import (
    DEFAULT_SUMMARY_INTERVAL_SECONDS,
    DEFAULT_SUMMARY_MAX_CHARS,
    MIN_SUMMARY_MAX_CHARS,
    ConfigurationError,
    Settings,
)
from .diagnostics import (
    close_diagnostics,
    configure_diagnostics,
    log_event,
    log_exception,
)
from .obs_output import (
    ENGLISH_CAPTION,
    JAPANESE_CAPTION,
    JAPANESE_SUMMARY,
    ObsConfigurationError,
    ObsTextOutput,
)
from .state import CompletedTurn, SummaryState
from .text_processing import GeminiTextProcessor, TextProcessingPipeline
from .transcription import (
    RealtimeApiError,
    RealtimeTranscriber,
    TranscriptionEventRouter,
)


async def run(settings: Settings) -> None:
    run_id = uuid4().hex
    logger = configure_diagnostics(
        Path(".logs"),
        run_id=run_id,
        level=settings.log_level,
    )
    log_event(
        logger,
        logging.INFO,
        subsystem="app",
        event="started",
        app_version=__version__,
        python_version=platform.python_version(),
        summary_max_chars=settings.summary_max_chars,
        summary_interval_seconds=settings.summary_interval_seconds,
        status="credentials_present",
    )

    obs_client = None
    google_client = None
    microphone = None
    pipeline = None
    stop_status = "normal"
    try:
        log_event(
            logger,
            logging.INFO,
            subsystem="obs",
            event="connecting",
            host=settings.obs_host,
            port=settings.obs_port,
        )
        obs_client = await asyncio.to_thread(
            obs.ReqClient,
            host=settings.obs_host,
            port=settings.obs_port,
            password=settings.obs_websocket_password,
            timeout=3,
        )
        output = ObsTextOutput(obs_client)
        await asyncio.to_thread(output.preflight)
        await asyncio.to_thread(output.clear)
        log_event(
            logger,
            logging.INFO,
            subsystem="obs",
            event="preflight_completed",
        )

        async def update_obs(source_name: str, text: str) -> None:
            started = monotonic()
            try:
                await asyncio.to_thread(output.set_text, source_name, text)
                log_event(
                    logger,
                    logging.DEBUG,
                    subsystem="obs",
                    event="text_updated",
                    source_name=source_name,
                    duration_ms=round((monotonic() - started) * 1000, 1),
                    output_chars=len(text),
                )
            except Exception as exc:
                log_exception(
                    logger,
                    subsystem="obs",
                    event="text_update_failed",
                    exc=exc,
                    source_name=source_name,
                    duration_ms=round((monotonic() - started) * 1000, 1),
                    output_chars=len(text),
                )

        google_client = genai.Client(api_key=settings.gemini_api_key)
        summary_state = SummaryState(window_seconds=settings.summary_window_seconds)
        pipeline = TextProcessingPipeline(
            processor=GeminiTextProcessor(
                google_client, summary_max_chars=settings.summary_max_chars
            ),
            summary_state=summary_state,
            on_english=lambda text: update_obs(ENGLISH_CAPTION, text),
            on_summary=lambda text: update_obs(JAPANESE_SUMMARY, text),
            logger=logger,
            summary_interval_seconds=settings.summary_interval_seconds,
        )

        def submit_turn(turn: CompletedTurn) -> None:
            assert pipeline is not None
            pipeline.submit_turn(turn)

        router = TranscriptionEventRouter(
            on_caption=lambda text: update_obs(JAPANESE_CAPTION, text),
            on_completed=submit_turn,
        )
        chunks = AudioChunkQueue(max_chunks=50)
        microphone = MicrophoneInput(
            chunks,
            logger=logger,
            device=settings.audio_input_device,
        )
        transcriber = RealtimeTranscriber(
            api_key=settings.openai_api_key,
            router=router,
            logger=logger,
        )

        microphone.start(asyncio.get_running_loop())
        print("字幕・英訳・概要の更新を開始しました。終了するには Ctrl+C を押してください。")
        await transcriber.run(chunks)
    except asyncio.CancelledError:
        stop_status = "cancelled"
        raise
    except Exception as exc:
        stop_status = "error"
        log_exception(logger, subsystem="app", event="stopped_by_error", exc=exc)
        raise
    finally:
        if microphone is not None:
            try:
                microphone.close()
            except Exception as exc:
                log_exception(
                    logger,
                    subsystem="audio",
                    event="microphone_close_failed",
                    exc=exc,
                )
        if pipeline is not None:
            await pipeline.close()
        if obs_client is not None:
            try:
                output = ObsTextOutput(obs_client)
                await asyncio.to_thread(output.clear)
            except Exception as exc:
                log_exception(
                    logger,
                    subsystem="obs",
                    event="clear_failed",
                    exc=exc,
                )
            try:
                await asyncio.to_thread(obs_client.disconnect)
            except Exception as exc:
                log_exception(
                    logger,
                    subsystem="obs",
                    event="disconnect_failed",
                    exc=exc,
                )
        if google_client is not None:
            await google_client.aio.aclose()
        log_event(
            logger,
            logging.INFO,
            subsystem="app",
            event="stopped",
            status=stop_status,
        )
        close_diagnostics(logger)


def _summary_max_chars(value: str) -> int:
    try:
        max_chars = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Nには4以上の整数を指定してください。") from exc
    if max_chars < MIN_SUMMARY_MAX_CHARS:
        raise argparse.ArgumentTypeError("Nには4以上の整数を指定してください。")
    return max_chars


def _summary_interval_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "SECONDSには正の秒数を指定してください。"
        ) from exc
    if not isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("SECONDSには正の秒数を指定してください。")
    return seconds


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OBS向け字幕・概要生成ツール")
    parser.add_argument(
        "--summary-max-chars",
        type=_summary_max_chars,
        default=DEFAULT_SUMMARY_MAX_CHARS,
        metavar="N",
        help="概要の1行あたりの最大文字数。4以上、既定値は30。",
    )
    parser.add_argument(
        "--summary-interval-seconds",
        type=_summary_interval_seconds,
        default=DEFAULT_SUMMARY_INTERVAL_SECONDS,
        metavar="SECONDS",
        help="概要更新の最短間隔。確定発話時に判定。既定値は10秒。",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_cli_args()
    try:
        settings = replace(
            Settings.from_environment(),
            summary_max_chars=args.summary_max_chars,
            summary_interval_seconds=args.summary_interval_seconds,
        )
    except ConfigurationError as exc:
        print(f"設定エラー: {exc}", file=sys.stderr)
        return 2

    try:
        asyncio.run(run(settings))
    except KeyboardInterrupt:
        print("終了しました。")
        return 0
    except (ObsConfigurationError, RealtimeApiError) as exc:
        print(f"実行エラー: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            f"実行エラー: {type(exc).__name__}。詳細は .logs/realtime-summary.jsonl を確認してください。",
            file=sys.stderr,
        )
        return 1
    return 0
