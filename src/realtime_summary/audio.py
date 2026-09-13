from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Any

import sounddevice as sd

from .diagnostics import log_event


class AudioChunkQueue:
    def __init__(self, *, max_chunks: int = 50) -> None:
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=max_chunks)
        self.dropped_chunks = 0

    def put_nowait(self, chunk: bytes) -> None:
        if self._queue.full():
            self._queue.get_nowait()
            self.dropped_chunks += 1
        self._queue.put_nowait(chunk)

    def put_from_thread(
        self,
        loop: asyncio.AbstractEventLoop,
        chunk: bytes,
    ) -> None:
        loop.call_soon_threadsafe(self.put_nowait, chunk)

    async def get(self) -> bytes:
        return await self._queue.get()

    def qsize(self) -> int:
        return self._queue.qsize()


class MicrophoneInput:
    def __init__(
        self,
        chunks: AudioChunkQueue,
        *,
        logger: logging.Logger,
        device: int | None = None,
    ) -> None:
        self._chunks = chunks
        self._logger = logger
        self._device = device
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: Any = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        device = sd.query_devices(self._device, kind="input")
        device_name = str(device.get("name", "default"))
        self._stream = sd.RawInputStream(
            device=self._device,
            samplerate=24000,
            blocksize=2400,
            channels=1,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()
        log_event(
            self._logger,
            logging.INFO,
            subsystem="audio",
            event="microphone_started",
            device_name=device_name,
            input_format="pcm16/24000/mono",
        )

    def close(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        log_event(
            self._logger,
            logging.INFO,
            subsystem="audio",
            event="microphone_stopped",
            audio_chunks_dropped=self._chunks.dropped_chunks,
        )

    def _callback(self, indata: Any, _frames: int, _time: Any, status: Any) -> None:
        if self._loop is None:
            return
        self._chunks.put_from_thread(self._loop, bytes(indata))
        if status:
            self._loop.call_soon_threadsafe(
                partial(
                    log_event,
                    self._logger,
                    logging.WARNING,
                    subsystem="audio",
                    event="input_status",
                    overflow=bool(getattr(status, "input_overflow", False)),
                    queue_depth=self._chunks.qsize(),
                    audio_chunks_dropped=self._chunks.dropped_chunks,
                )
            )
