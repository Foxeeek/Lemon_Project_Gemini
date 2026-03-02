"""Audio capture and VAD-based chunking for SpeakPilot.

Module 1 responsibilities:
- Capture microphone input continuously.
- Run Voice Activity Detection (VAD) on incoming frames.
- Emit in-memory utterance chunks when speech ends after configurable silence.

This module is intentionally diskless: audio never touches the filesystem.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator, Deque, Optional

import numpy as np
import sounddevice as sd
import webrtcvad

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AudioChunk:
    """A single speech utterance captured from microphone."""

    pcm16: bytes
    sample_rate: int
    channels: int
    started_at: float
    ended_at: float

    @property
    def duration_seconds(self) -> float:
        total_samples = len(self.pcm16) // 2
        return total_samples / float(self.sample_rate * self.channels)


class AudioManager:
    """Continuously captures mic audio and splits speech chunks using VAD.

    Designed for low-latency streaming with asyncio:
    - InputStream callback pushes fixed-duration frames to an async queue.
    - Async worker applies VAD and emits completed chunks through another queue.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        channels: int = 1,
        frame_duration_ms: int = 30,
        vad_aggressiveness: int = 2,
        silence_timeout_s: float = 0.6,
        pre_speech_padding_ms: int = 300,
        device: Optional[int | str] = None,
        max_segment_s: float = 1.0,
    ) -> None:
        if frame_duration_ms not in (10, 20, 30):
            raise ValueError("frame_duration_ms must be one of: 10, 20, 30")
        if channels != 1:
            raise ValueError("webrtcvad requires mono audio; set channels=1")
        if not 0 <= vad_aggressiveness <= 3:
            raise ValueError("vad_aggressiveness must be between 0 and 3")

        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_duration_ms = frame_duration_ms
        self.frame_samples = int(sample_rate * frame_duration_ms / 1000)
        self.frame_bytes = self.frame_samples * channels * 2  # int16
        self.silence_timeout_s = silence_timeout_s
        self.pre_speech_padding_frames = max(
            1, int(pre_speech_padding_ms / frame_duration_ms)
        )
        self.max_segment_frames = int((max_segment_s * 1000) / frame_duration_ms)
        self.device = device

        self.vad = webrtcvad.Vad(vad_aggressiveness)

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._frame_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        self._chunk_queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=32)
        self._stream: Optional[sd.InputStream] = None
        self._worker_task: Optional[asyncio.Task[None]] = None
        self._running = False

    @staticmethod
    def list_input_devices() -> list[dict]:
        """Return input-capable devices for UI/device selection."""
        devices = sd.query_devices()
        result: list[dict] = []
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                result.append(
                    {
                        "id": idx,
                        "name": dev.get("name"),
                        "default_samplerate": int(dev.get("default_samplerate", 0)),
                        "max_input_channels": int(dev.get("max_input_channels", 0)),
                    }
                )
        return result

    async def start(self) -> None:
        """Start microphone capture and VAD processing."""
        if self._running:
            return

        self._loop = asyncio.get_running_loop()
        self._running = True

        self._worker_task = asyncio.create_task(self._process_frames(), name="vad-processor")

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                blocksize=self.frame_samples,
                callback=self._audio_callback,
                device=self.device,
                latency="low",
            )
            self._stream.start()
            logger.info("AudioManager started (sr=%s, frame=%sms)", self.sample_rate, self.frame_duration_ms)
        except Exception:
            self._running = False
            if self._worker_task:
                self._worker_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._worker_task
            logger.exception("Failed to start microphone capture")
            raise

    async def stop(self) -> None:
        """Stop audio capture and processing gracefully."""
        if not self._running:
            return

        self._running = False

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.exception("Error while closing input stream")
            finally:
                self._stream = None

        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None

        self._drain_queue(self._frame_queue)
        logger.info("AudioManager stopped")

    async def chunks(self) -> AsyncIterator[AudioChunk]:
        """Async iterator yielding finalized utterance chunks."""
        while True:
            chunk = await self._chunk_queue.get()
            yield chunk

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.warning("Audio callback status: %s", status)

        if not self._running or self._loop is None:
            return

        # indata shape: (frames, channels), dtype=int16
        frame_bytes = indata.tobytes()

        def enqueue() -> None:
            if self._frame_queue.full():
                # Drop oldest frame to keep latency low.
                try:
                    self._frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                self._frame_queue.put_nowait(frame_bytes)
            except asyncio.QueueFull:
                logger.debug("Frame queue full; dropped one frame")

        self._loop.call_soon_threadsafe(enqueue)

    async def _process_frames(self) -> None:
        """Apply VAD state machine to frames and emit speech chunks."""
        pre_buffer: Deque[bytes] = deque(maxlen=self.pre_speech_padding_frames)
        current_segment = bytearray()
        segment_start: Optional[float] = None
        last_voice_time: Optional[float] = None
        is_speaking = False

        while self._running:
            try:
                frame = await self._frame_queue.get()
            except asyncio.CancelledError:
                return

            now = time.time()
            if len(frame) != self.frame_bytes:
                logger.debug("Skipping frame with unexpected size: %s", len(frame))
                continue

            try:
                voiced = self.vad.is_speech(frame, self.sample_rate)
            except Exception:
                logger.exception("VAD failed for frame")
                continue

            if not is_speaking:
                pre_buffer.append(frame)
                if voiced:
                    is_speaking = True
                    segment_start = now
                    last_voice_time = now
                    current_segment = bytearray().join(pre_buffer)
                    current_segment.extend(frame)
            else:
                current_segment.extend(frame)
                if voiced:
                    last_voice_time = now

                elapsed_silence = (now - last_voice_time) if last_voice_time else 0.0
                segment_frame_count = len(current_segment) // self.frame_bytes

                should_flush = elapsed_silence >= self.silence_timeout_s
                reached_max = segment_frame_count >= self.max_segment_frames

                if should_flush or reached_max:
                    chunk = AudioChunk(
                        pcm16=bytes(current_segment),
                        sample_rate=self.sample_rate,
                        channels=self.channels,
                        started_at=segment_start or now,
                        ended_at=now,
                    )
                    await self._emit_chunk(chunk)

                    # reset state
                    pre_buffer.clear()
                    current_segment = bytearray()
                    segment_start = None
                    last_voice_time = None
                    is_speaking = False

    async def _emit_chunk(self, chunk: AudioChunk) -> None:
        if chunk.duration_seconds <= 0:
            return

        if self._chunk_queue.full():
            try:
                self._chunk_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

        try:
            self._chunk_queue.put_nowait(chunk)
            logger.debug("Emitted chunk duration=%.2fs", chunk.duration_seconds)
        except asyncio.QueueFull:
            logger.warning("Chunk queue full; dropping chunk")

    @staticmethod
    def _drain_queue(queue: asyncio.Queue) -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break


# local import at bottom to avoid widening public API.
import contextlib  # noqa: E402


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    async def demo() -> None:
        manager = AudioManager()
        await manager.start()
        print("Listening... speak into your mic. Press Ctrl+C to stop.")
        try:
            async for chunk in manager.chunks():
                print(
                    f"Chunk: duration={chunk.duration_seconds:.2f}s, "
                    f"bytes={len(chunk.pcm16)}"
                )
        finally:
            await manager.stop()

    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        pass
