"""SpeakPilot entrypoint for Module 1 + Module 2 pipeline."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime

from speakpilot.audio_manager import AudioChunk, AudioManager
from speakpilot.transcriber import Transcriber

logger = logging.getLogger(__name__)


async def produce_audio_chunks(audio_manager: AudioManager, queue: asyncio.Queue[AudioChunk]) -> None:
    """Read chunk stream from AudioManager and enqueue for transcription."""
    async for chunk in audio_manager.chunks():
        await queue.put(chunk)


async def consume_transcriptions(
    transcriber: Transcriber,
    queue: asyncio.Queue[AudioChunk],
) -> None:
    """Transcribe chunks and print final text with timestamps."""
    while True:
        chunk = await queue.get()
        try:
            text = await transcriber.transcribe_bytes(chunk.pcm16)
            if text:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] {text}")
        finally:
            queue.task_done()


async def run() -> None:
    audio_manager = AudioManager()
    transcriber = Transcriber(model_size="tiny.en")
    queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=16)

    await audio_manager.start()

    producer_task = asyncio.create_task(produce_audio_chunks(audio_manager, queue), name="audio-producer")
    consumer_task = asyncio.create_task(consume_transcriptions(transcriber, queue), name="transcription-consumer")

    logger.info("SpeakPilot started. Press Ctrl+C to stop.")

    try:
        await asyncio.gather(producer_task, consumer_task)
    except asyncio.CancelledError:
        raise
    finally:
        producer_task.cancel()
        consumer_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await producer_task
        with contextlib.suppress(asyncio.CancelledError):
            await consumer_task
        await audio_manager.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Shutting down SpeakPilot")
