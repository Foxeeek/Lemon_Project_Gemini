"""SpeakPilot entrypoint for Module 1 + Module 2 + Module 3 pipeline."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import datetime

from speakpilot.audio_manager import AudioChunk, AudioManager
from speakpilot.corrector import Corrector
from speakpilot.transcriber import Transcriber

logger = logging.getLogger(__name__)

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"


async def produce_audio_chunks(audio_manager: AudioManager, queue: asyncio.Queue[AudioChunk]) -> None:
    """Read chunk stream from AudioManager and enqueue for transcription."""
    async for chunk in audio_manager.chunks():
        await queue.put(chunk)


async def consume_transcriptions(
    transcriber: Transcriber,
    audio_queue: asyncio.Queue[AudioChunk],
    correction_queue: asyncio.Queue[str],
) -> None:
    """Transcribe chunks and forward non-empty text to correction queue."""
    while True:
        chunk = await audio_queue.get()
        try:
            text = await transcriber.transcribe_bytes(chunk.pcm16)
            if text:
                await correction_queue.put(text)
        finally:
            audio_queue.task_done()


async def consume_corrections(corrector: Corrector, queue: asyncio.Queue[str]) -> None:
    """Analyze text using LLM and print formatted JSON to terminal."""
    while True:
        text = await queue.get()
        try:
            result = await corrector.analyze_text(text)
            if result:
                ts = datetime.now().strftime("%H:%M:%S")
                color = RED if result.get("is_error") else GREEN
                payload = json.dumps(result, ensure_ascii=False)
                print(f"{color}[{ts}] {payload}{RESET}")
        finally:
            queue.task_done()


async def run() -> None:
    audio_manager = AudioManager()
    transcriber = Transcriber(model_size="tiny.en")
    corrector = Corrector(model="gpt-4o-mini")

    audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=16)
    correction_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=32)

    await audio_manager.start()

    producer_task = asyncio.create_task(
        produce_audio_chunks(audio_manager, audio_queue),
        name="audio-producer",
    )
    transcription_task = asyncio.create_task(
        consume_transcriptions(transcriber, audio_queue, correction_queue),
        name="transcription-consumer",
    )
    correction_task = asyncio.create_task(
        consume_corrections(corrector, correction_queue),
        name="correction-consumer",
    )

    logger.info("SpeakPilot started. Press Ctrl+C to stop.")

    try:
        await asyncio.gather(producer_task, transcription_task, correction_task)
    except asyncio.CancelledError:
        raise
    finally:
        producer_task.cancel()
        transcription_task.cancel()
        correction_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await producer_task
        with contextlib.suppress(asyncio.CancelledError):
            await transcription_task
        with contextlib.suppress(asyncio.CancelledError):
            await correction_task
        await audio_manager.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Shutting down SpeakPilot")
