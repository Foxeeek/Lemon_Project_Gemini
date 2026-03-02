"""SpeakPilot entrypoint integrating async backend with PyQt6 overlay UI."""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from speakpilot.audio_manager import AudioChunk, AudioManager
from speakpilot.corrector import Corrector
from speakpilot.overlay_ui import FloatingWindow
from speakpilot.transcriber import Transcriber

logger = logging.getLogger(__name__)


class BackendBridge(QObject):
    correction_ready = pyqtSignal(dict)
    backend_error = pyqtSignal(str)
    backend_stopped = pyqtSignal()


async def produce_audio_chunks(
    audio_manager: AudioManager,
    audio_queue: asyncio.Queue[AudioChunk],
    stop_event: threading.Event,
) -> None:
    async for chunk in audio_manager.chunks():
        if stop_event.is_set():
            return
        await audio_queue.put(chunk)


async def consume_transcriptions(
    transcriber: Transcriber,
    audio_queue: asyncio.Queue[AudioChunk],
    correction_queue: asyncio.Queue[str],
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.2)
        except asyncio.TimeoutError:
            continue

        try:
            text = await transcriber.transcribe_bytes(chunk.pcm16)
            if text:
                await correction_queue.put(text)
        finally:
            audio_queue.task_done()


async def consume_corrections(
    corrector: Corrector,
    correction_queue: asyncio.Queue[str],
    bridge: BackendBridge,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            text = await asyncio.wait_for(correction_queue.get(), timeout=0.2)
        except asyncio.TimeoutError:
            continue

        try:
            result = await corrector.analyze_text(text)
            if result:
                bridge.correction_ready.emit(result)
        finally:
            correction_queue.task_done()


async def backend_main(bridge: BackendBridge, stop_event: threading.Event) -> None:
    audio_manager = AudioManager()
    transcriber = Transcriber(model_size="tiny.en")
    corrector = Corrector(model="gpt-4o-mini")

    audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=16)
    correction_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=32)

    producer_task: Optional[asyncio.Task] = None
    transcription_task: Optional[asyncio.Task] = None
    correction_task: Optional[asyncio.Task] = None

    try:
        await audio_manager.start()
        producer_task = asyncio.create_task(
            produce_audio_chunks(audio_manager, audio_queue, stop_event),
            name="audio-producer",
        )
        transcription_task = asyncio.create_task(
            consume_transcriptions(transcriber, audio_queue, correction_queue, stop_event),
            name="transcription-consumer",
        )
        correction_task = asyncio.create_task(
            consume_corrections(corrector, correction_queue, bridge, stop_event),
            name="correction-consumer",
        )

        while not stop_event.is_set():
            await asyncio.sleep(0.2)
    except Exception as exc:
        logger.exception("Backend pipeline crashed")
        bridge.backend_error.emit(str(exc))
    finally:
        for task in (producer_task, transcription_task, correction_task):
            if task is not None:
                task.cancel()
        for task in (producer_task, transcription_task, correction_task):
            if task is not None:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("Error while stopping task %s", task.get_name())

        await audio_manager.stop()
        bridge.backend_stopped.emit()


def run_backend_thread(bridge: BackendBridge, stop_event: threading.Event) -> None:
    asyncio.run(backend_main(bridge, stop_event))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app = QApplication(sys.argv)
    overlay = FloatingWindow()
    bridge = BackendBridge()
    stop_event = threading.Event()

    bridge.correction_ready.connect(overlay.update_from_result)
    bridge.backend_error.connect(lambda msg: logger.error("Backend error: %s", msg))
    bridge.backend_stopped.connect(lambda: logger.info("Backend stopped"))

    backend_thread = threading.Thread(
        target=run_backend_thread,
        args=(bridge, stop_event),
        name="speakpilot-backend",
        daemon=True,
    )
    backend_thread.start()

    def on_quit() -> None:
        stop_event.set()
        backend_thread.join(timeout=3)

    app.aboutToQuit.connect(on_quit)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
