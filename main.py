"""SpeakPilot entrypoint integrating async backend with PyQt6 overlay UI."""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from speakpilot.audio_manager import AudioChunk, AudioManager
from speakpilot.config import (
    CHANNELS,
    DEBOUNCE_SILENCE_SECONDS,
    INTERVIEW_DEBOUNCE_SILENCE_SECONDS,
    INTERVIEW_MODE,
    MAX_SEGMENT_SECONDS,
    MIN_WORDS_FOR_CORRECTION,
    SAMPLE_RATE,
)
from speakpilot.corrector import Corrector
from speakpilot.overlay_ui import FloatingWindow
from speakpilot.transcriber import Transcriber

logger = logging.getLogger(__name__)


class BackendBridge(QObject):
    correction_ready = pyqtSignal(dict)
    interview_summary = pyqtSignal(dict)
    backend_error = pyqtSignal(str)
    backend_stopped = pyqtSignal()


@dataclass
class SessionStats:
    total_sentences: int = 0
    total_corrections: int = 0
    error_counter: Counter = field(default_factory=Counter)

    def record(self, result: dict) -> None:
        self.total_sentences += 1
        if result.get("is_error"):
            self.total_corrections += 1
            key = self._normalize_explanation(str(result.get("explanation", "")))
            if key:
                self.error_counter[key] += 1

    @staticmethod
    def _normalize_explanation(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def summary(self) -> dict:
        if self.total_sentences == 0:
            return {
                "score": 100,
                "error_rate": 0.0,
                "total_sentences": 0,
                "total_corrections": 0,
                "common_error": "None",
            }

        error_rate = self.total_corrections / self.total_sentences
        score = max(0, int(round(100 - (error_rate * 100))))
        common_error = self.error_counter.most_common(1)[0][0] if self.error_counter else "None"
        return {
            "score": score,
            "error_rate": error_rate,
            "total_sentences": self.total_sentences,
            "total_corrections": self.total_corrections,
            "common_error": common_error,
        }


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z']+", text))


def has_terminal_punctuation(text: str) -> bool:
    return text.rstrip().endswith((".", "?", "!"))


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
    correction_in_flight: threading.Event,
    stop_event: threading.Event,
) -> None:
    pending_text = ""
    last_update = 0.0
    silence_threshold = INTERVIEW_DEBOUNCE_SILENCE_SECONDS if INTERVIEW_MODE else DEBOUNCE_SILENCE_SECONDS

    while not stop_event.is_set():
        try:
            chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
            got_item = True
        except asyncio.TimeoutError:
            got_item = False

        if got_item:
            try:
                text = await transcriber.transcribe_bytes(chunk.pcm16)
                if text:
                    pending_text = text
                    last_update = time.monotonic()

                    if (
                        word_count(pending_text) >= MIN_WORDS_FOR_CORRECTION
                        and has_terminal_punctuation(pending_text)
                        and not correction_in_flight.is_set()
                    ):
                        await correction_queue.put(pending_text)
                        pending_text = ""
                        last_update = 0.0
            finally:
                audio_queue.task_done()
            continue

        if (
            pending_text
            and word_count(pending_text) >= MIN_WORDS_FOR_CORRECTION
            and (time.monotonic() - last_update) >= silence_threshold
            and not correction_in_flight.is_set()
        ):
            await correction_queue.put(pending_text)
            pending_text = ""
            last_update = 0.0


async def consume_corrections(
    corrector: Corrector,
    correction_queue: asyncio.Queue[str],
    bridge: BackendBridge,
    stats: SessionStats,
    correction_in_flight: threading.Event,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            text = await asyncio.wait_for(correction_queue.get(), timeout=0.2)
        except asyncio.TimeoutError:
            continue

        if correction_in_flight.is_set():
            correction_queue.task_done()
            continue

        correction_in_flight.set()
        try:
            result = await corrector.analyze_text(text)
            if result:
                stats.record(result)
                if result.get("is_error"):
                    bridge.correction_ready.emit(result)
        finally:
            correction_in_flight.clear()
            correction_queue.task_done()


async def backend_main(bridge: BackendBridge, stop_event: threading.Event) -> None:
    audio_manager = AudioManager(sample_rate=SAMPLE_RATE, channels=CHANNELS, max_segment_s=MAX_SEGMENT_SECONDS)
    transcriber = Transcriber(model_size="tiny.en")
    corrector = Corrector(model="gpt-4o-mini")

    audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=16)
    correction_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=8)

    correction_in_flight = threading.Event()
    stats = SessionStats()

    producer_task: Optional[asyncio.Task] = None
    transcription_task: Optional[asyncio.Task] = None
    correction_task: Optional[asyncio.Task] = None

    try:
        await audio_manager.start()
        producer_task = asyncio.create_task(produce_audio_chunks(audio_manager, audio_queue, stop_event), name="audio-producer")
        transcription_task = asyncio.create_task(
            consume_transcriptions(transcriber, audio_queue, correction_queue, correction_in_flight, stop_event),
            name="transcription-consumer",
        )
        correction_task = asyncio.create_task(
            consume_corrections(corrector, correction_queue, bridge, stats, correction_in_flight, stop_event),
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

        if INTERVIEW_MODE:
            bridge.interview_summary.emit(stats.summary())

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
    bridge.interview_summary.connect(
        lambda s: overlay.show_summary(
            summary_text=(
                f"Interview Score: {s['score']}/100 | "
                f"Sentences: {s['total_sentences']} | Corrections: {s['total_corrections']}"
            ),
            common_error=s["common_error"],
        )
    )
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
