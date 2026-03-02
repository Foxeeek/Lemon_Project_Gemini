"""Module 2: Faster-Whisper transcription engine for SpeakPilot."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    started_at: float
    ended_at: float
    created_at: datetime


class Transcriber:
    """Loads faster-whisper once and transcribes audio chunks asynchronously."""

    def __init__(
        self,
        model_size: str = "tiny.en",
        cpu_threads: int = 4,
        beam_size: int = 1,
    ) -> None:
        self.model_size = model_size
        self.cpu_threads = cpu_threads
        self.beam_size = beam_size

        self.device = "cpu"
        self.compute_type = "int8"

        logger.info(
            "Loading whisper model size=%s device=%s compute_type=%s",
            self.model_size,
            self.device,
            self.compute_type,
        )
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
        )

    @staticmethod
    def pcm16_bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
        """Convert signed int16 PCM bytes into normalized float32 [-1, 1]."""
        if not audio_bytes:
            return np.empty(0, dtype=np.float32)

        pcm = np.frombuffer(audio_bytes, dtype=np.int16)
        if pcm.size == 0:
            return np.empty(0, dtype=np.float32)

        # 32768 keeps -32768 mapping to -1.0 exactly.
        audio = pcm.astype(np.float32) / 32768.0
        return np.clip(audio, -1.0, 1.0)

    async def transcribe_bytes(self, audio_bytes: bytes) -> Optional[str]:
        """Asynchronously transcribe raw PCM16 bytes; returns cleaned text or None."""
        audio_np = self.pcm16_bytes_to_float32(audio_bytes)
        if audio_np.size == 0:
            return None

        text = await asyncio.to_thread(self._transcribe_array_sync, audio_np)
        cleaned = self._clean_and_filter(text)
        return cleaned

    def _transcribe_array_sync(self, audio_np: np.ndarray) -> str:
        """Blocking whisper call, isolated for thread offloading."""
        try:
            segments, _ = self.model.transcribe(
                audio_np,
                language="en",
                beam_size=self.beam_size,
                vad_filter=False,
                condition_on_previous_text=False,
                temperature=0.0,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
            return text
        except Exception:
            logger.exception("Whisper transcription failed")
            return ""

    @staticmethod
    def _clean_and_filter(text: str) -> Optional[str]:
        if not text:
            return None

        normalized = re.sub(r"\s+", " ", text).strip()
        lower = normalized.lower()

        if not normalized:
            return None
        if lower in {"[blank_audio]", "blank_audio", "[ silence ]", "silence"}:
            return None

        # Drop ultra-short common silence hallucinations.
        hallucinations = {
            "thank you",
            "thanks",
            "you",
            "okay",
            "ok",
            "bye",
            "goodbye",
        }
        tokens = re.findall(r"[a-zA-Z']+", lower)
        if len(tokens) <= 2 and " ".join(tokens) in hallucinations:
            return None

        # Drop 1-word transcripts to suppress noise.
        if len(tokens) < 2:
            return None

        return normalized
