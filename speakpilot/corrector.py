"""Module 3: AI grammar correction engine for SpeakPilot."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an English grammar checker for non-native speakers. Analyze the given text. "
    "If there is a grammatical error, awkward phrasing, or a better way to say it in a professional context, correct it. "
    'ALWAYS respond in pure JSON format without markdown blocks: {"original": "the input", "corrected": "the corrected version", '
    '"is_error": true or false, "explanation": "Short 3-6 word explanation of the error"}. '
    'If there are no errors, return "is_error": false and make "corrected" the same as original.'
)


class Corrector:
    """Async LLM correction engine using OpenAI's latest async client."""

    def __init__(self, model: str = "gpt-4o-mini", timeout: float = 8.0) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to your environment or .env file.")

        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout)

    async def analyze_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Return correction JSON, or None for short input/non-actionable cases."""
        normalized = re.sub(r"\s+", " ", text).strip()
        if self._word_count(normalized) < 3:
            return None

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": normalized},
                ],
                temperature=0,
            )
        except Exception:
            logger.exception("LLM correction call failed")
            return None

        content = (response.choices[0].message.content or "").strip()
        if not content:
            return None

        parsed = self._safe_parse(content)
        if not parsed:
            return None

        original = str(parsed.get("original", normalized)).strip() or normalized
        corrected = str(parsed.get("corrected", original)).strip() or original
        is_error = bool(parsed.get("is_error", False))
        explanation = str(parsed.get("explanation", "")).strip()

        result = {
            "original": original,
            "corrected": corrected,
            "is_error": is_error,
            "explanation": explanation,
        }

        if not result["is_error"]:
            return None

        return result

    @staticmethod
    def _word_count(text: str) -> int:
        return len(re.findall(r"[A-Za-z']+", text))

    @staticmethod
    def _safe_parse(content: str) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(content)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            logger.warning("Model returned invalid JSON: %s", content)
            return None
