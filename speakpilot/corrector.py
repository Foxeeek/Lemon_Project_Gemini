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

SYSTEM_PROMPT = "You are an expert English coach for non-native IT professionals. You are analyzing LIVE transcribed speech-to-text (STT) output. IMPORTANT: The input may contain STT hallucinations, phonetic misspellings, or cut-off words (e.g. 'except' instead of 'accept', 'right' instead of 'write'). FIRST, silently infer what the user ACTUALLY meant based on IT/conversational context. SECOND, check for actual grammatical errors or awkward phrasing in their intended meaning. ALWAYS respond in pure JSON format: {\"original\": \"the exact input\", \"corrected\": \"the better version\", \"is_error\": true/false, \"explanation\": \"Short 3-6 word reason\"}. If it's just a minor STT phonetic typo but the grammar of the intended sentence is fine, return is_error: false."



class Corrector:
    """Async LLM correction engine using OpenAI async client."""

    def __init__(self, model: str = "gpt-4o-mini", timeout: float = 8.0) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to your environment or .env file.")

        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout)

    async def analyze_text(self, text: str) -> Optional[Dict[str, Any]]:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if len(re.findall(r"[A-Za-z']+", normalized)) < 3:
            return None

        content = await self._request_json(normalized)
        if not content:
            return self._fallback(normalized)

        payload = self._safe_parse(content)
        if not payload:
            # Retry once if invalid JSON.
            content_retry = await self._request_json(normalized)
            payload = self._safe_parse(content_retry) if content_retry else None

        if not payload:
            return self._fallback(normalized)

        corrected = str(payload.get("corrected", normalized)).strip() or normalized
        explanation = str(payload.get("explanation", "Grammar adjusted")).strip() or "Grammar adjusted"
        words = explanation.split()
        if len(words) > 15:
            explanation = " ".join(words[:15])

        is_error = corrected != normalized
        return {
            "original": normalized,
            "corrected": corrected,
            "explanation": explanation,
            "is_error": is_error,
        }

    async def _request_json(self, text: str) -> Optional[str]:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            logger.exception("LLM correction call failed")
            return None

    @staticmethod
    def _safe_parse(content: str) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(content)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _fallback(text: str) -> Dict[str, Any]:
        return {
            "original": text,
            "corrected": text,
            "explanation": "No correction available",
            "is_error": False,
        }
