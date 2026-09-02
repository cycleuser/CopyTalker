"""
Ollama local LLM translation backend.
"""

import logging
import time
from typing import List, Optional

import requests

from copytalker.core.config import TranslationConfig
from copytalker.core.exceptions import ModelError
from copytalker.core.types import TranslationResult

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:0.6b"

LANG_CODE_TO_NAME = {
    "auto": "auto-detect",
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
    "it": "Italian",
    "pt": "Portuguese",
    "ar": "Arabic",
    "vi": "Vietnamese",
    "th": "Thai",
    "hi": "Hindi",
}


class OllamaTranslator:
    """
    Translation using Ollama local LLM.

    Supports any Ollama model that supports instruction following.
    """

    def __init__(
        self,
        config: Optional[TranslationConfig] = None,
        ollama_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize Ollama translator.

        Args:
            config: Translation configuration
            ollama_url: Ollama server URL (default: http://localhost:11434)
            model_name: Ollama model name (default: qwen3:0.6b)
        """
        self.config = config or TranslationConfig()
        self._ollama_url = (ollama_url or DEFAULT_OLLAMA_URL).rstrip("/")
        self._model = model_name or DEFAULT_MODEL
        self._available = None

    def _check_availability(self) -> bool:
        """Check if Ollama server is available."""
        if self._available is not None:
            return self._available
        try:
            resp = requests.get(f"{self._ollama_url}/api/tags", timeout=5)
            self._available = resp.status_code == 200
            if self._available:
                logger.info(f"Ollama server available at {self._ollama_url}")
            else:
                logger.warning(f"Ollama server returned {resp.status_code}")
        except requests.exceptions.ConnectionError:
            self._available = False
            logger.warning(f"Ollama server not available at {self._ollama_url}")
        except Exception as e:
            self._available = False
            logger.error(f"Ollama connection error: {e}")
        return self._available

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        return self._check_availability()

    def list_models(self) -> list[str]:
        """List available models on Ollama server."""
        if not self._check_availability():
            return []
        try:
            resp = requests.get(f"{self._ollama_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
        return []

    def _build_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[List[TranslationResult]] = None,
    ) -> str:
        """Build translation prompt for the LLM, optionally with prior turns."""
        src_name = LANG_CODE_TO_NAME.get(source_lang, source_lang)
        tgt_name = LANG_CODE_TO_NAME.get(target_lang, target_lang)

        parts: list[str] = []
        parts.append(
            f"You are a professional translator. Translate the following text "
            f"from {src_name} to {tgt_name}. "
            f"Only output the translation, nothing else."
        )

        if context:
            parts.append("\nPrevious turns (use for context/coherence):")
            for turn in context:
                src_label = LANG_CODE_TO_NAME.get(turn.source_lang, turn.source_lang)
                tgt_label = LANG_CODE_TO_NAME.get(turn.target_lang, turn.target_lang)
                parts.append(f"Source ({src_label}): {turn.original_text}")
                parts.append(f"Translation ({tgt_label}): {turn.translated_text}")

        parts.append(f"\nSource ({src_name}): {text}")
        parts.append(f"\nTranslation ({tgt_name}):")
        return "\n".join(parts)

    def supports_pair(self, source_lang: str, target_lang: str) -> bool:
        """Check if this backend supports the language pair.

        An LLM can attempt any pair, but we only claim support for languages
        we can name in the prompt. Availability of the server is checked too.
        """
        if not self._check_availability():
            return False
        known = set(LANG_CODE_TO_NAME.keys())
        # 'auto' source is always acceptable
        src_ok = source_lang in known or source_lang == "auto"
        tgt_ok = target_lang in known
        return src_ok and tgt_ok

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[List[TranslationResult]] = None,
    ) -> TranslationResult:
        """
        Translate text using Ollama LLM.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            context: Optional prior turns for conversation-aware translation.

        Returns:
            TranslationResult with translated text
        """
        if not self._check_availability():
            raise ModelError(
                f"Ollama not available at {self._ollama_url}. "
                f"Please start Ollama or select a different translation backend."
            )

        if source_lang == target_lang:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                model_used="none",
            )

        start_time = time.time()

        prompt = self._build_prompt(text, source_lang, target_lang, context)

        try:
            payload = {
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 512,
                },
            }
            resp = requests.post(
                f"{self._ollama_url}/api/generate",
                json=payload,
                timeout=120,
            )

            if resp.status_code != 200:
                raise ModelError(f"Ollama API error: {resp.status_code} - {resp.text}")

            data = resp.json()
            translated = data.get("response", "").strip()

            duration = time.time() - start_time

            logger.info(
                f"Translated via Ollama ({source_lang}->{target_lang}): "
                f"'{text[:50]}...' -> '{translated[:50]}...' ({duration:.2f}s)"
            )

            return TranslationResult(
                original_text=text,
                translated_text=translated,
                source_lang=source_lang,
                target_lang=target_lang,
                model_used=f"ollama/{self._model}",
            )

        except requests.exceptions.Timeout as e:
            raise ModelError("Ollama request timed out") from e
        except requests.exceptions.ConnectionError as e:
            raise ModelError(f"Cannot connect to Ollama at {self._ollama_url}") from e
        except Exception as e:
            logger.error(f"Ollama translation error: {e}")
            raise ModelError(f"Translation failed: {e}") from e

    def unload_models(self) -> None:
        """Unload models (Ollama manages this server-side)."""
        logger.info("Ollama translation unloaded (server-side management)")
