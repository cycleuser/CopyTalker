"""
Unified translation interface supporting multiple backends.
"""

import logging
import os
from typing import Optional

from copytalker.core.config import TranslationConfig
from copytalker.core.constants import get_translation_models
from copytalker.core.exceptions import UnsupportedLanguageError
from copytalker.core.types import TranslationResult
from copytalker.translation.helsinki import HelsinkiTranslator
from copytalker.translation.nllb import NLLBTranslator
from copytalker.translation.ollama import DEFAULT_MODEL, DEFAULT_OLLAMA_URL, OllamaTranslator

logger = logging.getLogger(__name__)


class UnifiedTranslator:
    """
    Unified translation interface that automatically selects the best backend.

    Order of preference:
    1. Ollama (local LLM) - if available and configured
    2. Helsinki-NLP - for supported language pairs
    3. NLLB-200 - for multilingual support
    """

    def __init__(
        self,
        config: Optional[TranslationConfig] = None,
        preferred_model: Optional[str] = None,
        ollama_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
    ):
        """
        Initialize unified translator.

        Args:
            config: Translation configuration
            preferred_model: Preferred model name (overrides auto-selection)
            ollama_url: Ollama server URL (default: http://localhost:11434)
            ollama_model: Ollama model name (default: qwen3:0.6b)
        """
        self.config = config or TranslationConfig()
        self._preferred_model = preferred_model or self.config.model_name

        self._ollama_url = ollama_url or os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
        self._ollama_model = ollama_model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)

        self._helsinki = HelsinkiTranslator(self.config)
        self._nllb = NLLBTranslator(self.config)
        self._ollama = OllamaTranslator(
            self.config,
            ollama_url=self._ollama_url,
            model_name=self._ollama_model,
        )

    def _select_backend(self, source_lang: str, target_lang: str) -> str:
        """
        Select the best translation backend for a language pair.

        Returns:
            'ollama', 'helsinki', or 'nllb'
        """
        # Check if user explicitly selected a model type
        if self._preferred_model:
            model_lower = self._preferred_model.lower()
            if model_lower in ("ollama", "llama", "qwen"):
                return "ollama"
            elif model_lower in ("helsinki", "helsinki-nlp"):
                return "helsinki"
            elif model_lower in ("nllb", "facebook/nllb"):
                return "nllb"
            elif self._preferred_model.startswith("Helsinki-NLP"):
                return "helsinki"
            elif self._preferred_model.startswith("facebook/nllb"):
                return "nllb"

        # Prefer Ollama if available (local LLM - good quality, no downloads)
        if self._ollama.is_available():
            logger.debug("Ollama available, using for translation")
            return "ollama"

        # Check if Helsinki-NLP has a specific model
        models = get_translation_models(source_lang, target_lang)
        for model in models:
            if model.startswith("Helsinki-NLP"):
                return "helsinki"

        # Default to NLLB for multilingual support
        return "nllb"

    def supports_pair(self, source_lang: str, target_lang: str) -> bool:
        """Check if language pair is supported by any backend."""
        return (
            self._ollama.supports_pair(source_lang, target_lang)
            or self._helsinki.supports_pair(source_lang, target_lang)
            or self._nllb.supports_pair(source_lang, target_lang)
        )

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        """
        Translate text using the best available backend.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            TranslationResult with translated text
        """
        if not text or not text.strip():
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                model_used="none",
            )

        if source_lang == target_lang:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                model_used="none",
            )

        backend = self._select_backend(source_lang, target_lang)

        logger.debug(f"Using {backend} backend for {source_lang} -> {target_lang}")

        try:
            if backend == "ollama":
                return self._ollama.translate(text, source_lang, target_lang)
            elif backend == "helsinki":
                return self._helsinki.translate(text, source_lang, target_lang)
            else:
                return self._nllb.translate(text, source_lang, target_lang)

        except UnsupportedLanguageError:
            # Try fallback to NLLB
            if backend == "helsinki":
                logger.warning("Helsinki-NLP failed, falling back to NLLB")
                return self._nllb.translate(text, source_lang, target_lang)
            # Try Ollama if Helsinki/NLLB failed
            if backend != "ollama" and self._ollama.is_available():
                logger.warning(f"{backend} failed, falling back to Ollama")
                return self._ollama.translate(text, source_lang, target_lang)
            raise

    def unload_models(self) -> None:
        """Unload all loaded models."""
        self._helsinki.unload_models()
        self._nllb.unload()
        self._ollama.unload_models()
        logger.info("All translation models unloaded")

    def get_available_models(self, source_lang: str, target_lang: str) -> list:
        """Get list of available models for a language pair."""
        return get_translation_models(source_lang, target_lang)
