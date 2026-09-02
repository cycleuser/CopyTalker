"""
Unified translation interface supporting multiple backends.
"""

import logging
import os
from typing import List, Optional

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

    Order of preference (lightweight-first):
    1. Helsinki-NLP - per-pair opus-mt models (~300MB, fast, good quality)
    2. NLLB-200 - multilingual distilled-600M (~1.2GB) fallback
    3. Ollama - local LLM, larger but supports conversation context

    A user can override with preferred_model ('helsinki'/'nllb'/'ollama').
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

        # Backends are lazily instantiated to avoid importing heavy deps
        # (e.g. torch) until a translation is actually requested.
        self._helsinki = None
        self._nllb = None
        self._ollama = None

    @property
    def helsinki(self) -> HelsinkiTranslator:
        if self._helsinki is None:
            self._helsinki = HelsinkiTranslator(self.config)
        return self._helsinki

    @property
    def nllb(self) -> NLLBTranslator:
        if self._nllb is None:
            self._nllb = NLLBTranslator(self.config)
        return self._nllb

    @property
    def ollama(self) -> OllamaTranslator:
        if self._ollama is None:
            self._ollama = OllamaTranslator(
                self.config,
                ollama_url=self._ollama_url,
                model_name=self._ollama_model,
            )
        return self._ollama

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

        # Backend selection order (lightweight-first):
        #   1. Helsinki-NLP opus-mt  ~300MB per pair, fast, good quality
        #   2. NLLB-200 distilled-600M ~1.2GB, multilingual fallback
        #   3. Ollama local LLM        larger but supports conversation context
        # Check if Helsinki-NLP has a specific model for this pair
        models = get_translation_models(source_lang, target_lang)
        for model in models:
            if model.startswith("Helsinki-NLP"):
                logger.debug("Helsinki-NLP model available, using for translation")
                return "helsinki"

        # NLLB as multilingual fallback (smaller than full Ollama LLM)
        if self.nllb.supports_pair(source_lang, target_lang):
            logger.debug("Using NLLB for multilingual translation")
            return "nllb"

        # Last resort: Ollama if available (best for conversation mode)
        if self.ollama.is_available():
            logger.debug("Falling back to Ollama for translation")
            return "ollama"

        # If nothing matched, default to NLLB (will lazy-load on use)
        return "nllb"

    def supports_pair(self, source_lang: str, target_lang: str) -> bool:
        """Check if language pair is supported by any backend."""
        return (
            self.ollama.supports_pair(source_lang, target_lang)
            or self.helsinki.supports_pair(source_lang, target_lang)
            or self.nllb.supports_pair(source_lang, target_lang)
        )

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[List[TranslationResult]] = None,
    ) -> TranslationResult:
        """
        Translate text using the best available backend.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            context: Optional prior turns for conversation-aware translation.
                Only the Ollama (LLM) backend currently uses this; MT backends
                ignore it.

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
                return self.ollama.translate(text, source_lang, target_lang, context)
            elif backend == "helsinki":
                return self.helsinki.translate(text, source_lang, target_lang, context)
            else:
                return self.nllb.translate(text, source_lang, target_lang, context)

        except UnsupportedLanguageError:
            # Try fallback to NLLB
            if backend == "helsinki":
                logger.warning("Helsinki-NLP failed, falling back to NLLB")
                return self.nllb.translate(text, source_lang, target_lang, context)
            # Try Ollama if Helsinki/NLLB failed
            if backend != "ollama" and self.ollama.is_available():
                logger.warning(f"{backend} failed, falling back to Ollama")
                return self.ollama.translate(text, source_lang, target_lang, context)
            raise

    def unload_models(self) -> None:
        """Unload all loaded models."""
        if self._helsinki is not None:
            self._helsinki.unload_models()
        if self._nllb is not None:
            self._nllb.unload()
        if self._ollama is not None:
            self._ollama.unload_models()
        logger.info("All translation models unloaded")

    def get_available_models(self, source_lang: str, target_lang: str) -> list:
        """Get list of available models for a language pair."""
        return get_translation_models(source_lang, target_lang)
