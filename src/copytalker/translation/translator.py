"""
Unified translation interface supporting multiple backends.
"""

import logging
from typing import Optional

from copytalker.core.config import TranslationConfig
from copytalker.core.constants import get_translation_models
from copytalker.core.exceptions import UnsupportedLanguageError, ModelError
from copytalker.core.types import TranslationResult
from copytalker.translation.helsinki import HelsinkiTranslator
from copytalker.translation.nllb import NLLBTranslator

logger = logging.getLogger(__name__)


class UnifiedTranslator:
    """
    Unified translation interface that automatically selects the best backend.
    
    Prefers language-specific Helsinki-NLP models when available,
    falls back to NLLB-200 for other language pairs.
    """
    
    def __init__(
        self,
        config: Optional[TranslationConfig] = None,
        preferred_model: Optional[str] = None,
    ):
        """
        Initialize unified translator.
        
        Args:
            config: Translation configuration
            preferred_model: Preferred model name (overrides auto-selection)
        """
        self.config = config or TranslationConfig()
        self._preferred_model = preferred_model or self.config.model_name
        
        self._helsinki = HelsinkiTranslator(self.config)
        self._nllb = NLLBTranslator(self.config)
        
    def _select_backend(self, source_lang: str, target_lang: str) -> str:
        """
        Select the best translation backend for a language pair.
        
        Returns:
            'helsinki' or 'nllb'
        """
        # Check if user explicitly selected a model type
        if self._preferred_model:
            model_lower = self._preferred_model.lower()
            if model_lower in ("helsinki", "helsinki-nlp"):
                return "helsinki"
            elif model_lower in ("nllb", "facebook/nllb"):
                return "nllb"
            elif self._preferred_model.startswith("Helsinki-NLP"):
                return "helsinki"
            elif self._preferred_model.startswith("facebook/nllb"):
                return "nllb"
        
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
            self._helsinki.supports_pair(source_lang, target_lang) or
            self._nllb.supports_pair(source_lang, target_lang)
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
            if backend == "helsinki":
                return self._helsinki.translate(text, source_lang, target_lang)
            else:
                return self._nllb.translate(text, source_lang, target_lang)
                
        except UnsupportedLanguageError:
            # Try fallback to NLLB
            if backend == "helsinki":
                logger.warning(f"Helsinki-NLP failed, falling back to NLLB")
                return self._nllb.translate(text, source_lang, target_lang)
            raise
    
    def unload_models(self) -> None:
        """Unload all loaded models."""
        self._helsinki.unload_models()
        self._nllb.unload()
        logger.info("All translation models unloaded")
    
    def get_available_models(self, source_lang: str, target_lang: str) -> list:
        """Get list of available models for a language pair."""
        return get_translation_models(source_lang, target_lang)
