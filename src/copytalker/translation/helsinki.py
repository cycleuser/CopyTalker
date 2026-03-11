"""
Helsinki-NLP MarianMT translation backend.
"""

import logging
import re
import time
from typing import Dict, Optional

import torch

from copytalker.core.config import TranslationConfig
from copytalker.core.constants import get_translation_models
from copytalker.core.exceptions import ModelError, UnsupportedLanguageError
from copytalker.core.types import TranslationResult

logger = logging.getLogger(__name__)


class HelsinkiTranslator:
    """
    Translation using Helsinki-NLP MarianMT models.

    Supports language-pair specific models for higher quality.
    """

    def __init__(self, config: Optional[TranslationConfig] = None):
        """
        Initialize Helsinki translator.

        Args:
            config: Translation configuration
        """
        self.config = config or TranslationConfig()

        self._models: Dict[str, any] = {}
        self._tokenizers: Dict[str, any] = {}
        self._device = self._device = self._resolve_device()

    def _resolve_device(self) -> str:
        """Resolve the actual device to use."""
        device = self.config.device

        if device == "cuda":
            if not torch.cuda.is_available():
                logger.info("CUDA not available, falling back to CPU")
                return "cpu"
        elif device == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                logger.info("MPS not available, falling back to CPU")
                return "cpu"
        elif device == "rocm":
            if not torch.cuda.is_available():
                logger.info("ROCm not available, falling back to CPU")
                return "cpu"

        return device

    def _get_model_name(self, source_lang: str, target_lang: str) -> Optional[str]:
        """Get the Helsinki-NLP model name for a language pair."""
        models = get_translation_models(source_lang, target_lang)

        for model in models:
            if model.startswith("Helsinki-NLP/opus-mt-"):
                return model

        return None

    def _load_model(self, model_name: str) -> tuple:
        """Load a translation model and tokenizer."""
        if model_name in self._models:
            return self._tokenizers[model_name], self._models[model_name]

        logger.info(f"Loading Helsinki-NLP model: {model_name}")

        try:
            from transformers import MarianMTModel, MarianTokenizer

            tokenizer = MarianTokenizer.from_pretrained(model_name)

            # Fix for "Cannot copy out of meta tensor" error
            # Load model without low_cpu_mem_usage to avoid meta tensor issues
            try:
                # Try loading directly to target device
                model = MarianMTModel.from_pretrained(
                    model_name,
                    low_cpu_mem_usage=False,
                )
            except TypeError:
                # Older transformers version doesn't support low_cpu_mem_usage
                model = MarianMTModel.from_pretrained(model_name)

            # Move to device
            try:
                model = model.to(self._device)
            except RuntimeError as e:
                if "meta tensor" in str(e).lower() or "no data" in str(e).lower():
                    # If meta tensor error, load fresh without optimization
                    logger.warning(f"Meta tensor issue, reloading model: {e}")
                    del model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

                    model = MarianMTModel.from_pretrained(
                        model_name,
                        low_cpu_mem_usage=False,
                        torch_dtype=torch.float32,
                    )
                    model = model.to(self._device)
                else:
                    raise

            model.eval()

            self._tokenizers[model_name] = tokenizer
            self._models[model_name] = model

            logger.info(f"Loaded {model_name} on {self._device}")

            return tokenizer, model

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise ModelError(f"Failed to load translation model: {e}") from e

    def supports_pair(self, source_lang: str, target_lang: str) -> bool:
        """Check if language pair is supported."""
        return self._get_model_name(source_lang, target_lang) is not None

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        """
        Translate text using Helsinki-NLP model.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            TranslationResult with translated text
        """
        if source_lang == target_lang:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                model_used="none",
            )

        model_name = self._get_model_name(source_lang, target_lang)
        if not model_name:
            raise UnsupportedLanguageError(
                f"No Helsinki-NLP model for {source_lang} -> {target_lang}"
            )

        start_time = time.time()

        tokenizer, model = self._load_model(model_name)

        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_length,
                )

            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translated = re.sub(r"\s+", " ", translated).strip()

            duration = time.time() - start_time

            logger.info(
                f"Translated ({source_lang}->{target_lang}): "
                f"'{text[:50]}...' -> '{translated[:50]}...' ({duration:.2f}s)"
            )

            return TranslationResult(
                original_text=text,
                translated_text=translated,
                source_lang=source_lang,
                target_lang=target_lang,
                model_used=model_name,
            )

        except Exception as e:
            logger.error(f"Translation error: {e}")
            raise ModelError(f"Translation failed: {e}") from e

    def unload_models(self) -> None:
        """Unload all loaded models."""
        self._models.clear()
        self._tokenizers.clear()
        logger.info("Helsinki-NLP models unloaded")
