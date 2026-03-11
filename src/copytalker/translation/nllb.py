"""
Facebook NLLB-200 translation backend.
"""

import logging
import os
import re
import time
from typing import Dict, Optional

import torch

from copytalker.core.config import TranslationConfig
from copytalker.core.constants import get_nllb_code, get_translation_models
from copytalker.core.exceptions import ModelError
from copytalker.core.types import TranslationResult

logger = logging.getLogger(__name__)


class NLLBTranslator:
    """
    Translation using Facebook NLLB-200 multilingual model.

    Supports translation between 200+ languages with a single model.
    """

    DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"

    def __init__(
        self,
        config: Optional[TranslationConfig] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize NLLB translator.

        Args:
            config: Translation configuration
            model_name: Specific NLLB model to use (defaults to distilled-600M)
        """
        self.config = config or TranslationConfig()

        self._model_name = self._resolve_model_name(model_name, self.config.model_name)

        self._model = None
        self._tokenizer = None
        self._device = self._resolve_device()
        self._loaded = False

        # Disable meta tensor loading
        os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"

    def _resolve_device(self) -> str:
        """Resolve the actual device to use."""
        device = self.config.device

        if device == "cuda":
            if not torch.cuda.is_available():
                logger.info("CUDA not available, using CPU for NLLB")
                return "cpu"
        elif device == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                logger.info("MPS not available, using CPU for NLLB")
                return "cpu"
        elif device == "rocm":
            if not torch.cuda.is_available():
                logger.info("ROCm not available, using CPU for NLLB")
                return "cpu"

        return device

    def _resolve_model_name(self, explicit: Optional[str], from_config: Optional[str]) -> str:
        """Resolve actual HuggingFace model name, ignoring backend selectors."""
        BACKEND_SELECTORS = {"helsinki", "helsinki-nlp", "nllb", "auto", ""}

        for candidate in (explicit, from_config):
            if candidate and candidate.lower().strip() not in BACKEND_SELECTORS:
                if "/" in candidate:
                    return candidate

        return self.DEFAULT_MODEL

    def _ensure_model(self) -> None:
        """Lazy-load the NLLB model."""
        if self._loaded:
            return

        logger.info(f"Loading NLLB model: {self._model_name}")

        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

            # Fix for "Cannot copy out of meta tensor" error
            try:
                self._model = AutoModelForSeq2SeqLM.from_pretrained(
                    self._model_name,
                    low_cpu_mem_usage=False,
                )
            except TypeError:
                # Older transformers version
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)

            # Move to device
            try:
                self._model = self._model.to(self._device)
            except RuntimeError as e:
                if "meta tensor" in str(e).lower() or "no data" in str(e).lower():
                    logger.warning(f"Meta tensor issue, reloading model: {e}")
                    del self._model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

                    self._model = AutoModelForSeq2SeqLM.from_pretrained(
                        self._model_name,
                        low_cpu_mem_usage=False,
                        torch_dtype=torch.float32,
                    )
                    self._model = self._model.to(self._device)
                else:
                    raise

            self._model.eval()

            self._loaded = True
            logger.info(f"NLLB model loaded on {self._device}")

        except Exception as e:
            logger.error(f"Failed to load NLLB model: {e}")
            raise ModelError(f"Failed to load NLLB model: {e}") from e

    def supports_pair(self, source_lang: str, target_lang: str) -> bool:
        """
        Check if language pair is supported.

        NLLB supports most language pairs.
        """
        try:
            src_code = get_nllb_code(source_lang)
            tgt_code = get_nllb_code(target_lang)
            return src_code is not None and tgt_code is not None
        except:
            return False

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        """
        Translate text using NLLB model.

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

        self._ensure_model()

        start_time = time.time()

        src_lang_code = get_nllb_code(source_lang)
        tgt_lang_code = get_nllb_code(target_lang)

        try:
            self._tokenizer.src_lang = src_lang_code

            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(tgt_lang_code)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=self.config.max_length,
                )

            translated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            translated = re.sub(r"\s+", " ", translated).strip()

            duration = time.time() - start_time

            logger.info(
                f"NLLB translated ({source_lang}->{target_lang}): "
                f"'{text[:50]}...' -> '{translated[:50]}...' ({duration:.2f}s)"
            )

            return TranslationResult(
                original_text=text,
                translated_text=translated,
                source_lang=source_lang,
                target_lang=target_lang,
                model_used=self._model_name,
            )

        except Exception as e:
            logger.error(f"NLLB translation error: {e}")
            raise ModelError(f"NLLB translation failed: {e}") from e

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._loaded = False
            logger.info("NLLB model unloaded")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
