"""
Kokoro TTS engine - Primary high-quality TTS.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from copytalker.core.config import TTSConfig
from copytalker.core.constants import (
    KOKORO_VOICE_MAP,
    KOKORO_SAMPLE_RATE,
    get_kokoro_lang_code,
)
from copytalker.core.exceptions import TTSError, TTSEngineNotAvailableError
from copytalker.core.types import AudioArray
from copytalker.tts.base import TTSEngineBase

logger = logging.getLogger(__name__)


def setup_hf_mirror():
    """Setup HuggingFace mirror for users in China or with network issues."""
    if os.environ.get("HF_ENDPOINT"):
        return

    if os.environ.get("https_proxy") or os.environ.get("HTTP_PROXY"):
        return

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    logger.info("Using HuggingFace mirror: https://hf-mirror.com")


def setup_torch_for_device():
    """Setup torch to avoid meta tensor issues on certain devices."""
    # Disable accelerated model loading which causes meta tensor issues
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"


class KokoroTTS(TTSEngineBase):
    """
    Kokoro TTS engine - High quality neural TTS.

    Supports English, Chinese, and Japanese with multiple voices.

    For Kokoro v0.9+:
    - Model is auto-downloaded from HuggingFace on first use
    - Supports HF_ENDPOINT environment variable for mirror
    - Supports https_proxy/HTTP_PROXY for proxy
    """

    HF_MODEL_ID = "hexgrad/Kokoro-82M"

    def __init__(self, config: Optional[TTSConfig] = None):
        super().__init__(config)

        self._pipeline = None
        self._is_available = None
        self._init_error = None

        setup_hf_mirror()
        setup_torch_for_device()

    @property
    def name(self) -> str:
        return "kokoro"

    @property
    def is_available(self) -> bool:
        """Check if Kokoro is installed."""
        if self._is_available is not None:
            return self._is_available

        try:
            from kokoro import KPipeline

            self._is_available = True
            logger.debug("Kokoro package is installed")
        except ImportError:
            logger.debug("Kokoro package not installed")
            self._is_available = False
            self._init_error = "Kokoro package not installed. Run: pip install kokoro"

        return self._is_available

    def _resolve_device(self) -> str:
        """Resolve the actual device to use, avoiding meta tensor issues."""
        import torch

        device = self.config.device

        if device == "cuda":
            if not torch.cuda.is_available():
                logger.info("CUDA not available, using CPU for Kokoro TTS")
                return "cpu"
            return device

        if device == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                logger.info("MPS not available, using CPU for Kokoro TTS")
                return "cpu"
            # MPS has issues with meta tensors, use CPU for now
            logger.info("MPS detected - using CPU to avoid meta tensor issues")
            return "cpu"

        if device == "rocm":
            if not torch.cuda.is_available():
                logger.info("ROCm not available, using CPU for Kokoro TTS")
                return "cpu"
            return device

        return device

    def _ensure_pipeline(self, language: str) -> None:
        """Initialize the Kokoro pipeline."""
        if self._pipeline is not None:
            return

        if not self.is_available:
            raise TTSEngineNotAvailableError(self._init_error or "Kokoro TTS is not available")

        from kokoro import KPipeline

        lang_code = get_kokoro_lang_code(language)
        device = self._resolve_device()

        hf_endpoint = os.environ.get("HF_ENDPOINT", "huggingface.co")
        logger.info(
            f"Initializing Kokoro TTS (lang={lang_code}, device={device}, hf={hf_endpoint})"
        )

        try:
            self._pipeline = KPipeline(
                lang_code=lang_code,
                device=device,
            )
            logger.info("Kokoro TTS initialized successfully")
        except Exception as e:
            error_msg = str(e).lower()

            if "timeout" in error_msg or "connection" in error_msg or "huggingface" in error_msg:
                self._init_error = (
                    "Kokoro model download failed: Cannot connect to HuggingFace.\n\n"
                    "Solutions:\n"
                    "1. Set proxy: export https_proxy=http://127.0.0.1:7897\n"
                    "2. Use mirror: export HF_ENDPOINT=https://hf-mirror.com\n"
                    "3. Or use edge-tts: --tts-engine edge-tts"
                )
                logger.error(self._init_error)
                raise TTSEngineNotAvailableError(self._init_error)

            if "meta tensor" in error_msg or "no data" in error_msg:
                logger.warning(f"Meta tensor issue on {device}, trying CPU: {e}")
                try:
                    self._pipeline = KPipeline(
                        lang_code=lang_code,
                        device="cpu",
                    )
                    logger.info("Kokoro TTS initialized on CPU (fallback from meta tensor issue)")
                    return
                except Exception as cpu_err:
                    logger.error(f"CPU fallback also failed: {cpu_err}")
                    raise TTSError(f"Kokoro initialization failed: {cpu_err}") from cpu_err

            if device != "cpu" and ("cuda" in error_msg or "mps" in error_msg):
                logger.warning(f"GPU initialization failed ({e}), falling back to CPU")
                try:
                    self._pipeline = KPipeline(
                        lang_code=lang_code,
                        device="cpu",
                    )
                    logger.info("Kokoro TTS initialized on CPU (fallback)")
                    return
                except Exception as cpu_err:
                    logger.error(f"CPU fallback also failed: {cpu_err}")
                    raise TTSError(f"Kokoro initialization failed: {cpu_err}") from cpu_err

            logger.error(f"Failed to initialize Kokoro: {e}")
            raise TTSError(f"Kokoro initialization failed: {e}") from e

    def synthesize(
        self,
        text: str,
        language: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[AudioArray, int]:
        """
        Synthesize text to speech using Kokoro.

        Args:
            text: Text to synthesize
            language: Target language code
            voice: Voice name (e.g., 'af_heart', 'zm_yunxi')
            speed: Speech speed multiplier

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if not text or not text.strip():
            return np.array([], dtype=np.float32), KOKORO_SAMPLE_RATE

        self._ensure_pipeline(language)

        if voice is None:
            voice = self.get_default_voice(language)

        logger.debug(f"Kokoro synthesizing: '{text[:50]}...' (voice={voice})")

        try:
            audio_segments = []

            generator = self._pipeline(
                text,
                voice=voice,
                speed=speed,
            )

            for gs, ps, audio in generator:
                audio_segments.append(audio)

            if not audio_segments:
                logger.warning("Kokoro produced no audio")
                return np.array([], dtype=np.float32), KOKORO_SAMPLE_RATE

            full_audio = np.concatenate(audio_segments)

            if full_audio.dtype != np.float32:
                full_audio = full_audio.astype(np.float32)

            logger.debug(f"Kokoro generated {len(full_audio)} samples")

            return full_audio, KOKORO_SAMPLE_RATE

        except Exception as e:
            logger.error(f"Kokoro synthesis error: {e}")
            raise TTSError(f"Kokoro synthesis failed: {e}") from e

    def get_available_voices(self, language: str) -> List[str]:
        """Get available Kokoro voices for a language."""
        return KOKORO_VOICE_MAP.get(language, KOKORO_VOICE_MAP["en"])

    def close(self) -> None:
        """Release resources."""
        self._pipeline = None
        logger.debug("Kokoro TTS closed")
