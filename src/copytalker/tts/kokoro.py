"""
Kokoro TTS engine - Primary high-quality TTS.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from copytalker.core.config import TTSConfig, get_default_cache_dir
from copytalker.core.constants import (
    KOKORO_LANG_MAP,
    KOKORO_VOICE_MAP,
    KOKORO_SAMPLE_RATE,
    get_kokoro_lang_code,
)
from copytalker.core.exceptions import TTSError, TTSEngineNotAvailableError
from copytalker.core.types import AudioArray
from copytalker.tts.base import TTSEngineBase

logger = logging.getLogger(__name__)


class KokoroTTS(TTSEngineBase):
    """
    Kokoro TTS engine - High quality neural TTS.
    
    Supports English, Chinese, and Japanese with multiple voices.
    """
    
    # HuggingFace model ID for auto-download
    HF_MODEL_ID = "hexgrad/Kokoro-82M"
    
    def __init__(self, config: Optional[TTSConfig] = None):
        """
        Initialize Kokoro TTS engine.
        
        Args:
            config: TTS configuration
        """
        super().__init__(config)
        
        self._pipeline = None
        self._is_available = None
        self._model_path = None
        
        # Check for custom model path
        if self.config.kokoro_model_path:
            self._model_path = Path(self.config.kokoro_model_path)
        else:
            # Check common locations
            common_paths = [
                Path.home() / "Documents" / "GitHub" / "Kokoro-82M",
                get_default_cache_dir() / "kokoro",
                Path.home() / ".cache" / "huggingface" / "hub" / "models--hexgrad--Kokoro-82M",
            ]
            for path in common_paths:
                if path.exists():
                    self._model_path = path
                    break
    
    @property
    def name(self) -> str:
        return "kokoro"
    
    @property
    def is_available(self) -> bool:
        """Check if Kokoro is installed and model is available."""
        if self._is_available is not None:
            return self._is_available
        
        try:
            from kokoro import KPipeline
            self._is_available = True
        except ImportError:
            logger.debug("Kokoro package not installed")
            self._is_available = False
            return False
        
        # Check if model exists
        if self._model_path is None or not self._model_path.exists():
            logger.debug("Kokoro model not found")
            self._is_available = False
            return False
        
        return self._is_available
    
    def _find_model_file(self) -> Optional[Path]:
        """Find the Kokoro model file."""
        if self._model_path is None:
            return None
        
        model_patterns = [
            "kokoro-v1_0.pth",
            "kokoro-v0_19.pt",
            "kokoro-v0.19.pt",
            "kokoro-82M.pt",
            "model.pt",
            "model.pth",
        ]
        
        for pattern in model_patterns:
            model_file = self._model_path / pattern
            if model_file.exists():
                return model_file
        
        # Search for any .pt or .pth file
        for ext in [".pt", ".pth"]:
            files = list(self._model_path.glob(f"*{ext}"))
            if files:
                return files[0]
        
        return None
    
    def _ensure_pipeline(self, language: str) -> None:
        """Initialize the Kokoro pipeline."""
        if self._pipeline is not None:
            return
        
        if not self.is_available:
            raise TTSEngineNotAvailableError("Kokoro TTS is not available")
        
        from kokoro import KPipeline
        
        model_file = self._find_model_file()
        if not model_file:
            raise TTSError(f"No Kokoro model file found in {self._model_path}")
        
        lang_code = get_kokoro_lang_code(language)
        device = self.config.device
        
        # Force CPU if CUDA is not available
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            logger.info("CUDA not available, using CPU for Kokoro TTS")
        
        logger.info(f"Initializing Kokoro TTS (lang={lang_code}, device={device})")
        
        try:
            self._pipeline = KPipeline(
                lang_code=lang_code,
                model=str(model_file),
                device=device,
            )
            logger.info("Kokoro TTS initialized successfully")
        except Exception as e:
            error_msg = str(e)
            # Fallback to CPU if CUDA fails
            if device == "cuda" and ("CUDA" in error_msg or "cuDNN" in error_msg or "CUDNN" in error_msg):
                logger.warning(f"CUDA initialization failed ({error_msg}), falling back to CPU")
                try:
                    self._pipeline = KPipeline(
                        lang_code=lang_code,
                        model=str(model_file),
                        device="cpu",
                    )
                    logger.info("Kokoro TTS initialized on CPU (fallback)")
                    return
                except Exception as cpu_err:
                    logger.error(f"CPU fallback also failed: {cpu_err}")
                    raise TTSError(f"Kokoro initialization failed on both CUDA and CPU: {cpu_err}") from cpu_err
            
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
        
        # Select voice if not specified
        if voice is None:
            voice = self.get_default_voice(language)
        
        logger.debug(f"Kokoro synthesizing: '{text[:50]}...' (voice={voice})")
        
        try:
            # Generate audio
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
            
            # Concatenate all segments
            full_audio = np.concatenate(audio_segments)
            
            # Ensure float32
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
