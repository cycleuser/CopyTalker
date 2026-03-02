"""
Speech recognition using Faster-Whisper.
"""

import logging
import time
from typing import Optional, Tuple

import numpy as np

from copytalker.core.config import STTConfig
from copytalker.core.constants import normalize_language_code, AUTO_DETECT_CODE
from copytalker.core.exceptions import ModelError
from copytalker.core.types import AudioArray, TranscriptionResult

logger = logging.getLogger(__name__)


class WhisperRecognizer:
    """
    Speech recognition using Faster-Whisper model.
    
    Provides transcription and language detection capabilities.
    """
    
    def __init__(self, config: Optional[STTConfig] = None):
        """
        Initialize Whisper recognizer.
        
        Args:
            config: STT configuration (uses defaults if None)
        """
        self.config = config or STTConfig()
        self.config.validate()
        
        self._model = None
        self._model_loaded = False
        
    def _ensure_model(self) -> None:
        """Lazy-load the Whisper model."""
        if self._model_loaded:
            return
        
        logger.info(f"Loading Whisper model: {self.config.model_size} "
                   f"on {self.config.device} ({self.config.compute_type})")
        
        try:
            from faster_whisper import WhisperModel
            
            self._model = WhisperModel(
                self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
            self._model_loaded = True
            logger.info("Whisper model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise ModelError(f"Failed to load Whisper model: {e}") from e
    
    def transcribe(
        self,
        audio: AudioArray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio data as float32 numpy array
            sample_rate: Audio sample rate in Hz (should be 16000)
            language: Language hint (None for auto-detect)
            
        Returns:
            TranscriptionResult with transcribed text and metadata
        """
        self._ensure_model()
        
        start_time = time.time()
        
        # Use language from config if not specified
        whisper_lang = language
        if whisper_lang is None or whisper_lang == AUTO_DETECT_CODE:
            if self.config.language != AUTO_DETECT_CODE:
                whisper_lang = self.config.language
            else:
                whisper_lang = None  # Auto-detect
        
        try:
            segments, info = self._model.transcribe(
                audio,
                beam_size=self.config.beam_size,
                language=whisper_lang,
                condition_on_previous_text=self.config.condition_on_previous_text,
            )
            
            # Collect all segment text
            text = "".join(segment.text for segment in segments).strip()
            
            # Get detected language
            detected_lang = info.language if info else "en"
            confidence = info.language_probability if info else 0.0
            
            # Normalize language code
            normalized_lang = normalize_language_code(detected_lang)
            
            duration = time.time() - start_time
            
            # Filter by confidence threshold
            if confidence < self.config.min_confidence:
                logger.debug(f"Filtered (low confidence): '{text}' (conf={confidence:.2f} < {self.config.min_confidence})")
                return TranscriptionResult(
                    text="",
                    language=normalized_lang,
                    confidence=confidence,
                    duration=duration,
                )
            
            # Filter by minimum word count
            word_count = len(text.split()) if text else 0
            if word_count < self.config.min_words:
                logger.debug(f"Filtered (too short): '{text}' ({word_count} words < {self.config.min_words})")
                return TranscriptionResult(
                    text="",
                    language=normalized_lang,
                    confidence=confidence,
                    duration=duration,
                )
            
            result = TranscriptionResult(
                text=text,
                language=normalized_lang,
                confidence=confidence,
                duration=duration,
            )
            
            if text:
                logger.info(f"Transcribed: '{text}' (lang={normalized_lang}, "
                           f"conf={confidence:.2f}, time={duration:.2f}s)")
            else:
                logger.debug(f"No speech detected (time={duration:.2f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise ModelError(f"Transcription failed: {e}") from e
    
    def detect_language(
        self,
        audio: AudioArray,
        sample_rate: int = 16000,
    ) -> Tuple[str, float]:
        """
        Detect the language of audio.
        
        Args:
            audio: Audio data as float32 numpy array
            sample_rate: Audio sample rate in Hz
            
        Returns:
            Tuple of (language_code, confidence)
        """
        self._ensure_model()
        
        try:
            # Use Whisper's language detection
            _, info = self._model.transcribe(
                audio,
                beam_size=1,
                language=None,
                condition_on_previous_text=False,
            )
            
            detected_lang = info.language if info else "en"
            confidence = info.language_probability if info else 0.0
            
            normalized_lang = normalize_language_code(detected_lang)
            
            logger.debug(f"Detected language: {normalized_lang} (conf={confidence:.2f})")
            
            return normalized_lang, confidence
            
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return "en", 0.0
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded
    
    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_loaded = False
            logger.info("Whisper model unloaded")
    
    def __enter__(self) -> "WhisperRecognizer":
        """Context manager entry - load model."""
        self._ensure_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - optionally unload model."""
        # Keep model loaded by default for reuse
        pass
