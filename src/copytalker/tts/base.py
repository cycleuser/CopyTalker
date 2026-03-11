"""
Base TTS engine interface and factory.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Type

import numpy as np

from copytalker.core.config import TTSConfig
from copytalker.core.types import AudioArray

logger = logging.getLogger(__name__)


class TTSEngineBase(ABC):
    """Abstract base class for TTS engines."""
    
    def __init__(self, config: Optional[TTSConfig] = None):
        """
        Initialize TTS engine.
        
        Args:
            config: TTS configuration
        """
        self.config = config or TTSConfig()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name."""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if engine is available."""
        pass
    
    @abstractmethod
    def synthesize(
        self,
        text: str,
        language: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[AudioArray, int]:
        """
        Synthesize text to speech.
        
        Args:
            text: Text to synthesize
            language: Target language code
            voice: Voice name (optional)
            speed: Speech speed multiplier
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        pass
    
    @abstractmethod
    def get_available_voices(self, language: str) -> List[str]:
        """Get available voices for a language."""
        pass
    
    def get_default_voice(self, language: str) -> Optional[str]:
        """Get default voice for a language."""
        voices = self.get_available_voices(language)
        return voices[0] if voices else None


# Registry of available TTS engines
_TTS_ENGINE_REGISTRY: dict = {}


def register_tts_engine(name: str, engine_class: Type[TTSEngineBase]) -> None:
    """Register a TTS engine."""
    _TTS_ENGINE_REGISTRY[name] = engine_class


def get_tts_engine(
    engine_name: str = "auto",
    config: Optional[TTSConfig] = None,
) -> TTSEngineBase:
    """
    Get a TTS engine by name.
    
    Args:
        engine_name: Engine name ('kokoro', 'edge-tts', 'pyttsx3',
                     'indextts', 'fish-speech', 'auto')
        config: TTS configuration
        
    Returns:
        TTSEngineBase instance
    """
    # Lazy import to avoid circular dependencies
    from copytalker.tts.kokoro import KokoroTTS
    from copytalker.tts.edge import EdgeTTS
    from copytalker.tts.pyttsx3_engine import Pyttsx3TTS
    from copytalker.tts.indextts import IndexTTS
    from copytalker.tts.fish_speech import FishSpeechTTS
    
    engines = {
        "kokoro": KokoroTTS,
        "edge-tts": EdgeTTS,
        "pyttsx3": Pyttsx3TTS,
        "indextts": IndexTTS,
        "fish-speech": FishSpeechTTS,
    }
    
    if engine_name == "auto":
        # Try engines in order of preference:
        # edge-tts and pyttsx3 first (included in base install, work out of the box)
        # then kokoro and others (require extra install / model downloads)
        for name in ["edge-tts", "pyttsx3", "kokoro", "indextts", "fish-speech"]:
            try:
                engine = engines[name](config)
                if engine.is_available:
                    logger.info(f"Auto-selected TTS engine: {name}")
                    return engine
            except Exception as e:
                logger.debug(f"Engine {name} not available: {e}")
                continue
        
        raise RuntimeError("No TTS engine available")
    
    if engine_name not in engines:
        raise ValueError(f"Unknown TTS engine: {engine_name}")
    
    return engines[engine_name](config)
