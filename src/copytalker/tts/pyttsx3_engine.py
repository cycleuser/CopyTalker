"""
Pyttsx3 TTS engine - Offline fallback.
"""

import logging
import tempfile
import wave
from typing import List, Optional, Tuple

import numpy as np

from copytalker.core.config import TTSConfig
from copytalker.core.exceptions import TTSError, TTSEngineNotAvailableError
from copytalker.core.types import AudioArray
from copytalker.tts.base import TTSEngineBase

logger = logging.getLogger(__name__)


class Pyttsx3TTS(TTSEngineBase):
    """
    Pyttsx3 TTS engine - Offline system TTS.
    
    Uses system TTS engines (SAPI on Windows, espeak on Linux, NSSpeechSynthesizer on Mac).
    Lower quality but works offline without downloads.
    """
    
    DEFAULT_SAMPLE_RATE = 22050
    
    def __init__(self, config: Optional[TTSConfig] = None):
        """
        Initialize Pyttsx3 TTS engine.
        
        Args:
            config: TTS configuration
        """
        super().__init__(config)
        self._engine = None
        self._is_available = None
        self._voices = {}
    
    @property
    def name(self) -> str:
        return "pyttsx3"
    
    @property
    def is_available(self) -> bool:
        """Check if pyttsx3 is installed and working."""
        if self._is_available is not None:
            return self._is_available
        
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.stop()
            self._is_available = True
        except Exception as e:
            logger.debug(f"pyttsx3 not available: {e}")
            self._is_available = False
        
        return self._is_available
    
    def _ensure_engine(self) -> None:
        """Initialize pyttsx3 engine."""
        if self._engine is not None:
            return
        
        if not self.is_available:
            raise TTSEngineNotAvailableError("pyttsx3 is not available")
        
        import pyttsx3
        self._engine = pyttsx3.init()
        
        # Cache available voices
        for voice in self._engine.getProperty('voices'):
            # Try to determine language from voice
            lang = self._guess_voice_language(voice)
            if lang not in self._voices:
                self._voices[lang] = []
            self._voices[lang].append(voice.id)
    
    def _guess_voice_language(self, voice) -> str:
        """Guess language from voice properties."""
        name_lower = voice.name.lower()
        voice_id_lower = voice.id.lower()
        
        if any(x in name_lower or x in voice_id_lower for x in ['chinese', 'zh', 'mandarin']):
            return 'zh'
        elif any(x in name_lower or x in voice_id_lower for x in ['japanese', 'ja']):
            return 'ja'
        elif any(x in name_lower or x in voice_id_lower for x in ['korean', 'ko']):
            return 'ko'
        elif any(x in name_lower or x in voice_id_lower for x in ['spanish', 'es']):
            return 'es'
        elif any(x in name_lower or x in voice_id_lower for x in ['french', 'fr']):
            return 'fr'
        elif any(x in name_lower or x in voice_id_lower for x in ['german', 'de']):
            return 'de'
        elif any(x in name_lower or x in voice_id_lower for x in ['russian', 'ru']):
            return 'ru'
        elif any(x in name_lower or x in voice_id_lower for x in ['arabic', 'ar']):
            return 'ar'
        else:
            return 'en'
    
    def synthesize(
        self,
        text: str,
        language: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[AudioArray, int]:
        """
        Synthesize text to speech using pyttsx3.
        
        Args:
            text: Text to synthesize
            language: Target language code
            voice: Voice ID
            speed: Speech speed multiplier
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if not text or not text.strip():
            return np.array([], dtype=np.float32), self.DEFAULT_SAMPLE_RATE
        
        self._ensure_engine()
        
        logger.debug(f"Pyttsx3 synthesizing: '{text[:50]}...'")
        
        try:
            # Set voice if available
            voices = self.get_available_voices(language)
            if voice and voice in voices:
                self._engine.setProperty('voice', voice)
            elif voices:
                self._engine.setProperty('voice', voices[0])
            
            # Set rate (words per minute, default ~200)
            base_rate = 150
            self._engine.setProperty('rate', int(base_rate * speed))
            
            # Synthesize to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
            
            self._engine.save_to_file(text, tmp_path)
            self._engine.runAndWait()
            
            # Read WAV file
            audio_array = self._read_wav(tmp_path)
            
            # Clean up
            import os
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            logger.debug(f"Pyttsx3 generated {len(audio_array)} samples")
            
            return audio_array, self.DEFAULT_SAMPLE_RATE
            
        except Exception as e:
            logger.error(f"Pyttsx3 synthesis error: {e}")
            raise TTSError(f"Pyttsx3 synthesis failed: {e}") from e
    
    def _read_wav(self, wav_path: str) -> AudioArray:
        """Read WAV file to numpy array."""
        with wave.open(wav_path, 'rb') as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            
            raw_data = wav_file.readframes(n_frames)
        
        # Convert to numpy
        if sample_width == 1:
            dtype = np.uint8
            max_val = 128.0
        elif sample_width == 2:
            dtype = np.int16
            max_val = 32768.0
        else:
            dtype = np.int32
            max_val = 2147483648.0
        
        audio = np.frombuffer(raw_data, dtype=dtype)
        
        # Convert to mono if stereo
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)
        
        # Normalize to float32 [-1, 1]
        audio = audio.astype(np.float32) / max_val
        
        return audio
    
    def get_available_voices(self, language: str) -> List[str]:
        """Get available pyttsx3 voices for a language."""
        self._ensure_engine()
        return self._voices.get(language, self._voices.get('en', []))
    
    def close(self) -> None:
        """Release resources."""
        if self._engine is not None:
            try:
                self._engine.stop()
            except:
                pass
            self._engine = None
        logger.debug("Pyttsx3 TTS closed")
