"""
Type definitions and protocols for CopyTalker.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Optional,
    List,
    Dict,
    Tuple,
    Protocol,
    runtime_checkable,
    Callable,
    Any,
)
import numpy as np
from numpy.typing import NDArray


# Type aliases
AudioArray = NDArray[np.float32]
SampleRate = int
LanguageCode = str


class TTSEngineType(Enum):
    """Supported TTS engines."""
    KOKORO = "kokoro"
    EDGE_TTS = "edge-tts"
    PYTTSX3 = "pyttsx3"
    INDEXTTS = "indextts"
    FISH_SPEECH = "fish-speech"
    AUTO = "auto"


class DeviceType(Enum):
    """Compute device types."""
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class WhisperModelSize(Enum):
    """Whisper model sizes."""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass
class AudioFrame:
    """Represents an audio frame with metadata."""
    data: AudioArray
    sample_rate: int
    timestamp: float
    is_speech: bool = False


@dataclass
class TranscriptionResult:
    """Result from speech recognition."""
    text: str
    language: str
    confidence: float
    duration: float = 0.0
    
    def is_empty(self) -> bool:
        """Check if transcription is empty or whitespace."""
        return not self.text or not self.text.strip()


@dataclass
class TranslationResult:
    """Result from translation."""
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    model_used: str = ""
    
    def is_same_language(self) -> bool:
        """Check if source and target languages are the same."""
        return self.source_lang == self.target_lang


@dataclass
class SynthesisResult:
    """Result from TTS synthesis."""
    audio: AudioArray
    sample_rate: int
    text: str
    voice: str
    duration: float = 0.0


@dataclass
class PipelineEvent:
    """Event emitted by the translation pipeline."""
    event_type: str
    data: Any
    timestamp: float = 0.0


# Callback type for pipeline events
PipelineCallback = Callable[[PipelineEvent], None]


@runtime_checkable
class TTSEngine(Protocol):
    """Protocol for text-to-speech engines."""
    
    @property
    def name(self) -> str:
        """Engine name."""
        ...
    
    @property
    def is_available(self) -> bool:
        """Check if engine is available."""
        ...
    
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
        ...
    
    def get_available_voices(self, language: str) -> List[str]:
        """Get available voices for a language."""
        ...


@runtime_checkable
class Translator(Protocol):
    """Protocol for translation engines."""
    
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        """
        Translate text between languages.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            TranslationResult with translated text
        """
        ...
    
    def supports_pair(self, source_lang: str, target_lang: str) -> bool:
        """Check if language pair is supported."""
        ...


@runtime_checkable
class SpeechRecognizer(Protocol):
    """Protocol for speech recognition engines."""
    
    def transcribe(
        self,
        audio: AudioArray,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio data as float32 numpy array
            sample_rate: Audio sample rate in Hz
            language: Language hint (None for auto-detect)
            
        Returns:
            TranscriptionResult with transcribed text
        """
        ...
    
    def detect_language(
        self,
        audio: AudioArray,
        sample_rate: int,
    ) -> Tuple[str, float]:
        """
        Detect language of audio.
        
        Returns:
            Tuple of (language_code, confidence)
        """
        ...
