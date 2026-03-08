"""
Configuration management for CopyTalker.

Supports environment variables, config files, and programmatic configuration.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from copytalker.core.constants import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_FRAME_DURATION_MS,
    DEFAULT_VAD_AGGRESSIVENESS,
    DEFAULT_SILENCE_THRESHOLD_S,
    DEFAULT_AUDIO_BUFFER_SIZE,
    AUTO_DETECT_CODE,
)

logger = logging.getLogger(__name__)


def get_default_cache_dir() -> Path:
    """Get default cache directory for models."""
    env_cache = os.environ.get("COPYTALKER_CACHE_DIR")
    if env_cache:
        return Path(env_cache)
    return Path.home() / ".cache" / "copytalker"


def get_default_config_path() -> Path:
    """Get default config file path."""
    env_config = os.environ.get("COPYTALKER_CONFIG")
    if env_config:
        return Path(env_config)
    return Path.home() / ".config" / "copytalker" / "config.yaml"


def get_device() -> str:
    """Detect the best available device."""
    env_device = os.environ.get("COPYTALKER_DEVICE", "auto")
    
    if env_device.lower() == "cpu":
        return "cpu"
    
    # Lazy import torch to avoid import errors when not installed
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False
    
    if env_device.lower() == "cuda" and cuda_available:
        return "cuda"
    elif env_device.lower() == "auto":
        return "cuda" if cuda_available else "cpu"
    return "cpu"


@dataclass
class AudioConfig:
    """Audio capture and playback configuration."""
    
    sample_rate: int = DEFAULT_SAMPLE_RATE
    frame_duration_ms: int = DEFAULT_FRAME_DURATION_MS
    vad_aggressiveness: int = DEFAULT_VAD_AGGRESSIVENESS
    silence_threshold_s: float = DEFAULT_SILENCE_THRESHOLD_S
    buffer_size: int = DEFAULT_AUDIO_BUFFER_SIZE
    channels: int = 1
    
    # Noise filtering parameters
    min_energy_threshold: float = 0.01  # Minimum RMS energy (0.0-1.0)
    min_speech_duration_s: float = 0.5  # Minimum speech duration in seconds
    calibrated_noise_level: float = 0.0  # Calibrated noise floor (set during calibration)
    
    @property
    def frame_size(self) -> int:
        """Calculate frame size in samples."""
        return int(self.sample_rate * self.frame_duration_ms / 1000)
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(f"Unsupported sample rate: {self.sample_rate}")
        if self.vad_aggressiveness not in [0, 1, 2, 3]:
            raise ValueError(f"VAD aggressiveness must be 0-3, got: {self.vad_aggressiveness}")
        if self.frame_duration_ms not in [10, 20, 30]:
            raise ValueError(f"Frame duration must be 10, 20, or 30 ms, got: {self.frame_duration_ms}")


@dataclass
class STTConfig:
    """Speech-to-text configuration."""
    
    model_size: str = "small"
    device: str = field(default_factory=get_device)
    compute_type: str = ""  # Auto-detect based on device
    language: str = AUTO_DETECT_CODE
    beam_size: int = 5
    condition_on_previous_text: bool = False
    
    # Filtering parameters
    min_confidence: float = 0.5  # Minimum confidence to accept transcription
    min_words: int = 2  # Minimum word count to accept transcription
    
    def __post_init__(self):
        if not self.compute_type:
            self.compute_type = "float16" if self.device == "cuda" else "float32"
    
    def validate(self) -> None:
        """Validate configuration values."""
        valid_sizes = ["tiny", "base", "small", "medium", "large"]
        if self.model_size not in valid_sizes:
            raise ValueError(f"Invalid model size: {self.model_size}. Must be one of {valid_sizes}")


@dataclass
class TranslationConfig:
    """Translation configuration."""
    
    source_lang: str = AUTO_DETECT_CODE
    target_lang: str = "en"
    model_name: Optional[str] = None  # Auto-select based on language pair
    max_length: int = 400
    device: str = field(default_factory=get_device)
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.max_length < 1 or self.max_length > 1024:
            raise ValueError(f"max_length must be 1-1024, got: {self.max_length}")


@dataclass
class TTSConfig:
    """Text-to-speech configuration."""
    
    engine: str = "kokoro"
    voice: Optional[str] = None  # Auto-select based on language
    language: str = "en"
    speed: float = 1.0
    device: str = field(default_factory=get_device)
    kokoro_model_path: Optional[str] = None  # Custom path for Kokoro model
    
    # IndexTTS settings
    indextts_model_path: Optional[str] = None  # Custom path for IndexTTS model
    indextts_reference_audio: Optional[str] = None  # Default reference audio for voice cloning
    indextts_emotion: Optional[str] = None  # Emotion name (happy, sad, angry, etc.)
    indextts_emotion_audio: Optional[str] = None  # Path to emotion reference audio
    indextts_target_duration: Optional[float] = None  # Target duration in seconds
    
    # Fish-Speech settings
    fish_speech_model_path: Optional[str] = None  # Custom path for Fish-Speech model
    fish_speech_api_key: Optional[str] = None  # Fish Audio cloud API key
    fish_speech_api_url: str = "https://api.fish.audio"  # Fish Audio API base URL
    fish_speech_voice_id: Optional[str] = None  # Registered voice/speaker ID
    fish_speech_reference_audio: Optional[str] = None  # Default reference audio
    fish_speech_emotion: Optional[str] = None  # Emotion tag (happy, whisper, etc.)
    
    def validate(self) -> None:
        """Validate configuration values."""
        valid_engines = [
            "kokoro", "edge-tts", "pyttsx3", "indextts", "fish-speech", "auto",
        ]
        if self.engine not in valid_engines:
            raise ValueError(f"Invalid TTS engine: {self.engine}. Must be one of {valid_engines}")
        if self.speed < 0.5 or self.speed > 2.0:
            raise ValueError(f"Speed must be 0.5-2.0, got: {self.speed}")


@dataclass
class CacheConfig:
    """Model caching configuration."""
    
    cache_dir: Path = field(default_factory=get_default_cache_dir)
    auto_download: bool = True
    offline_mode: bool = False
    
    def __post_init__(self):
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
    
    def ensure_cache_dir(self) -> Path:
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir
    
    @property
    def whisper_cache_dir(self) -> Path:
        """Get Whisper model cache directory."""
        return self.cache_dir / "whisper"
    
    @property
    def translation_cache_dir(self) -> Path:
        """Get translation model cache directory."""
        return self.cache_dir / "translation"
    
    @property
    def tts_cache_dir(self) -> Path:
        """Get TTS model cache directory."""
        return self.cache_dir / "tts"


@dataclass
class AppConfig:
    """Master application configuration."""
    
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    debug: bool = False
    log_level: str = "INFO"
    
    def validate(self) -> None:
        """Validate all configuration values."""
        self.audio.validate()
        self.stt.validate()
        self.translation.validate()
        self.tts.validate()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        """Create configuration from dictionary."""
        return cls(
            audio=AudioConfig(**data.get("audio", {})),
            stt=STTConfig(**data.get("stt", {})),
            translation=TranslationConfig(**data.get("translation", {})),
            tts=TTSConfig(**data.get("tts", {})),
            cache=CacheConfig(**data.get("cache", {})),
            debug=data.get("debug", False),
            log_level=data.get("log_level", "INFO"),
        )
    
    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        """Load configuration from YAML file."""
        if not path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return cls()
        
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except ImportError:
            logger.warning("PyYAML not installed, using defaults")
            return cls()
        
        return cls.from_dict(data)
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "AppConfig":
        """Load configuration from file or defaults."""
        if config_path is None:
            config_path = get_default_config_path()
        
        if config_path.exists():
            return cls.from_yaml(config_path)
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "audio": {
                "sample_rate": self.audio.sample_rate,
                "frame_duration_ms": self.audio.frame_duration_ms,
                "vad_aggressiveness": self.audio.vad_aggressiveness,
                "silence_threshold_s": self.audio.silence_threshold_s,
                "buffer_size": self.audio.buffer_size,
                "channels": self.audio.channels,
            },
            "stt": {
                "model_size": self.stt.model_size,
                "device": self.stt.device,
                "compute_type": self.stt.compute_type,
                "language": self.stt.language,
                "beam_size": self.stt.beam_size,
            },
            "translation": {
                "source_lang": self.translation.source_lang,
                "target_lang": self.translation.target_lang,
                "model_name": self.translation.model_name,
                "max_length": self.translation.max_length,
                "device": self.translation.device,
            },
            "tts": {
                "engine": self.tts.engine,
                "voice": self.tts.voice,
                "language": self.tts.language,
                "speed": self.tts.speed,
                "device": self.tts.device,
            },
            "cache": {
                "cache_dir": str(self.cache.cache_dir),
                "auto_download": self.cache.auto_download,
                "offline_mode": self.cache.offline_mode,
            },
            "debug": self.debug,
            "log_level": self.log_level,
        }
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to YAML file."""
        try:
            import yaml
        except ImportError:
            logger.error("PyYAML not installed, cannot save config")
            return
        
        if path is None:
            path = get_default_config_path()
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        
        logger.info(f"Configuration saved to: {path}")


def setup_logging(config: Optional[AppConfig] = None) -> None:
    """Configure logging based on configuration."""
    if config is None:
        config = AppConfig()
    
    level = logging.DEBUG if config.debug else getattr(logging, config.log_level, logging.INFO)
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
