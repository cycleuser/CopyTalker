"""Configuration management for CopyTalker."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from copytalker.core.constants import (
    DEFAULT_SAMPLE_RATE, DEFAULT_FRAME_DURATION_MS,
    DEFAULT_VAD_AGGRESSIVENESS, DEFAULT_SILENCE_THRESHOLD_S,
    DEFAULT_AUDIO_BUFFER_SIZE, AUTO_DETECT_CODE,
)

logger = logging.getLogger(__name__)


def get_default_cache_dir() -> Path:
    env = os.environ.get("COPYTALKER_CACHE_DIR")
    return Path(env) if env else Path.home() / ".cache" / "copytalker"


def get_default_config_path() -> Path:
    env = os.environ.get("COPYTALKER_CONFIG")
    return Path(env) if env else Path.home() / ".config" / "copytalker" / "config.yaml"


def get_device() -> str:
    """Detect best device: CUDA, MPS, ROCm, or CPU."""
    env = os.environ.get("COPYTALKER_DEVICE", "auto").lower()
    if env == "cpu": return "cpu"
    try:
        import torch
        cuda = torch.cuda.is_available()
        mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        rocm = False
        try:
            if hasattr(torch.version, "hip") and torch.version.hip: rocm = True
            elif os.environ.get("ROCM_VISIBLE_DEVICES"): rocm = True
        except: pass
    except ImportError: cuda = mps = rocm = False
    if env == "cuda" and cuda: return "cuda"
    if env == "mps" and mps: return "mps"
    if env == "rocm" and rocm: return "rocm"
    if env == "auto":
        if rocm: return "rocm"
        if cuda: return "cuda"
        if mps: return "mps"
        return "cpu"
    return "cpu"


def list_available_devices() -> list[dict[str, str | bool]]:
    """List available compute devices."""
    devs = [{"type": "cpu", "available": True, "info": "CPU"}]
    try:
        import torch
        if torch.cuda.is_available():
            devs.append({"type": "cuda", "available": True, "info": f"CUDA: {torch.cuda.get_device_name(0)}"})
        else:
            devs.append({"type": "cuda", "available": False, "info": "No NVIDIA GPU"})
        if hasattr(torch.backends, "mps"):
            mps_ok = torch.backends.mps.is_available()
            devs.append({"type": "mps", "available": mps_ok, "info": "MPS (Apple Silicon)" if mps_ok else "MPS N/A"})
        rocm_ok, rocm_msg = False, "ROCm not available"
        try:
            if hasattr(torch.version, "hip") and torch.version.hip: rocm_ok, rocm_msg = True, f"ROCm: {torch.version.hip}"
            elif os.environ.get("ROCM_VISIBLE_DEVICES"): rocm_ok, rocm_msg = True, "ROCm via env"
        except Exception as e: rocm_msg = f"ROCm error: {e}"
        devs.append({"type": "rocm", "available": rocm_ok, "info": rocm_msg})
    except ImportError:
        for t in ["cuda", "mps", "rocm"]: devs.append({"type": t, "available": False, "info": "PyTorch not installed"})
    return devs

@dataclass
class AudioConfig:
    """Audio configuration."""
    sample_rate: int = DEFAULT_SAMPLE_RATE
    frame_duration_ms: int = DEFAULT_FRAME_DURATION_MS
    vad_aggressiveness: int = DEFAULT_VAD_AGGRESSIVENESS
    silence_threshold_s: float = DEFAULT_SILENCE_THRESHOLD_S
    buffer_size: int = DEFAULT_AUDIO_BUFFER_SIZE
    channels: int = 1
    min_energy_threshold: float = 0.01
    min_speech_duration_s: float = 0.5
    calibrated_noise_level: float = 0.0
    @property
    def frame_size(self) -> int:
        return int(self.sample_rate * self.frame_duration_ms / 1000)
    def validate(self) -> None:
        if self.sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(f"Unsupported sample rate: {self.sample_rate}")
        if self.vad_aggressiveness not in [0, 1, 2, 3]:
            raise ValueError(f"VAD must be 0-3: {self.vad_aggressiveness}")
        if self.frame_duration_ms not in [10, 20, 30]:
            raise ValueError(f"Frame duration must be 10/20/30 ms")


@dataclass
class STTConfig:
    """STT configuration."""
    model_size: str = "small"
    device: str = field(default_factory=get_device)
    compute_type: str = ""
    language: str = AUTO_DETECT_CODE
    beam_size: int = 5
    condition_on_previous_text: bool = False
    min_confidence: float = 0.5
    min_words: int = 2
    def __post_init__(self):
        if not self.compute_type:
            if self.device in ["cuda", "rocm"]:
                self.compute_type = "float16"
            elif self.device == "mps":
                # MPS: ctranslate2 does not support MPS, use CPU with float32
                self.device = "cpu"
                self.compute_type = "float32"
            else:
                self.compute_type = "float32"
    def validate(self) -> None:
        valid = ["tiny", "base", "small", "medium", "large"]
        if self.model_size not in valid:
            raise ValueError(f"Invalid model: {self.model_size}")


@dataclass
class TranslationConfig:
    """Translation configuration."""
    source_lang: str = AUTO_DETECT_CODE
    target_lang: str = "en"
    model_name: Optional[str] = None
    max_length: int = 400
    device: str = field(default_factory=get_device)
    def validate(self) -> None:
        if not (1 <= self.max_length <= 1024):
            raise ValueError(f"max_length must be 1-1024")


@dataclass
class TTSConfig:
    """TTS configuration."""
    engine: str = "auto"
    voice: Optional[str] = None
    language: str = "en"
    speed: float = 1.0
    device: str = field(default_factory=get_device)
    kokoro_model_path: Optional[str] = None
    indextts_model_path: Optional[str] = None
    indextts_reference_audio: Optional[str] = None
    indextts_emotion: Optional[str] = None
    indextts_emotion_audio: Optional[str] = None
    indextts_target_duration: Optional[float] = None
    fish_speech_model_path: Optional[str] = None
    fish_speech_api_key: Optional[str] = None
    fish_speech_api_url: str = "https://api.fish.audio"
    fish_speech_voice_id: Optional[str] = None
    fish_speech_reference_audio: Optional[str] = None
    fish_speech_emotion: Optional[str] = None
    def validate(self) -> None:
        valid = ["kokoro", "edge-tts", "pyttsx3", "indextts", "fish-speech", "auto"]
        if self.engine not in valid:
            raise ValueError(f"Invalid TTS engine: {self.engine}")
        if not (0.5 <= self.speed <= 2.0):
            raise ValueError(f"Speed must be 0.5-2.0")

@dataclass
class CacheConfig:
    """Cache configuration."""
    cache_dir: Path = field(default_factory=get_default_cache_dir)
    auto_download: bool = True
    offline_mode: bool = False
    def __post_init__(self):
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
    def ensure_cache_dir(self) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir
    @property
    def whisper_cache_dir(self) -> Path:
        return self.cache_dir / "whisper"
    @property
    def translation_cache_dir(self) -> Path:
        return self.cache_dir / "translation"
    @property
    def tts_cache_dir(self) -> Path:
        return self.cache_dir / "tts"


@dataclass
class AppConfig:
    """Master configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    debug: bool = False
    log_level: str = "INFO"
    def validate(self) -> None:
        self.audio.validate()
        self.stt.validate()
        self.translation.validate()
        self.tts.validate()
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
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
        if not path.exists():
            logger.warning(f"Config not found: {path}")
            return cls()
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except ImportError:
            logger.warning("PyYAML not installed")
            return cls()
        return cls.from_dict(data)
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "AppConfig":
        if config_path is None:
            config_path = get_default_config_path()
        if config_path.exists():
            return cls.from_yaml(config_path)
        return cls()
    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio": {"sample_rate": self.audio.sample_rate, "frame_duration_ms": self.audio.frame_duration_ms},
            "stt": {"model_size": self.stt.model_size, "device": self.stt.device, "compute_type": self.stt.compute_type},
            "translation": {"source_lang": self.translation.source_lang, "target_lang": self.translation.target_lang, "device": self.translation.device},
            "tts": {"engine": self.tts.engine, "language": self.tts.language, "device": self.tts.device},
            "cache": {"cache_dir": str(self.cache.cache_dir)},
            "debug": self.debug,
            "log_level": self.log_level,
        }
    def save(self, path: Optional[Path] = None) -> None:
        try:
            import yaml
        except ImportError:
            logger.error("PyYAML not installed")
            return
        if path is None:
            path = get_default_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        logger.info(f"Configuration saved to: {path}")


def setup_logging(config: Optional[AppConfig] = None) -> None:
    """Configure logging."""
    if config is None:
        config = AppConfig()
    level = logging.DEBUG if config.debug else getattr(logging, config.log_level, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")