"""
Core module containing configuration, types, and pipeline orchestration.
"""

from copytalker.core.config import (
    AppConfig,
    AudioConfig,
    STTConfig,
    TranslationConfig,
    TTSConfig,
)
from copytalker.core.exceptions import (
    CopyTalkerError,
    AudioError,
    ModelError,
    TranslationError,
    TTSError,
)

__all__ = [
    "AppConfig",
    "AudioConfig",
    "STTConfig",
    "TranslationConfig",
    "TTSConfig",
    "CopyTalkerError",
    "AudioError",
    "ModelError",
    "TranslationError",
    "TTSError",
]
