"""
CopyTalker: Cross-modal Data Conversion Driven Asynchronous Multi-Voice Translation System

A real-time speech-to-speech translation system supporting multiple languages and TTS engines.
"""

__version__ = "1.0.0"
__author__ = "CopyTalker Team"
__license__ = "GPL-3.0"


def __getattr__(name):
    """Lazy import for heavy dependencies."""
    if name == "AppConfig":
        from copytalker.core.config import AppConfig
        return AppConfig
    elif name == "AudioConfig":
        from copytalker.core.config import AudioConfig
        return AudioConfig
    elif name == "STTConfig":
        from copytalker.core.config import STTConfig
        return STTConfig
    elif name == "TranslationConfig":
        from copytalker.core.config import TranslationConfig
        return TranslationConfig
    elif name == "TTSConfig":
        from copytalker.core.config import TTSConfig
        return TTSConfig
    elif name == "TranslationPipeline":
        from copytalker.core.pipeline import TranslationPipeline
        return TranslationPipeline
    raise AttributeError(f"module 'copytalker' has no attribute '{name}'")


__all__ = [
    "__version__",
    "AppConfig",
    "AudioConfig",
    "STTConfig",
    "TranslationConfig",
    "TTSConfig",
    "TranslationPipeline",
]
