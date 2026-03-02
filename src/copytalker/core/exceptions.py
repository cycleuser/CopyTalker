"""
Custom exceptions for CopyTalker.
"""


class CopyTalkerError(Exception):
    """Base exception for all CopyTalker errors."""
    pass


class AudioError(CopyTalkerError):
    """Exception raised for audio-related errors."""
    pass


class ModelError(CopyTalkerError):
    """Exception raised when model loading or inference fails."""
    pass


class ModelNotFoundError(ModelError):
    """Exception raised when a required model is not found."""
    pass


class ModelDownloadError(ModelError):
    """Exception raised when model download fails."""
    pass


class TranslationError(CopyTalkerError):
    """Exception raised for translation-related errors."""
    pass


class UnsupportedLanguageError(TranslationError):
    """Exception raised when a language pair is not supported."""
    pass


class TTSError(CopyTalkerError):
    """Exception raised for text-to-speech errors."""
    pass


class TTSEngineNotAvailableError(TTSError):
    """Exception raised when TTS engine is not available."""
    pass


class ConfigurationError(CopyTalkerError):
    """Exception raised for configuration-related errors."""
    pass


class PipelineError(CopyTalkerError):
    """Exception raised for pipeline orchestration errors."""
    pass
