"""
CopyTalker - Unified Python API.

Provides ToolResult-based wrappers for programmatic usage
and agent integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolResult:
    """Standardised return type for all CopyTalker API functions."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }


def translate(
    *,
    target: str,
    source: str = "auto",
    voice: str | None = None,
    tts_engine: str = "auto",
    whisper_model: str = "small",
    device: str = "auto",
    duration: float | None = None,
) -> ToolResult:
    """Start a real-time speech-to-speech translation session.

    Parameters
    ----------
    target : str
        Target language code (e.g. 'en', 'zh', 'ja').
    source : str
        Source language code, or 'auto' for auto-detect.
    voice : str or None
        TTS voice name.
    tts_engine : str
        TTS engine: kokoro, edge-tts, pyttsx3, or auto.
    whisper_model : str
        Whisper model size: tiny, base, small, medium, large.
    device : str
        Compute device: cpu, cuda, or auto.
    duration : float or None
        Run for this many seconds then stop. None = until interrupted.

    Returns
    -------
    ToolResult
        With data containing session stats.
    """
    import time

    try:
        from copytalker import __version__
        from copytalker.core.config import AppConfig
        from copytalker.core.pipeline import TranslationPipeline

        config = AppConfig()
        config.stt.model_size = whisper_model
        config.stt.language = source
        config.translation.source_lang = source
        config.translation.target_lang = target
        config.tts.engine = tts_engine
        config.tts.language = target

        if voice:
            config.tts.voice = voice
        if device != "auto":
            config.stt.device = device
            config.translation.device = device
            config.tts.device = device

        transcriptions = []
        translations = []
        errors = []

        pipeline = TranslationPipeline(config)
        pipeline.register_callback(
            "transcription", lambda e: transcriptions.append(e.data.text)
        )
        pipeline.register_callback(
            "translation", lambda e: translations.append(e.data.translated_text)
        )
        pipeline.register_callback(
            "error", lambda e: errors.append(str(e.data))
        )

        pipeline.start()

        if duration:
            time.sleep(duration)
            pipeline.stop()
        else:
            try:
                while pipeline.is_running:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
            finally:
                pipeline.stop()

        return ToolResult(
            success=len(errors) == 0,
            data={
                "transcriptions": transcriptions,
                "translations": translations,
                "errors": errors,
            },
            metadata={
                "source": source,
                "target": target,
                "tts_engine": tts_engine,
                "whisper_model": whisper_model,
                "version": __version__,
            },
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def list_voices(
    *,
    language: str | None = None,
    engine: str = "kokoro",
) -> ToolResult:
    """List available TTS voices.

    Parameters
    ----------
    language : str or None
        Filter by language code.
    engine : str
        TTS engine: kokoro or edge-tts.

    Returns
    -------
    ToolResult
        With data containing voice lists by language.
    """
    try:
        from copytalker import __version__
        from copytalker.core.constants import (
            SUPPORTED_LANGUAGES,
            get_available_voices,
            get_language_name,
        )

        if language:
            languages = [language]
        else:
            languages = [code for code, _ in SUPPORTED_LANGUAGES]

        result = {}
        for lang in languages:
            voices = get_available_voices(lang, engine)
            if voices:
                result[lang] = {
                    "name": get_language_name(lang),
                    "voices": voices,
                }

        return ToolResult(
            success=True,
            data=result,
            metadata={"engine": engine, "version": __version__},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def list_languages() -> ToolResult:
    """List supported languages.

    Returns
    -------
    ToolResult
        With data containing language code-name pairs.
    """
    try:
        from copytalker import __version__
        from copytalker.core.constants import SUPPORTED_LANGUAGES

        data = [{"code": code, "name": name} for code, name in SUPPORTED_LANGUAGES]

        return ToolResult(
            success=True,
            data=data,
            metadata={"version": __version__},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))
