"""
CopyTalker - Unified Python API.

Provides ToolResult-based wrappers for programmatic usage
and agent integration.
"""

from __future__ import annotations

import tempfile
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
        TTS voice name or reference audio path.
    tts_engine : str
        TTS engine: kokoro, edge-tts, pyttsx3, indextts, fish-speech, or auto.
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


def tts_synthesize(
    *,
    text: str,
    language: str = "en",
    engine: str = "auto",
    voice: str | None = None,
    speed: float = 1.0,
    output_path: str | None = None,
    emotion: str | None = None,
    emotion_audio: str | None = None,
    target_duration: float | None = None,
    reference_audio: str | None = None,
) -> ToolResult:
    """Synthesize text to speech using any supported TTS engine.

    Supports voice cloning (IndexTTS/Fish-Speech), emotion control,
    duration control, and emotion text tags.

    Parameters
    ----------
    text : str
        Text to synthesize.
    language : str
        Target language code.
    engine : str
        TTS engine to use.
    voice : str or None
        Voice name or reference audio path.
    speed : float
        Speech speed multiplier.
    output_path : str or None
        Output WAV file path. If None, a temp file is used.
    emotion : str or None
        Emotion name (IndexTTS: happy, sad, etc. Fish-Speech: text tag).
    emotion_audio : str or None
        Path to emotion reference audio (IndexTTS v2).
    target_duration : float or None
        Target duration in seconds (IndexTTS v2).
    reference_audio : str or None
        Path to speaker reference audio for voice cloning.

    Returns
    -------
    ToolResult
        With data containing output_path and audio info.
    """
    try:
        import wave

        import numpy as np

        from copytalker import __version__
        from copytalker.core.config import TTSConfig
        from copytalker.tts.base import get_tts_engine

        tts_config = TTSConfig(
            engine=engine,
            voice=voice,
            language=language,
            speed=speed,
        )

        # Set engine-specific config
        if reference_audio:
            tts_config.indextts_reference_audio = reference_audio
            tts_config.fish_speech_reference_audio = reference_audio
        if emotion:
            tts_config.indextts_emotion = emotion
            tts_config.fish_speech_emotion = emotion
        if emotion_audio:
            tts_config.indextts_emotion_audio = emotion_audio

        tts = get_tts_engine(engine, tts_config)

        # Determine effective voice/reference for voice-cloning engines
        effective_voice = reference_audio or voice

        # Use engine-specific advanced features when applicable
        audio_data = None
        sample_rate = 0

        if engine == "indextts" and hasattr(tts, "synthesize_with_emotion") and (
            emotion or emotion_audio
        ):
            from copytalker.tts.indextts import IndexTTS as IndexTTSEngine

            if isinstance(tts, IndexTTSEngine) and effective_voice:
                emotion_vector = None
                if emotion and not emotion_audio:
                    emotion_vector = IndexTTSEngine.make_emotion_vector(emotion)
                audio_data, sample_rate = tts.synthesize_with_emotion(
                    text,
                    effective_voice,
                    emotion_audio=emotion_audio,
                    emotion_vector=emotion_vector,
                )

        elif engine == "indextts" and hasattr(tts, "synthesize_with_duration") and target_duration:
            from copytalker.tts.indextts import IndexTTS as IndexTTSEngine

            if isinstance(tts, IndexTTSEngine) and effective_voice:
                audio_data, sample_rate = tts.synthesize_with_duration(
                    text, effective_voice, target_duration
                )

        elif engine == "fish-speech" and emotion:
            from copytalker.tts.fish_speech import FishSpeechTTS

            if isinstance(tts, FishSpeechTTS):
                audio_data, sample_rate = tts.synthesize_with_emotion(
                    text, emotion, effective_voice, speed, language
                )

        # Default: use standard synthesize
        if audio_data is None:
            audio_data, sample_rate = tts.synthesize(
                text, language, effective_voice, speed
            )

        # Write output
        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = tmp.name
            tmp.close()

        _write_wav(output_path, audio_data, sample_rate)

        return ToolResult(
            success=True,
            data={
                "output_path": output_path,
                "sample_rate": sample_rate,
                "duration_seconds": len(audio_data) / sample_rate if sample_rate > 0 else 0,
                "samples": len(audio_data),
            },
            metadata={
                "engine": engine,
                "language": language,
                "voice": voice,
                "emotion": emotion,
                "version": __version__,
            },
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def clone_voice(
    *,
    text: str,
    reference_audio: str,
    engine: str = "indextts",
    language: str = "en",
    output_path: str | None = None,
    emotion: str | None = None,
) -> ToolResult:
    """Clone a voice from reference audio and synthesize text.

    Parameters
    ----------
    text : str
        Text to speak in the cloned voice.
    reference_audio : str
        Path to reference audio file (5-30 seconds).
    engine : str
        Voice cloning engine: indextts or fish-speech.
    language : str
        Target language code.
    output_path : str or None
        Output WAV file path.
    emotion : str or None
        Optional emotion to apply.

    Returns
    -------
    ToolResult
        With data containing output_path and audio info.
    """
    return tts_synthesize(
        text=text,
        language=language,
        engine=engine,
        reference_audio=reference_audio,
        output_path=output_path,
        emotion=emotion,
    )


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
        TTS engine: kokoro, edge-tts, indextts, or fish-speech.

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


def list_emotions(
    *,
    engine: str = "fish-speech",
) -> ToolResult:
    """List available emotion tags for a TTS engine.

    Parameters
    ----------
    engine : str
        TTS engine: indextts or fish-speech.

    Returns
    -------
    ToolResult
        With data containing emotion names/tags.
    """
    try:
        from copytalker import __version__

        if engine == "indextts":
            from copytalker.core.constants import INDEXTTS_EMOTIONS

            data = {
                "emotions": INDEXTTS_EMOTIONS,
                "control_methods": [
                    "emotion_audio: provide a reference audio with desired emotion",
                    "emotion_vector: 8-element float list "
                    "[happy, angry, sad, fearful, surprised, disgusted, contemptuous, neutral]",
                    "use_emotion_text: infer emotion from text content (v2 only)",
                ],
            }
        elif engine == "fish-speech":
            from copytalker.core.constants import FISH_SPEECH_EMOTION_TAGS

            data = {
                "emotions": FISH_SPEECH_EMOTION_TAGS,
                "usage": "Prepend emotion tag to text: '(happy) Hello world!'",
            }
        else:
            return ToolResult(
                success=False,
                error=f"Emotion control not supported for engine: {engine}",
            )

        return ToolResult(
            success=True,
            data=data,
            metadata={"engine": engine, "version": __version__},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


def _write_wav(path: str, audio: Any, sample_rate: int) -> None:
    """Write a float32 numpy array to a WAV file."""
    import wave

    import numpy as np

    # Convert float32 [-1, 1] to int16
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
