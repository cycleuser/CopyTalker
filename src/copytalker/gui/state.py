"""Centralized application state for CopyTalker GUI."""

from __future__ import annotations

from dataclasses import dataclass, field

from copytalker.core.config import (
    AppConfig,
    AudioConfig,
    STTConfig,
    TranslationConfig,
    TTSConfig,
    HistoryConfig,
    get_device,
)
from copytalker.core.constants import AUTO_DETECT_CODE


@dataclass
class AppState:
    """Observable application state shared between views and controllers."""

    # Language settings
    source_lang: str = "auto"
    target_lang: str = "zh"

    # TTS settings
    tts_engine: str = "auto"
    voice: str = ""
    ref_audio_path: str = ""
    emotion: str = "neutral"

    # Advanced settings
    translation_model: str = "helsinki"
    trans_device: str = field(default_factory=lambda: get_device())
    tts_device: str = field(default_factory=lambda: get_device())
    calibrated_noise_level: float = 0.0

    # History settings
    history_enabled: bool = True
    save_original_audio: bool = True
    save_translated_audio: bool = True

    # Input mode
    capture_mode: str = "ptt"  # "ptt" or "vad"

    # Runtime state (not persisted)
    is_running: bool = False
    is_recording_ptt: bool = False


def build_app_config(state: AppState) -> AppConfig:
    """Convert AppState to the AppConfig used by TranslationPipeline."""
    source = state.source_lang if state.source_lang != "auto" else AUTO_DETECT_CODE
    target = state.target_lang

    tts_config = TTSConfig(
        engine=state.tts_engine,
        voice=state.voice if state.voice else None,
        language=target,
        device=state.tts_device,
    )

    # Apply voice cloning reference if set
    if state.ref_audio_path:
        tts_config.indextts_reference_audio = state.ref_audio_path
        tts_config.fish_speech_reference_audio = state.ref_audio_path

    # Apply emotion if set
    if state.emotion and state.emotion != "neutral":
        tts_config.indextts_emotion = state.emotion
        tts_config.fish_speech_emotion = state.emotion

    history_config = HistoryConfig(
        enabled=state.history_enabled,
        save_original_audio=state.save_original_audio,
        save_translated_audio=state.save_translated_audio,
    )

    config = AppConfig(
        audio=AudioConfig(
            calibrated_noise_level=state.calibrated_noise_level,
        ),
        stt=STTConfig(
            language=source,
        ),
        translation=TranslationConfig(
            source_lang=source,
            target_lang=target,
            model_name=state.translation_model,
            device=state.trans_device,
        ),
        tts=tts_config,
        history=history_config,
    )
    return config
