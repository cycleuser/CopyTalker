"""Qt-compatible application state for CopyTalker GUI."""

from __future__ import annotations

try:
    from PySide6.QtCore import Property, QObject, Signal
except ImportError:
    # Fallback for environments without PySide6
    class QObject:
        def __init__(self):
            pass

    class Signal:
        def __init__(self, *args):
            pass

        def emit(self, *args):
            pass

    def Property(type_, fget, fset=None, notify=None):
        return property(fget, fset)

from copytalker.core.config import (
    AppConfig,
    AudioConfig,
    STTConfig,
    TranslationConfig,
    TTSConfig,
    get_device,
)
from copytalker.core.constants import AUTO_DETECT_CODE


class QtAppState(QObject):
    """Qt-compatible observable application state shared between views and controllers."""

    # Signals for property changes
    sourceLangChanged = Signal(str)
    targetLangChanged = Signal(str)
    ttsEngineChanged = Signal(str)
    voiceChanged = Signal(str)
    refAudioPathChanged = Signal(str)
    emotionChanged = Signal(str)
    translationModelChanged = Signal(str)
    transDeviceChanged = Signal(str)
    ttsDeviceChanged = Signal(str)
    calibratedNoiseLevelChanged = Signal(float)
    captureModeChanged = Signal(str)
    isRunningChanged = Signal(bool)
    isRecordingPttChanged = Signal(bool)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)

        # Language settings
        self._source_lang: str = "auto"
        self._target_lang: str = "zh"

        # TTS settings
        self._tts_engine: str = "auto"
        self._voice: str = ""
        self._ref_audio_path: str = ""
        self._emotion: str = "neutral"

        # Advanced settings
        self._translation_model: str = "helsinki"
        self._trans_device: str = get_device()
        self._tts_device: str = get_device()
        self._calibrated_noise_level: float = 0.0

        # Input mode
        self._capture_mode: str = "ptt"  # "ptt" or "vad"

        # Runtime state (not persisted)
        self._is_running: bool = False
        self._is_recording_ptt: bool = False

    # Property: source_lang
    @Property(str, notify=sourceLangChanged)
    def sourceLang(self) -> str:
        return self._source_lang

    @sourceLang.setter
    def sourceLang(self, value: str) -> None:
        if self._source_lang != value:
            self._source_lang = value
            self.sourceLangChanged.emit(value)

    # Property: target_lang
    @Property(str, notify=targetLangChanged)
    def targetLang(self) -> str:
        return self._target_lang

    @targetLang.setter
    def targetLang(self, value: str) -> None:
        if self._target_lang != value:
            self._target_lang = value
            self.targetLangChanged.emit(value)

    # Property: tts_engine
    @Property(str, notify=ttsEngineChanged)
    def ttsEngine(self) -> str:
        return self._tts_engine

    @ttsEngine.setter
    def ttsEngine(self, value: str) -> None:
        if self._tts_engine != value:
            self._tts_engine = value
            self.ttsEngineChanged.emit(value)

    # Property: voice
    @Property(str, notify=voiceChanged)
    def voice(self) -> str:
        return self._voice

    @voice.setter
    def voice(self, value: str) -> None:
        if self._voice != value:
            self._voice = value
            self.voiceChanged.emit(value)

    # Property: ref_audio_path
    @Property(str, notify=refAudioPathChanged)
    def refAudioPath(self) -> str:
        return self._ref_audio_path

    @refAudioPath.setter
    def refAudioPath(self, value: str) -> None:
        if self._ref_audio_path != value:
            self._ref_audio_path = value
            self.refAudioPathChanged.emit(value)

    # Property: emotion
    @Property(str, notify=emotionChanged)
    def emotion(self) -> str:
        return self._emotion

    @emotion.setter
    def emotion(self, value: str) -> None:
        if self._emotion != value:
            self._emotion = value
            self.emotionChanged.emit(value)

    # Property: translation_model
    @Property(str, notify=translationModelChanged)
    def translationModel(self) -> str:
        return self._translation_model

    @translationModel.setter
    def translationModel(self, value: str) -> None:
        if self._translation_model != value:
            self._translation_model = value
            self.translationModelChanged.emit(value)

    # Property: trans_device
    @Property(str, notify=transDeviceChanged)
    def transDevice(self) -> str:
        return self._trans_device

    @transDevice.setter
    def transDevice(self, value: str) -> None:
        if self._trans_device != value:
            self._trans_device = value
            self.transDeviceChanged.emit(value)

    # Property: tts_device
    @Property(str, notify=ttsDeviceChanged)
    def ttsDevice(self) -> str:
        return self._tts_device

    @ttsDevice.setter
    def ttsDevice(self, value: str) -> None:
        if self._tts_device != value:
            self._tts_device = value
            self.ttsDeviceChanged.emit(value)

    # Property: calibrated_noise_level
    @Property(float, notify=calibratedNoiseLevelChanged)
    def calibratedNoiseLevel(self) -> float:
        return self._calibrated_noise_level

    @calibratedNoiseLevel.setter
    def calibratedNoiseLevel(self, value: float) -> None:
        if self._calibrated_noise_level != value:
            self._calibrated_noise_level = value
            self.calibratedNoiseLevelChanged.emit(value)

    # Property: capture_mode
    @Property(str, notify=captureModeChanged)
    def captureMode(self) -> str:
        return self._capture_mode

    @captureMode.setter
    def captureMode(self, value: str) -> None:
        if self._capture_mode != value:
            self._capture_mode = value
            self.captureModeChanged.emit(value)

    # Property: is_running
    @Property(bool, notify=isRunningChanged)
    def isRunning(self) -> bool:
        return self._is_running

    @isRunning.setter
    def isRunning(self, value: bool) -> None:
        if self._is_running != value:
            self._is_running = value
            self.isRunningChanged.emit(value)

    # Property: is_recording_ptt
    @Property(bool, notify=isRecordingPttChanged)
    def isRecordingPtt(self) -> bool:
        return self._is_recording_ptt

    @isRecordingPtt.setter
    def isRecordingPtt(self, value: bool) -> None:
        if self._is_recording_ptt != value:
            self._is_recording_ptt = value
            self.isRecordingPttChanged.emit(value)


def build_app_config_from_qt(state: QtAppState) -> AppConfig:
    """Convert QtAppState to the AppConfig used by TranslationPipeline."""
    source = state.sourceLang if state.sourceLang != "auto" else AUTO_DETECT_CODE
    target = state.targetLang

    tts_config = TTSConfig(
        engine=state.ttsEngine,
        voice=state.voice if state.voice else None,
        language=target,
        device=state.ttsDevice,
    )

    # Apply voice cloning reference if set
    if state.refAudioPath:
        tts_config.indextts_reference_audio = state.refAudioPath
        tts_config.fish_speech_reference_audio = state.refAudioPath

    # Apply emotion if set
    if state.emotion and state.emotion != "neutral":
        tts_config.indextts_emotion = state.emotion
        tts_config.fish_speech_emotion = state.emotion

    config = AppConfig(
        audio=AudioConfig(
            calibrated_noise_level=state.calibratedNoiseLevel,
        ),
        stt=STTConfig(
            language=source,
        ),
        translation=TranslationConfig(
            source_lang=source,
            target_lang=target,
            model_name=state.translationModel,
            device=state.transDevice,
        ),
        tts=tts_config,
    )
    return config
