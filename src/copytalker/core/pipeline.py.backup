"""
Main translation pipeline orchestrating all components.
"""

import logging
import queue
import threading
import time
from typing import Optional, Callable, Dict, Any

from copytalker.core.config import AppConfig
from copytalker.core.constants import AUTO_DETECT_CODE, normalize_language_code
from copytalker.core.exceptions import PipelineError
from copytalker.core.types import (
    PipelineEvent,
    PipelineCallback,
    TranscriptionResult,
    TranslationResult,
)
from copytalker.audio.capture import AudioCapturer
from copytalker.audio.playback import ThreadSafeAudioPlayer
from copytalker.speech.recognizer import WhisperRecognizer
from copytalker.translation.translator import UnifiedTranslator
from copytalker.tts.base import get_tts_engine

logger = logging.getLogger(__name__)


class TranslationPipeline:
    """
    Main pipeline orchestrating real-time speech-to-speech translation.
    
    Components:
    1. Audio Capture with VAD
    2. Speech Recognition (Whisper)
    3. Translation (Helsinki-NLP/NLLB)
    4. Text-to-Speech (Kokoro/Edge/Pyttsx3)
    
    All components run in separate threads communicating via queues.
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize the translation pipeline.
        
        Args:
            config: Application configuration
        """
        self.config = config or AppConfig()
        
        # Queues for inter-thread communication
        self._text_queue: queue.Queue = queue.Queue()
        self._translation_queue: queue.Queue = queue.Queue()
        
        # Components (lazy initialized)
        self._audio_capturer: Optional[AudioCapturer] = None
        self._audio_player: Optional[ThreadSafeAudioPlayer] = None
        self._recognizer: Optional[WhisperRecognizer] = None
        self._translator: Optional[UnifiedTranslator] = None
        self._tts_engine = None
        
        # Thread control
        self._stop_event = threading.Event()
        self._threads: Dict[str, threading.Thread] = {}
        self._is_running = False
        
        # Callbacks for events
        self._callbacks: Dict[str, list] = {
            "transcription": [],
            "translation": [],
            "synthesis": [],
            "error": [],
            "status": [],
        }
    
    def register_callback(self, event_type: str, callback: PipelineCallback) -> None:
        """
        Register a callback for pipeline events.
        
        Args:
            event_type: Event type ('transcription', 'translation', 'synthesis', 'error', 'status')
            callback: Callback function
        """
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)
        else:
            raise ValueError(f"Unknown event type: {event_type}")
    
    def _emit_event(self, event_type: str, data: Any) -> None:
        """Emit an event to all registered callbacks."""
        event = PipelineEvent(
            event_type=event_type,
            data=data,
            timestamp=time.time(),
        )
        for callback in self._callbacks.get(event_type, []):
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Initializing pipeline components...")
        
        self._emit_event("status", "Initializing audio capture...")
        self._audio_capturer = AudioCapturer(self.config.audio)
        self._audio_player = ThreadSafeAudioPlayer()
        
        self._emit_event("status", "Loading speech recognition model...")
        self._recognizer = WhisperRecognizer(self.config.stt)
        
        self._emit_event("status", "Initializing translator...")
        self._translator = UnifiedTranslator(self.config.translation)
        
        self._emit_event("status", "Initializing TTS engine...")
        self._tts_engine = get_tts_engine(self.config.tts.engine, self.config.tts)
        
        logger.info("All components initialized")
    
    def start(self) -> None:
        """Start the translation pipeline."""
        if self._is_running:
            logger.warning("Pipeline is already running")
            return
        
        self._initialize_components()
        
        self._stop_event.clear()
        
        # Start audio capture
        self._audio_capturer.start()
        
        # Start processing threads
        self._threads["stt"] = threading.Thread(
            target=self._stt_loop,
            name="STTThread",
            daemon=True,
        )
        self._threads["translation"] = threading.Thread(
            target=self._translation_loop,
            name="TranslationThread",
            daemon=True,
        )
        self._threads["tts"] = threading.Thread(
            target=self._tts_loop,
            name="TTSThread",
            daemon=True,
        )
        
        for thread in self._threads.values():
            thread.start()
        
        self._is_running = True
        self._emit_event("status", "Pipeline started - listening...")
        logger.info("Translation pipeline started")
    
    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the translation pipeline.
        
        Args:
            timeout: Maximum time to wait for threads
        """
        if not self._is_running:
            return
        
        logger.info("Stopping translation pipeline...")
        self._emit_event("status", "Stopping...")
        
        self._stop_event.set()
        
        # Stop audio capture
        if self._audio_capturer:
            self._audio_capturer.stop()
        
        # Wait for threads
        for name, thread in self._threads.items():
            if thread.is_alive():
                thread.join(timeout=timeout / len(self._threads))
                if thread.is_alive():
                    logger.warning(f"Thread {name} did not stop in time")
        
        # Cleanup
        if self._audio_player:
            self._audio_player.close()
        
        self._threads.clear()
        self._is_running = False
        
        self._emit_event("status", "Stopped")
        logger.info("Translation pipeline stopped")
    
    def _stt_loop(self) -> None:
        """Speech-to-text processing loop."""
        logger.info("STT thread started")
        
        while not self._stop_event.is_set():
            try:
                audio = self._audio_capturer.get_audio_segment(timeout=1.0)
                if audio is None:
                    continue
                
                # Transcribe
                result = self._recognizer.transcribe(
                    audio,
                    self.config.audio.sample_rate,
                )
                
                if result.is_empty():
                    continue
                
                # Emit transcription event
                self._emit_event("transcription", result)
                
                # Queue for translation
                self._text_queue.put(result)
                
            except Exception as e:
                logger.error(f"STT error: {e}")
                self._emit_event("error", f"STT error: {e}")
        
        logger.info("STT thread stopped")
    
    def _translation_loop(self) -> None:
        """Translation processing loop."""
        logger.info("Translation thread started")
        
        target_lang = self.config.translation.target_lang
        
        while not self._stop_event.is_set():
            try:
                transcription: TranscriptionResult = self._text_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            try:
                source_lang = transcription.language
                
                # Translate
                result = self._translator.translate(
                    transcription.text,
                    source_lang,
                    target_lang,
                )
                
                # Emit translation event
                self._emit_event("translation", result)
                
                # Queue for TTS
                self._translation_queue.put(result)
                
            except Exception as e:
                logger.error(f"Translation error: {e}")
                self._emit_event("error", f"Translation error: {e}")
        
        logger.info("Translation thread stopped")
    
    def _tts_loop(self) -> None:
        """Text-to-speech processing loop."""
        logger.info("TTS thread started")
        
        target_lang = self.config.translation.target_lang
        voice = self.config.tts.voice
        
        while not self._stop_event.is_set():
            try:
                translation: TranslationResult = self._translation_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            try:
                text = translation.translated_text
                if not text:
                    continue
                
                # Synthesize
                audio, sample_rate = self._tts_engine.synthesize(
                    text,
                    target_lang,
                    voice,
                    self.config.tts.speed,
                )
                
                if len(audio) == 0:
                    continue
                
                # Emit synthesis event
                self._emit_event("synthesis", {
                    "text": text,
                    "audio_length": len(audio),
                    "sample_rate": sample_rate,
                })
                
                # Play audio (blocks during playback)
                # Note: ThreadSafeAudioPlayer already handles locking internally
                self._audio_player.play(audio, sample_rate, blocking=True)
                
            except Exception as e:
                logger.error(f"TTS error: {e}")
                self._emit_event("error", f"TTS error: {e}")
        
        logger.info("TTS thread stopped")
    
    @property
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._is_running
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "is_running": self._is_running,
            "source_lang": self.config.stt.language,
            "target_lang": self.config.translation.target_lang,
            "tts_engine": self.config.tts.engine,
            "voice": self.config.tts.voice,
            "threads": {name: t.is_alive() for name, t in self._threads.items()},
        }
    
    def __enter__(self) -> "TranslationPipeline":
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
