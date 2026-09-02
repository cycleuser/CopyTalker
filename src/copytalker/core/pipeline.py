"""
Main translation pipeline orchestrating all components.
"""

import logging
import queue
import threading
import time
from typing import Optional, Callable, Dict, Any

import numpy as np

from copytalker.core.config import AppConfig
from copytalker.core.constants import AUTO_DETECT_CODE, normalize_language_code
from copytalker.core.conversation import ConversationFSM, ConversationState
from copytalker.core.exceptions import PipelineError
from copytalker.core.history import ConversationHistory
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
        
        # Conversation history
        self._history: Optional[ConversationHistory] = None
        self._current_audio: Optional[np.ndarray] = None
        self._current_entry_index: Optional[int] = None

        # Conversation turn state machine (continuous dialogue mode)
        self.conversation_fsm = ConversationFSM()
        self._continuous_mode: bool = False
    
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
        
        # Initialize conversation history if enabled
        if self.config.history.enabled:
            self._history = ConversationHistory(history_dir=self.config.history.history_dir)
            self._history.start_session()
            logger.info(f"Conversation history initialized: {self._history.session_dir}")
        
        logger.info("All components initialized")
    
    def start(self, capture_mode: str = "vad", continuous: bool = False) -> None:
        """Start the translation pipeline.

        Args:
            capture_mode: "vad" for continuous VAD capture (default),
                          "ptt" to skip auto-capture (audio injected externally).
            continuous: If True, enable continuous conversation mode with
                        turn-taking FSM (Xiaoai-style). Implies VAD capture.
        """
        if self._is_running:
            logger.warning("Pipeline is already running")
            return

        self._initialize_components()

        self._stop_event.clear()
        self._continuous_mode = continuous

        # Continuous mode forces VAD capture and arms the FSM
        if continuous:
            capture_mode = "vad"
            self.conversation_fsm.transition_to(ConversationState.LISTENING)
            self._emit_event("status", "Continuous conversation started")

        # Start audio capture only in VAD mode
        if capture_mode == "vad":
            self._audio_capturer.start()
        else:
            logger.info("PTT mode: audio capture managed externally")
        
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
    
    def inject_audio_segment(self, audio: np.ndarray) -> None:
        """Inject an audio segment directly into the STT queue.

        Used in PTT mode where audio capture is managed externally.

        Args:
            audio: Float32 numpy array normalised to [-1, 1].
        """
        if self._audio_capturer is None:
            logger.warning("Cannot inject audio: capturer not initialised")
            return
        
        # Save original audio for history (PTT mode)
        if self._history and self.config.history.save_original_audio:
            self._current_audio = audio.copy()
        
        self._audio_capturer._audio_queue.put(audio)
        logger.debug(f"Injected audio segment: {len(audio)} samples")
    
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
        self.conversation_fsm.reset()
        self.conversation_fsm.request_barge_in()  # unblock any TTS wait
        
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
        
        # End conversation history session
        if self._history:
            try:
                self._history.end_session()
                logger.info(f"Conversation history saved: {self._history.markdown_path}")
            except Exception as e:
                logger.error(f"Error saving conversation history: {e}")
        
        self._threads.clear()
        self._is_running = False
        
        self._emit_event("status", "Stopped")
        logger.info("Translation pipeline stopped")
    
    def _stt_loop(self) -> None:
        """Speech-to-text processing loop."""
        logger.info("STT thread started")
        consecutive_errors = 0

        while not self._stop_event.is_set():
            try:
                audio = self._audio_capturer.get_audio_segment(timeout=1.0)
                if audio is None:
                    continue

                # Continuous mode: gate STT by conversation state.
                # Discard user speech captured while the assistant is busy
                # (processing or speaking) — prevents self-retrigger / overlap.
                if self._continuous_mode and not self.conversation_fsm.can_accept_user_speech:
                    logger.debug("Discarding audio: assistant busy")
                    continue

                # Transcription phase: transition to PROCESSING
                if self._continuous_mode:
                    self.conversation_fsm.transition_to(ConversationState.PROCESSING)

                # Save original audio for history
                if self._history and self.config.history.save_original_audio:
                    self._current_audio = audio.copy()

                # Transcribe
                result = self._recognizer.transcribe(
                    audio,
                    self.config.audio.sample_rate,
                )

                if result.is_empty():
                    self._current_audio = None
                    continue

                # Save to history
                if self._history:
                    entry_index = self._history.create_entry()
                    if self.config.history.save_original_audio and self._current_audio is not None:
                        self._history.save_original_audio(
                            self._current_audio,
                            self.config.audio.sample_rate,
                        )
                    self._history.add_transcription(result.text, result.language)
                    self._current_entry_index = entry_index

                # Emit transcription event
                self._emit_event("transcription", result)

                # Queue for translation (include entry index)
                self._text_queue.put((result, self._current_entry_index))

                self._current_audio = None
                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"STT error ({consecutive_errors}): {e}")
                self._emit_event("error", f"STT error: {e}")
                if consecutive_errors >= 5:
                    self._emit_event(
                        "error",
                        f"STT fatal: {consecutive_errors} consecutive errors, "
                        f"stopping pipeline",
                    )
                    self._stop_event.set()
                    break
                time.sleep(min(0.1 * (2 ** (consecutive_errors - 1)), 5.0))

        logger.info("STT thread stopped")
    
    def _translation_loop(self) -> None:
        """Translation processing loop."""
        logger.info("Translation thread started")
        
        target_lang = self.config.translation.target_lang
        
        while not self._stop_event.is_set():
            try:
                item = self._text_queue.get(timeout=1.0)
                if isinstance(item, tuple):
                    transcription, entry_index = item
                else:
                    transcription = item
                    entry_index = None
            except queue.Empty:
                continue
            
            try:
                source_lang = transcription.language

                # Fetch conversation context (prior completed turns) if enabled
                context = None
                if self._history and self.config.history.context_window > 0:
                    context_entries = self._history.get_context(
                        self.config.history.context_window
                    )
                    context = [
                        TranslationResult(
                            original_text=e.original_text,
                            translated_text=e.translated_text,
                            source_lang=e.original_lang,
                            target_lang=e.target_lang,
                        )
                        for e in context_entries
                    ]

                # Translate
                result = self._translator.translate(
                    transcription.text,
                    source_lang,
                    target_lang,
                    context=context,
                )
                
                # Save to history
                if self._history and entry_index is not None:
                    self._history.add_translation(
                        result.translated_text,
                        target_lang,
                        entry_index,
                    )
                
                # Emit translation event
                self._emit_event("translation", result)
                
                # Queue for TTS (include entry index)
                self._translation_queue.put((result, entry_index))
                
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
                item = self._translation_queue.get(timeout=1.0)
                if isinstance(item, tuple):
                    translation, entry_index = item
                else:
                    translation = item
                    entry_index = None
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
                
                # Save translated audio to history
                if self._history and entry_index is not None and self.config.history.save_translated_audio:
                    self._history.save_translated_audio(audio, sample_rate, entry_index)
                
                # Emit synthesis event
                self._emit_event("synthesis", {
                    "text": text,
                    "audio_length": len(audio),
                    "sample_rate": sample_rate,
                })

                # Continuous mode: enter SPEAKING state (UI shows it, STT gated)
                if self._continuous_mode:
                    self.conversation_fsm.clear_barge_in()
                    self.conversation_fsm.transition_to(ConversationState.SPEAKING)

                # Echo cancellation: pause capture during TTS playback
                self._audio_capturer.set_playback_active(True)
                # Play audio (blocks during playback).  In continuous mode,
                # poll for barge-in so the user can interrupt the assistant.
                if self._continuous_mode:
                    self._play_with_barge_in(audio, sample_rate)
                else:
                    self._audio_player.play(audio, sample_rate, blocking=True)
                self._audio_capturer.set_playback_active(False)

                # Continuous mode: hand back to the user
                if self._continuous_mode and not self._stop_event.is_set():
                    if self.conversation_fsm.consume_barge_in():
                        logger.info("Barge-in: interrupted assistant speech")
                    self.conversation_fsm.transition_to(ConversationState.LISTENING)
                
            except Exception as e:
                logger.error(f"TTS error: {e}")
                self._emit_event("error", f"TTS error: {e}")
        
        logger.info("TTS thread stopped")

    def _play_with_barge_in(self, audio: np.ndarray, sample_rate: int) -> None:
        """Play audio in chunks, aborting early on a barge-in request.

        Splits the audio into ~100ms chunks and checks
        ``conversation_fsm.consume_barge_in()`` between chunks.  This lets
        the user press Space (or the GUI barge-in button) to interrupt the
        assistant mid-sentence and immediately resume listening.
        """
        chunk_samples = int(sample_rate * 0.1)  # 100 ms
        total = len(audio)
        played = 0
        while played < total and not self._stop_event.is_set():
            if self.conversation_fsm.consume_barge_in():
                # Stop immediate playback
                try:
                    self._audio_player.stop_playback()
                except Exception:
                    pass
                return
            end = min(played + chunk_samples, total)
            chunk = audio[played:end]
            self._audio_player.play(chunk, sample_rate, blocking=True)
            played = end

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
