"""Pipeline controller managing TranslationPipeline and PTT capture."""

from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING

from copytalker.audio.recorder import VoiceRecorder
from copytalker.gui.state import AppState, build_app_config

if TYPE_CHECKING:
    from copytalker.core.pipeline import TranslationPipeline

logger = logging.getLogger(__name__)


class PipelineController:
    """
    Wraps TranslationPipeline and manages Push-to-Talk capture.

    In PTT mode, uses VoiceRecorder to capture audio while the spacebar
    is held, then injects the recorded segment into the pipeline's STT
    queue on release.

    In VAD mode, delegates entirely to the pipeline's built-in AudioCapturer.
    """

    def __init__(self, event_queue: queue.Queue) -> None:
        self._event_queue = event_queue
        self._pipeline: TranslationPipeline | None = None
        self._ptt_recorder: VoiceRecorder | None = None
        self._is_ptt_recording = False
        self._state: AppState | None = None

    # ------------------------------------------------------------------
    # Pipeline lifecycle
    # ------------------------------------------------------------------

    def start(self, state: AppState) -> None:
        """Start the pipeline in a background thread."""
        self._state = state
        thread = threading.Thread(
            target=self._start_pipeline,
            args=(state,),
            name="PipelineStartThread",
            daemon=True,
        )
        thread.start()

    def _start_pipeline(self, state: AppState) -> None:
        """Initialize and start the translation pipeline (runs in thread)."""
        try:
            from copytalker.core.pipeline import TranslationPipeline

            config = build_app_config(state)
            self._pipeline = TranslationPipeline(config)

            # Register callbacks that post to the event queue
            for event_type in ("transcription", "translation", "status", "error", "synthesis"):
                self._pipeline.register_callback(
                    event_type,
                    lambda evt, et=event_type: self._event_queue.put((et, evt.data)),
                )

            capture_mode = state.capture_mode  # "ptt" or "vad"
            self._pipeline.start(capture_mode=capture_mode)

            state.is_running = True
            self._event_queue.put(("started", capture_mode))
        except Exception as exc:
            logger.error(f"Failed to start pipeline: {exc}")
            self._event_queue.put(("error", f"Start failed: {exc}"))

    def stop(self) -> None:
        """Stop the pipeline in a background thread."""
        thread = threading.Thread(
            target=self._stop_pipeline,
            name="PipelineStopThread",
            daemon=True,
        )
        thread.start()

    def _stop_pipeline(self) -> None:
        """Stop and clean up the pipeline (runs in thread)."""
        try:
            if self._ptt_recorder and self._ptt_recorder.is_recording:
                self._ptt_recorder.stop()
                self._is_ptt_recording = False

            if self._pipeline:
                self._pipeline.stop()
                self._pipeline = None

            if self._state:
                self._state.is_running = False
                self._state.is_recording_ptt = False

            self._event_queue.put(("stopped", None))
        except Exception as exc:
            logger.error(f"Error stopping pipeline: {exc}")
            self._event_queue.put(("error", f"Stop error: {exc}"))

    # ------------------------------------------------------------------
    # PTT capture
    # ------------------------------------------------------------------

    def start_ptt_capture(self) -> None:
        """Begin recording audio (spacebar pressed)."""
        if self._is_ptt_recording:
            return
        if self._pipeline is None:
            return

        self._ptt_recorder = VoiceRecorder()
        self._ptt_recorder.start()
        self._is_ptt_recording = True

        if self._state:
            self._state.is_recording_ptt = True

        self._event_queue.put(("ptt_recording", True))
        logger.info("PTT capture started")

    def stop_ptt_capture(self) -> None:
        """Stop recording and inject audio into pipeline (spacebar released)."""
        if not self._is_ptt_recording:
            return
        if self._ptt_recorder is None:
            return

        self._ptt_recorder.stop()
        self._is_ptt_recording = False

        if self._state:
            self._state.is_recording_ptt = False

        self._event_queue.put(("ptt_recording", False))
        self._event_queue.put(("ptt_processing", True))

        # Inject the recorded segment into the pipeline
        audio = self._ptt_recorder.get_audio_array()
        if len(audio) > 0 and self._pipeline is not None:
            self._pipeline.inject_audio_segment(audio)
            logger.info(f"PTT segment injected: {len(audio)} samples ({len(audio) / 16000:.1f}s)")
        else:
            logger.warning("PTT capture produced no audio")
            self._event_queue.put(("ptt_processing", False))

        self._ptt_recorder = None

    def get_ptt_rms_level(self) -> float:
        """Get current RMS level during PTT recording."""
        if self._ptt_recorder and self._is_ptt_recording:
            return self._ptt_recorder.get_rms_level()
        return 0.0

    @property
    def is_running(self) -> bool:
        return self._pipeline is not None and self._state is not None and self._state.is_running

    @property
    def is_ptt_recording(self) -> bool:
        return self._is_ptt_recording
