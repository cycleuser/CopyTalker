"""Qt pipeline controller for CopyTalker."""

from __future__ import annotations

import logging
import queue
import threading

try:
    from PySide6.QtCore import QObject, QThread, Signal
except ImportError:
    class QObject:
        def __init__(self, parent=None):
            pass

    class QThread:
        def __init__(self):
            self.started = Signal()
            self.finished = Signal()

        def start(self):
            pass

        def quit(self):
            pass

        def wait(self):
            pass

    class Signal:
        def __init__(self, *args):
            pass

        def emit(self, *args):
            pass

from copytalker.gui.qt.state import QtAppState, build_app_config_from_qt

logger = logging.getLogger(__name__)


class PipelineWorker(QThread):
    """Worker thread for running the translation pipeline."""

    # Signals
    started_signal = Signal(str)  # capture_mode
    stopped_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, state: QtAppState, event_queue: queue.Queue):
        super().__init__()
        self._state = state
        self._event_queue = event_queue
        self._pipeline = None
        self._should_stop = False

    def run(self) -> None:
        """Run the pipeline in background thread."""
        try:
            # Import pipeline lazily to avoid startup delays
            from copytalker.core.pipeline import TranslationPipeline

            # Build config from state
            config = build_app_config_from_qt(self._state)

            # Create pipeline
            self._pipeline = TranslationPipeline(config)

            # Register callbacks
            self._pipeline.register_callback("transcription", self._on_transcription)
            self._pipeline.register_callback("translation", self._on_translation)
            self._pipeline.register_callback("status", self._on_status)
            self._pipeline.register_callback("error", self._on_error)

            # Start pipeline
            self._pipeline.start()
            self.started_signal.emit(self._state.captureMode)

            # Keep thread alive while pipeline runs
            while not self._should_stop and self._pipeline:
                self.msleep(100)  # Sleep 100ms, then check again

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            logger.error(f"Pipeline error: {error_detail}")
            # Check if it's a CUDA/torch error
            error_str = str(e)
            if "nccl" in error_str.lower() or "cuda" in error_str.lower() or "torch" in error_str.lower():
                self.error_signal.emit(f"CUDA/PyTorch error. Try setting COPYTALKER_DEVICE=cpu: {error_str}")
            else:
                self.error_signal.emit(str(e))
        finally:
            self._cleanup()
            self.stopped_signal.emit()

    def stop_pipeline(self) -> None:
        """Request pipeline stop."""
        self._should_stop = True
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception as e:
                logger.error(f"Error stopping pipeline: {e}")

    def _cleanup(self) -> None:
        """Cleanup pipeline resources."""
        if self._pipeline:
            try:
                self._pipeline = None
            except Exception:
                pass

    def _on_transcription(self, event) -> None:
        """Handle transcription event."""
        self._event_queue.put(("transcription", event.data))

    def _on_translation(self, event) -> None:
        """Handle translation event."""
        self._event_queue.put(("translation", event.data))

    def _on_status(self, event) -> None:
        """Handle status event."""
        self._event_queue.put(("status", event.data))

    def _on_error(self, event) -> None:
        """Handle error event."""
        self._event_queue.put(("error", event.data))


class QtPipelineController(QObject):
    """Qt-compatible pipeline controller using QThread."""

    # Signals
    started = Signal(str)  # capture_mode
    stopped = Signal()
    ptt_recording = Signal(bool)
    ptt_processing = Signal(bool)
    error_occurred = Signal(str)

    def __init__(self, event_queue: queue.Queue, parent: QObject | None = None):
        super().__init__(parent)
        self._event_queue = event_queue
        self._worker: PipelineWorker | None = None
        self._is_running = False
        self._is_ptt_recording = False
        self._ptt_audio_data = None

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_ptt_recording(self) -> bool:
        return self._is_ptt_recording

    def start(self, state: QtAppState) -> None:
        """Start the translation pipeline."""
        if self._is_running:
            return

        logger.info("Starting Qt pipeline controller")

        # Create and start worker thread
        self._worker = PipelineWorker(state, self._event_queue)
        self._worker.started_signal.connect(self._on_worker_started)
        self._worker.stopped_signal.connect(self._on_worker_stopped)
        self._worker.error_signal.connect(self._on_worker_error)
        self._worker.start()

    def stop(self) -> None:
        """Stop the translation pipeline."""
        if not self._is_running:
            return

        logger.info("Stopping Qt pipeline controller")

        # Stop worker thread
        if self._worker:
            self._worker.stop_pipeline()
            self._worker.quit()
            self._worker.wait()
            self._worker = None

    def _on_worker_started(self, capture_mode: str) -> None:
        """Handle worker started signal."""
        self._is_running = True
        self.started.emit(capture_mode)

    def _on_worker_stopped(self) -> None:
        """Handle worker stopped signal."""
        self._is_running = False
        self.stopped.emit()

    def _on_worker_error(self, error_msg: str) -> None:
        """Handle worker error signal."""
        self._is_running = False
        self.error_occurred.emit(error_msg)
        self.stopped.emit()

    def start_ptt_capture(self) -> None:
        """Start PTT capture."""
        if not self._is_running:
            return

        logger.info("Starting PTT capture")
        self._is_ptt_recording = True
        self.ptt_recording.emit(True)

        # Start audio recording in background
        def record_audio():
            try:
                from copytalker.audio.recorder import VoiceRecorder
                recorder = VoiceRecorder(sample_rate=16000)
                recorder.start()

                # Store recorder for later use
                self._ptt_recorder = recorder

            except Exception as e:
                logger.error(f"PTT recording error: {e}")
                self.ptt_recording.emit(False)
                self._is_ptt_recording = False

        thread = threading.Thread(target=record_audio, daemon=True)
        thread.start()

    def stop_ptt_capture(self) -> None:
        """Stop PTT capture and inject audio."""
        if not self._is_ptt_recording:
            return

        logger.info("Stopping PTT capture")
        self._is_ptt_recording = False
        self.ptt_recording.emit(False)
        self.ptt_processing.emit(True)

        # Stop recording and get audio data
        def process_audio():
            try:
                if hasattr(self, '_ptt_recorder'):
                    recorder = self._ptt_recorder
                    recorder.stop()

                    # Get audio data
                    audio_array = recorder.get_audio_array()
                    if len(audio_array) > 0 and self._worker and self._worker._pipeline:
                        # Inject audio into pipeline
                        self._worker._pipeline.inject_audio_segment(audio_array)

                    # Cleanup
                    delattr(self, '_ptt_recorder')

            except Exception as e:
                logger.error(f"PTT processing error: {e}")
            finally:
                self.ptt_processing.emit(False)

        thread = threading.Thread(target=process_audio, daemon=True)
        thread.start()

    def get_ptt_rms_level(self) -> float:
        """Get current PTT recording RMS level."""
        try:
            if hasattr(self, '_ptt_recorder'):
                return self._ptt_recorder.get_rms_level()
        except Exception:
            pass
        return 0.0
