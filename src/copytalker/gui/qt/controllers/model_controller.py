"""Qt model controller for CopyTalker."""

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

logger = logging.getLogger(__name__)


class ModelDownloadWorker(QThread):
    """Worker thread for model downloads."""

    # Signals
    progress_signal = Signal(str)
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, download_func, *args):
        super().__init__()
        self._download_func = download_func
        self._args = args

    def run(self) -> None:
        """Run the download in background thread."""
        try:
            self._download_func(*self._args)
            self.finished_signal.emit()
        except Exception as e:
            logger.error(f"Download error: {e}")
            self.error_signal.emit(str(e))


class CalibrationWorker(QThread):
    """Worker thread for noise calibration."""

    # Signals
    completed_signal = Signal(float)
    error_signal = Signal(str)

    def __init__(self):
        super().__init__()

    def run(self) -> None:
        """Run calibration in background thread."""
        try:
            from copytalker.audio.capture import AudioCapturer
            from copytalker.core.config import AudioConfig

            config = AudioConfig()
            capturer = AudioCapturer(config)

            noise_level = capturer.calibrate_noise(duration_s=2.0)
            self.completed_signal.emit(noise_level)

        except Exception as e:
            logger.error(f"Calibration error: {e}")
            self.error_signal.emit(str(e))


class QtModelController(QObject):
    """Qt-compatible model download controller using QThread."""

    # Signals
    download_progress = Signal(str)
    download_finished = Signal()
    calibration_done = Signal(float)
    error_occurred = Signal(str)

    def __init__(self, event_queue: queue.Queue, parent: QObject | None = None):
        super().__init__(parent)
        self._event_queue = event_queue
        self._current_worker: QThread | None = None

    def download(self, what: str) -> None:
        """Download specified models."""
        if self._current_worker and self._current_worker.isRunning():
            logger.warning("Download already in progress")
            return

        logger.info(f"Starting download: {what}")

        def download_task():
            try:
                from copytalker.utils.model_cache import ModelCache
                cache = ModelCache()

                if what == "indextts":
                    self.download_progress.emit("Downloading IndexTTS v2...")
                    cache.download_indextts_model()
                elif what == "fish-speech":
                    self.download_progress.emit("Downloading Fish-Speech...")
                    cache.download_fish_speech_model()
                elif what == "kokoro":
                    self.download_progress.emit("Downloading Kokoro TTS...")
                    cache.download_kokoro_model()
                elif what == "whisper":
                    self.download_progress.emit("Downloading Whisper (small)...")
                    cache.download_whisper_model("small")
                elif what == "translation":
                    self.download_progress.emit("Downloading translation models...")
                    self._download_translation_models(cache)
                elif what == "all":
                    self.download_progress.emit("Downloading all models...")
                    self._download_all_models(cache)

                self.download_progress.emit("Download complete!")

            except Exception as e:
                logger.error(f"Download failed: {e}")
                self.error_occurred.emit(str(e))

        self._current_worker = ModelDownloadWorker(download_task)
        self._current_worker.progress_signal.connect(self.download_progress)
        self._current_worker.finished_signal.connect(self.download_finished)
        self._current_worker.error_signal.connect(self.error_occurred)
        self._current_worker.start()

    def _download_translation_models(self, cache) -> None:
        """Download translation models."""
        from copytalker.core.constants import DEFAULT_TRANSLATION_MODELS

        # Download Helsinki models
        for key, models in DEFAULT_TRANSLATION_MODELS.items():
            if key == "multilingual":
                continue
            for model_name in models:
                self.download_progress.emit(f"Downloading {model_name}...")
                cache.download_translation_model(model_name)

        # Download NLLB
        self.download_progress.emit("Downloading NLLB...")
        cache.download_translation_model("facebook/nllb-200-distilled-600M")

    def _download_all_models(self, cache) -> None:
        """Download all recommended models."""
        # Download TTS models
        self.download_progress.emit("Downloading TTS models...")
        cache.download_kokoro_model()
        cache.download_indextts_model()
        cache.download_fish_speech_model()

        # Download Whisper
        self.download_progress.emit("Downloading Whisper...")
        cache.download_whisper_model("small")

        # Download translation models
        self._download_translation_models(cache)

    def download_translation_for_langs(self, langs: list[str]) -> None:
        """Download translation models for specified languages."""
        if self._current_worker and self._current_worker.isRunning():
            logger.warning("Download already in progress")
            return

        logger.info(f"Downloading translation models for languages: {langs}")

        def download_task():
            try:
                from copytalker.core.constants import DEFAULT_TRANSLATION_MODELS
                from copytalker.utils.model_cache import ModelCache

                cache = ModelCache()

                for lang in langs:
                    # Download bidirectional models: en->lang and lang->en
                    directions = [f"en->{lang}", f"{lang}->en"]

                    for direction in directions:
                        models = DEFAULT_TRANSLATION_MODELS.get(direction, [])
                        for model_name in models:
                            self.download_progress.emit(f"Downloading {model_name}...")
                            cache.download_translation_model(model_name)

                    # Also download NLLB for broader coverage
                    self.download_progress.emit(f"Downloading NLLB for {lang}...")
                    cache.download_translation_model("facebook/nllb-200-distilled-600M")

                self.download_progress.emit("Translation model download complete!")

            except Exception as e:
                logger.error(f"Translation download failed: {e}")
                self.error_occurred.emit(str(e))

        self._current_worker = ModelDownloadWorker(download_task)
        self._current_worker.progress_signal.connect(self.download_progress)
        self._current_worker.finished_signal.connect(self.download_finished)
        self._current_worker.error_signal.connect(self.error_occurred)
        self._current_worker.start()

    def refresh_cache_info(self) -> None:
        """Refresh model cache information."""
        def refresh_task():
            try:
                from copytalker.utils.model_cache import ModelCache, format_size
                cache = ModelCache()
                size = format_size(cache.get_cache_size())
                cached = cache.get_cached_models()

                info_lines = [f"Cache dir: {cache.cache_dir}", f"Total size: {size}"]
                for cat, items in cached.items():
                    if items:
                        info_lines.append(f"  {cat}: {', '.join(items)}")

                # Send to event queue for UI update
                self._event_queue.put(("cache_info", "\n".join(info_lines)))

            except Exception as e:
                logger.error(f"Cache refresh error: {e}")
                self._event_queue.put(("cache_info", f"Error: {e}"))

        thread = threading.Thread(target=refresh_task, daemon=True)
        thread.start()

    def calibrate_noise(self) -> None:
        """Perform noise calibration."""
        if self._current_worker and self._current_worker.isRunning():
            logger.warning("Operation already in progress")
            return

        logger.info("Starting noise calibration")
        self.download_progress.emit("Calibrating noise... Please stay quiet for 2 seconds")

        self._current_worker = CalibrationWorker()
        self._current_worker.completed_signal.connect(self.calibration_done)
        self._current_worker.error_signal.connect(self.error_occurred)
        self._current_worker.start()
