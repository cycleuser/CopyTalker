"""Main Qt application for CopyTalker."""

from __future__ import annotations

import logging
import queue
import sys

try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as e:
    print(f"PySide6 not available: {e}")
    print("Install with: pip install copytalker[gui]")
    sys.exit(1)

from copytalker import __version__
from copytalker.gui.qt.controllers.model_controller import QtModelController
from copytalker.gui.qt.controllers.pipeline_controller import QtPipelineController
from copytalker.gui.qt.state import QtAppState
from copytalker.gui.qt.views.conversation import QtConversationView
from copytalker.gui.qt.views.settings import QtSettingsDialog

logger = logging.getLogger(__name__)


class CopyTalkerApp(QMainWindow):
    WINDOW_TITLE = f"CopyTalker v{__version__}"
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600

    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.WINDOW_TITLE)
        self.setMinimumSize(500, 400)
        self.resize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)

        self._state = QtAppState(self)
        self._event_queue: queue.Queue = queue.Queue()
        self._pipeline_ctrl = QtPipelineController(self._event_queue, self)
        self._model_ctrl = QtModelController(self._event_queue, self)

        self._space_down = False

        self._setup_ui()
        self._setup_menu_bar()
        self._setup_status_bar()
        self._setup_connections()
        self._start_event_loop()
        self._setup_level_polling()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self._conversation_view = QtConversationView(self)
        main_layout.addWidget(self._conversation_view)

        self._settings_dialog = None

    def _setup_menu_bar(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        start_action = file_menu.addAction("&Start Translation")
        start_action.triggered.connect(self._on_start)
        stop_action = file_menu.addAction("S&top")
        stop_action.triggered.connect(self._on_stop)
        file_menu.addSeparator()
        quit_action = file_menu.addAction("&Quit")
        quit_action.triggered.connect(self.close)

        edit_menu = menu_bar.addMenu("&Edit")
        settings_action = edit_menu.addAction("&Settings")
        settings_action.triggered.connect(self._show_settings_dialog)

        help_menu = menu_bar.addMenu("&Help")
        about_action = help_menu.addAction("&About CopyTalker")
        about_action.triggered.connect(self._show_about)

    def _setup_status_bar(self):
        sb = self.statusBar()
        sb.showMessage("Ready - Press Start to begin")
        version_label = QLabel(f"v{__version__}")
        sb.addPermanentWidget(version_label)

    def _setup_connections(self):
        self._conversation_view.ptt_pressed.connect(self._on_ptt_pressed)
        self._conversation_view.ptt_released.connect(self._on_ptt_released)
        self._conversation_view.start_requested.connect(self._on_start)
        self._conversation_view.stop_requested.connect(self._on_stop)
        self._conversation_view.settings_requested.connect(self._show_settings_dialog)

        self._pipeline_ctrl.started.connect(self._on_pipeline_started)
        self._pipeline_ctrl.stopped.connect(self._on_pipeline_stopped)

    def _start_event_loop(self):
        self._event_timer = QTimer()
        self._event_timer.timeout.connect(self._process_events)
        self._event_timer.start(100)

    def _process_events(self):
        try:
            while True:
                event_type, data = self._event_queue.get_nowait()
                self._handle_event(event_type, data)
        except queue.Empty:
            pass

    def _handle_event(self, event_type: str, data):
        handlers = {
            "transcription": lambda: self._conversation_view.add_transcription(data),
            "translation": lambda: self._conversation_view.add_translation(data),
            "status": lambda: self._on_status_event(str(data)),
            "error": lambda: self._on_error_event(str(data)),
            "ptt_recording": lambda: self._on_ptt_recording_event(bool(data)),
            "ptt_processing": lambda: self._conversation_view.set_processing(bool(data)),
            "synthesis": lambda: self._on_synthesis_done(),
        }
        handler = handlers.get(event_type)
        if handler:
            handler()

    def _on_status_event(self, message: str):
        self._conversation_view.set_status("info", message)
        self.statusBar().showMessage(message, 5000)

    def _on_error_event(self, message: str):
        self._conversation_view.set_status("error", message)
        self.statusBar().showMessage(f"Error: {message}", 8000)

    def _setup_level_polling(self):
        self._level_timer = QTimer()
        self._level_timer.timeout.connect(self._poll_audio_level)

    def _start_level_polling(self):
        self._level_timer.start(50)

    def _stop_level_polling(self):
        self._level_timer.stop()
        self._conversation_view.ptt_bar.set_audio_level(0.0)

    def _poll_audio_level(self):
        level = self._pipeline_ctrl.get_ptt_rms_level()
        self._conversation_view.ptt_bar.set_audio_level(level)

    def _show_settings_dialog(self):
        if self._settings_dialog is None:
            self._settings_dialog = QtSettingsDialog(
                state=self._state,
                event_queue=self._event_queue,
                model_ctrl=self._model_ctrl,
                parent=self,
                ui_lang=self._conversation_view._ui_lang,
            )
            self._settings_dialog.start_requested.connect(self._on_start)
            self._settings_dialog.stop_requested.connect(self._on_stop)
            self._conversation_view.ui_language_changed.connect(self._on_ui_language_changed)
        self._settings_dialog.show()
        self._settings_dialog.raise_()
        self._settings_dialog.activateWindow()

    def _on_ui_language_changed(self, lang: str):
        if self._settings_dialog:
            self._settings_dialog.set_ui_language(lang)

    def _show_about(self):
        QMessageBox.about(
            self,
            "About CopyTalker",
            f"CopyTalker v{__version__}\n\n"
            "Real-time speech-to-speech translation system.\n\n"
            "Supports multiple languages and TTS engines.",
        )

    def _on_start(self):
        logger.info("Starting translation pipeline")
        if self._settings_dialog:
            self._settings_dialog.sync_to_state()
        self._pipeline_ctrl.start(self._state)
        self.statusBar().showMessage("Starting pipeline...")

    def _on_stop(self):
        logger.info("Stopping translation pipeline")
        self._pipeline_ctrl.stop()
        self.statusBar().showMessage("Stopping pipeline...")

    def _on_pipeline_started(self, capture_mode: str):
        self._state.isRunning = True
        self._conversation_view.set_pipeline_mode(capture_mode)
        self.statusBar().showMessage(f"Running ({capture_mode.upper()} mode)")
        if self._settings_dialog:
            self._settings_dialog.set_running_state(True)

    def _on_pipeline_stopped(self):
        self._state.isRunning = False
        self._stop_level_polling()
        self._conversation_view.set_idle()
        self.statusBar().showMessage("Stopped")
        if self._settings_dialog:
            self._settings_dialog.set_running_state(False)

    def _on_ptt_pressed(self):
        if self._state.isRunning and self._state.captureMode == "ptt":
            self._pipeline_ctrl.start_ptt_capture()

    def _on_ptt_released(self):
        if self._state.isRunning and self._state.captureMode == "ptt":
            self._pipeline_ctrl.stop_ptt_capture()

    def _on_ptt_recording_event(self, is_recording: bool):
        self._conversation_view.set_recording_state(is_recording)
        if is_recording:
            self._start_level_polling()
        else:
            self._stop_level_polling()

    def _on_synthesis_done(self):
        if self._state.captureMode == "ptt":
            self._conversation_view.ptt_bar.set_mode("ready")
        else:
            self._conversation_view.ptt_bar.set_mode("vad")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            focus = QApplication.focusWidget()
            if isinstance(focus, (QLineEdit, QTextEdit, QComboBox)):
                super().keyPressEvent(event)
                return
            if not self._space_down and self._state.isRunning:
                self._space_down = True
                self._on_ptt_pressed()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            focus = QApplication.focusWidget()
            if isinstance(focus, (QLineEdit, QTextEdit, QComboBox)):
                super().keyReleaseEvent(event)
                return
            if self._space_down:
                self._space_down = False
                self._on_ptt_released()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def closeEvent(self, event):
        self._level_timer.stop()
        self._event_timer.stop()
        if self._pipeline_ctrl.is_running:
            self._pipeline_ctrl.stop()
        event.accept()


def main():
    logging.basicConfig(level=logging.INFO)
    try:
        app = QApplication(sys.argv)
        app.setApplicationName("CopyTalker")
        app.setApplicationVersion(__version__)
        window = CopyTalkerApp()
        window.show()
        return app.exec()
    except Exception as e:
        logger.error(f"GUI error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
