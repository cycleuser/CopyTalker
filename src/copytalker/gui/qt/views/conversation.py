"""Qt conversation view for CopyTalker."""

from __future__ import annotations

import sys
from datetime import datetime

try:
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QScrollArea,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    print("PySide6 required")
    sys.exit(1)

from copytalker.core.i18n import I18n, UI_LANGUAGES
from copytalker.gui.qt.widgets.common import CardMessage
from copytalker.gui.qt.widgets.ptt_bar import PushToTalkBar


class ChatArea(QScrollArea):
    """Scrollable area containing card-style messages."""

    MAX_MESSAGES = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._msg_count = 0

        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setAlignment(Qt.AlignTop)
        self._layout.setSpacing(10)
        self._layout.setContentsMargins(20, 20, 20, 20)
        self._layout.addStretch()
        self.setWidget(self._container)

    def add_message(self, card: CardMessage):
        count = self._layout.count()
        if count > 0:
            self._layout.insertWidget(count - 1, card)
        self._msg_count += 1

        while self._msg_count > self.MAX_MESSAGES:
            item = self._layout.itemAt(0)
            if item and item.widget():
                widget = item.widget()
                self._layout.removeWidget(widget)
                widget.deleteLater()
                self._msg_count -= 1
            else:
                break

        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def clear(self):
        while self._msg_count > 0:
            item = self._layout.itemAt(0)
            if item and item.widget():
                widget = item.widget()
                self._layout.removeWidget(widget)
                widget.deleteLater()
                self._msg_count -= 1
            else:
                break


class QtConversationView(QWidget):
    ptt_pressed = Signal()
    ptt_released = Signal()
    start_requested = Signal()
    stop_requested = Signal()
    settings_requested = Signal()
    ui_language_changed = Signal(str)

    def __init__(self, parent=None, ui_lang: str = "en"):
        super().__init__(parent)
        self._i18n = I18n(ui_lang)
        self._ui_lang = ui_lang
        self._space_down = False
        self._chat_area = ChatArea()
        self._ptt_bar = PushToTalkBar()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._create_header())
        layout.addWidget(self._chat_area, stretch=1)
        layout.addWidget(self._ptt_bar)

        self.setFocusPolicy(Qt.StrongFocus)

    @property
    def ptt_bar(self) -> PushToTalkBar:
        return self._ptt_bar

    @property
    def i18n(self) -> I18n:
        return self._i18n

    def set_ui_language(self, lang: str):
        self._ui_lang = lang
        self._i18n.lang = lang
        self._update_ui_text()
        self.ui_language_changed.emit(lang)

    def _update_ui_text(self):
        t = self._i18n
        self._status_label.setText(t.ready)
        self._start_btn.setText(t.start)
        self._stop_btn.setText(t.stop)
        self._settings_btn.setText(t.settings)
        self._clear_btn.setText(t.clear)
        self._ptt_bar.update_text(t.configure, t.hold_space, t.listening)

    def _create_header(self):
        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(20, 16, 20, 16)
        header_layout.setSpacing(12)

        title_row = QHBoxLayout()
        title_row.setSpacing(16)

        title_label = QLabel("CopyTalker")
        title_font = title_label.font()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_row.addWidget(title_label)

        from copytalker import __version__

        version_label = QLabel(f"v{__version__}")
        version_label.setStyleSheet("color: gray;")
        title_row.addWidget(version_label)

        title_row.addStretch()

        ui_lang_label = QLabel("Interface:")
        title_row.addWidget(ui_lang_label)

        self._ui_lang_combo = QComboBox()
        self._ui_lang_combo.setMinimumWidth(100)
        for code, name in UI_LANGUAGES:
            self._ui_lang_combo.addItem(name, code)
        current_idx = next((i for i, (c, _) in enumerate(UI_LANGUAGES) if c == self._ui_lang), 0)
        self._ui_lang_combo.setCurrentIndex(current_idx)
        self._ui_lang_combo.currentIndexChanged.connect(self._on_ui_lang_changed)
        title_row.addWidget(self._ui_lang_combo)

        header_layout.addLayout(title_row)

        status_row = QHBoxLayout()
        status_row.setSpacing(12)

        self._status_label = QLabel(self._i18n.ready)
        status_font = self._status_label.font()
        status_font.setPointSize(12)
        status_font.setBold(True)
        self._status_label.setFont(status_font)
        status_row.addWidget(self._status_label)

        self._mode_label = QLabel("")
        status_row.addWidget(self._mode_label)
        status_row.addStretch()

        header_layout.addLayout(status_row)

        button_row = QHBoxLayout()
        button_row.setSpacing(12)

        self._start_btn = QPushButton(self._i18n.start)
        self._start_btn.setMinimumWidth(150)
        self._start_btn.setMinimumHeight(36)
        self._start_btn.clicked.connect(self.start_requested.emit)
        button_row.addWidget(self._start_btn)

        self._stop_btn = QPushButton(self._i18n.stop)
        self._stop_btn.setMinimumWidth(100)
        self._stop_btn.setMinimumHeight(36)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self.stop_requested.emit)
        button_row.addWidget(self._stop_btn)

        button_row.addStretch()

        self._settings_btn = QPushButton(self._i18n.settings)
        self._settings_btn.setMinimumWidth(100)
        self._settings_btn.setMinimumHeight(36)
        self._settings_btn.clicked.connect(self.settings_requested.emit)
        button_row.addWidget(self._settings_btn)

        self._clear_btn = QPushButton(self._i18n.clear)
        self._clear_btn.setMinimumWidth(80)
        self._clear_btn.setMinimumHeight(36)
        self._clear_btn.clicked.connect(self.clear_chat)
        button_row.addWidget(self._clear_btn)

        header_layout.addLayout(button_row)

        return header

    def _on_ui_lang_changed(self, index):
        lang = self._ui_lang_combo.itemData(index)
        self.set_ui_language(lang)

    def set_pipeline_mode(self, capture_mode: str):
        t = self._i18n
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        if capture_mode == "ptt":
            self._ptt_bar.set_mode("ready")
            self._mode_label.setText(f"[{t.ptt_mode}]")
        else:
            self._ptt_bar.set_mode("vad")
            self._mode_label.setText(f"[{t.continuous_mode}]")
        self._status_label.setText(t.running)

    def set_idle(self):
        t = self._i18n
        self._ptt_bar.set_mode("idle")
        self._mode_label.setText("")
        self._status_label.setText(t.ready)
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    def add_transcription(self, data):
        text = getattr(data, "text", str(data))
        lang = getattr(data, "language", "")
        lang_name = self._get_lang_display_name(lang)
        ts = datetime.now().strftime("%H:%M:%S")
        card = CardMessage(
            text, msg_type="original", language=lang_name, timestamp=ts, label=self._i18n.original
        )
        self._chat_area.add_message(card)

    def add_translation(self, data):
        text = getattr(data, "translated_text", str(data))
        ts = datetime.now().strftime("%H:%M:%S")
        card = CardMessage(text, msg_type="translated", timestamp=ts, label=self._i18n.translated)
        self._chat_area.add_message(card)

    def _get_lang_display_name(self, code: str) -> str:
        from copytalker.core.constants import LANGUAGE_NAMES

        return LANGUAGE_NAMES.get(code, code.upper()) if code else ""

    def clear_chat(self):
        self._chat_area.clear()

    def set_status(self, status_type: str, message: str):
        self._status_label.setText(message)

    def set_recording_state(self, is_recording: bool):
        t = self._i18n
        if is_recording:
            self._ptt_bar.set_mode("recording")
            self._status_label.setText(t.recording)
        else:
            self._ptt_bar.set_mode("ready")
            self._status_label.setText(t.running)

    def set_processing(self, is_processing: bool):
        t = self._i18n
        if is_processing:
            self._ptt_bar.set_mode("processing")
            self._status_label.setText(t.processing)
        else:
            self._ptt_bar.set_mode("ready")
            self._status_label.setText(t.running)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            focus = QApplication.focusWidget()
            if isinstance(focus, (QLineEdit, QTextEdit, QComboBox)):
                super().keyPressEvent(event)
                return
            if not self._space_down:
                self._space_down = True
                self.ptt_pressed.emit()
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
                self.ptt_released.emit()
            event.accept()
            return
        super().keyReleaseEvent(event)
