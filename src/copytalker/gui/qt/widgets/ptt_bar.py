"""Push-to-talk bar widget for CopyTalker."""

from __future__ import annotations

try:
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QWidget
except ImportError:

    class QWidget:
        def __init__(self, parent=None):
            pass

    class QFont:
        def __init__(self, family="", pointSize=-1):
            pass

    class QTimer:
        def __init__(self):
            pass

        def start(self, interval):
            pass

        def stop(self):
            pass

    class QHBoxLayout:
        def __init__(self, *args):
            pass

    class QLabel:
        def __init__(self, text="", parent=None):
            pass

    class QProgressBar:
        def __init__(self, parent=None):
            pass

    class QPalette:
        pass


class PushToTalkBar(QWidget):
    """Qt widget showing PTT status and audio levels."""

    HEIGHT = 60

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedHeight(self.HEIGHT)
        self._mode = "idle"
        self._audio_level = 0.0
        self._level_timer = QTimer()
        self._level_timer.timeout.connect(self._update_level_display)

        self._texts = {
            "idle": "Configure settings to begin",
            "ready": "Hold [Space] to Talk",
            "recording": "Recording... Release [Space] to translate",
            "processing": "Processing...",
            "vad": "Listening... (continuous mode)",
        }
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(12)

        self._status_label = QLabel(self._texts["idle"])
        status_font = self._status_label.font()
        status_font.setPointSize(11)
        self._status_label.setFont(status_font)
        layout.addWidget(self._status_label)

        layout.addStretch()

        self._level_bar = QProgressBar()
        self._level_bar.setFixedWidth(150)
        self._level_bar.setFixedHeight(16)
        self._level_bar.setRange(0, 100)
        self._level_bar.setValue(0)
        self._level_bar.setTextVisible(False)
        self._level_bar.hide()
        layout.addWidget(self._level_bar)

    def update_text(self, idle: str, ready: str, vad: str):
        """Update the text for different modes."""
        self._texts["idle"] = idle
        self._texts["ready"] = ready
        self._texts["vad"] = vad
        if self._mode in self._texts:
            self._status_label.setText(self._texts[self._mode])

    def set_mode(self, mode: str) -> None:
        """Set the PTT mode."""
        self._mode = mode
        if mode in self._texts:
            self._status_label.setText(self._texts[mode])

        if mode == "recording":
            self._level_bar.show()
        else:
            self._level_bar.hide()
            self._level_bar.setValue(0)

    def set_audio_level(self, level: float) -> None:
        """Set the current audio level (0.0 to 1.0)."""
        self._audio_level = max(0.0, min(1.0, level))
        self._level_bar.setValue(int(self._audio_level * 100))
        if level > 0:
            self._level_timer.start(50)
        else:
            self._level_timer.stop()

    def _update_level_display(self) -> None:
        """Update level display animation."""
        self._level_bar.setValue(int(self._audio_level * 100))
