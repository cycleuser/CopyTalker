"""Custom Qt widgets for CopyTalker."""

from __future__ import annotations

try:
    from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt
    from PySide6.QtGui import QFont, QPalette
    from PySide6.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QLabel,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError:

    class QWidget:
        def __init__(self, parent=None):
            pass

    class QLabel:
        def __init__(self, text="", parent=None):
            pass

    class QToolButton:
        def __init__(self, parent=None):
            pass

    class QFont:
        def __init__(self, family="", pointSize=-1):
            pass

    class QPalette:
        Base = 0
        WindowText = 0

    class QPropertyAnimation:
        def __init__(self, *args):
            pass

    class QEasingCurve:
        InOutQuad = 0

    class QVBoxLayout:
        def __init__(self, *args):
            pass

    class QHBoxLayout:
        def __init__(self, *args):
            pass

    class QFrame:
        def __init__(self, parent=None):
            pass

        HLine = None
        Raised = None


class CollapsibleSection(QWidget):
    """A section with a clickable header that expands/collapses its content area."""

    _QWIDGETSIZE_MAX = 16777215

    def __init__(self, title: str, expanded: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._expanded = expanded

        self._toggle_button = QToolButton()
        self._toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._toggle_button.setCheckable(True)
        self._toggle_button.setChecked(expanded)
        self._toggle_button.toggled.connect(self._on_toggle)

        self._title_label = QLabel(title)
        title_font = self._title_label.font()
        title_font.setBold(True)
        self._title_label.setFont(title_font)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)
        header_layout.addWidget(self._toggle_button)
        header_layout.addWidget(self._title_label)
        header_layout.addStretch()

        self._content = QWidget()
        self._content_layout = QVBoxLayout()
        self._content_layout.setContentsMargins(24, 4, 0, 4)
        self._content.setLayout(self._content_layout)

        self._animation = QPropertyAnimation(self._content, b"maximumHeight")
        self._animation.setDuration(200)
        self._animation.setEasingCurve(QEasingCurve.InOutQuad)
        self._animation.finished.connect(self._on_animation_finished)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addLayout(header_layout)
        main_layout.addWidget(self._content)
        self.setLayout(main_layout)

        if expanded:
            self._content.setMaximumHeight(self._QWIDGETSIZE_MAX)
        else:
            self._content.setMaximumHeight(0)

    @property
    def content_layout(self) -> QVBoxLayout:
        """Return the layout of the content area for adding child widgets."""
        return self._content_layout

    def set_expanded(self, expanded: bool) -> None:
        """Programmatically expand or collapse (no animation)."""
        self._expanded = expanded
        self._toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._toggle_button.blockSignals(True)
        self._toggle_button.setChecked(expanded)
        self._toggle_button.blockSignals(False)
        if expanded:
            self._content.setMaximumHeight(self._QWIDGETSIZE_MAX)
        else:
            self._content.setMaximumHeight(0)

    def _on_toggle(self, checked: bool) -> None:
        self._expanded = checked
        self._toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        target_height = self._content_layout.sizeHint().height() + 8
        if checked:
            self._animation.setStartValue(0)
            self._animation.setEndValue(target_height)
        else:
            self._animation.setStartValue(self._content.height())
            self._animation.setEndValue(0)
        self._animation.start()

    def _on_animation_finished(self) -> None:
        if self._expanded:
            self._content.setMaximumHeight(self._QWIDGETSIZE_MAX)


class CardMessage(QFrame):
    """Card-style chat message with a subtle frame."""

    def __init__(
        self,
        text: str,
        msg_type: str = "original",
        language: str = "",
        timestamp: str = "",
        label: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._msg_type = msg_type
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setLineWidth(1)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        header_row = QHBoxLayout()

        if label:
            type_label = QLabel(label)
            type_font = type_label.font()
            type_font.setPointSize(9)
            type_font.setBold(True)
            type_label.setFont(type_font)
            if msg_type == "translated":
                type_label.setStyleSheet("color: #2e7d32;")
            else:
                type_label.setStyleSheet("color: #1565c0;")
            header_row.addWidget(type_label)

        if language:
            lang_label = QLabel(f"[{language}]")
            lang_font = lang_label.font()
            lang_font.setPointSize(9)
            lang_label.setFont(lang_font)
            lang_label.setForegroundRole(QPalette.ColorRole.PlaceholderText)
            header_row.addWidget(lang_label)

        header_row.addStretch()

        if timestamp:
            ts_label = QLabel(timestamp)
            ts_font = ts_label.font()
            ts_font.setPointSize(8)
            ts_label.setFont(ts_font)
            ts_label.setForegroundRole(QPalette.ColorRole.PlaceholderText)
            header_row.addWidget(ts_label)

        layout.addLayout(header_row)

        text_label = QLabel(text)
        text_font = text_label.font()
        text_font.setPointSize(11)
        text_label.setFont(text_font)
        text_label.setWordWrap(True)
        text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(text_label)

        self.setLayout(layout)
        self.setMinimumHeight(50)
