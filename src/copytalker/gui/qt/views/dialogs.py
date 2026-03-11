"""Qt dialogs for CopyTalker."""

from __future__ import annotations

import sys

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QCheckBox,
        QDialog,
        QHBoxLayout,
        QPushButton,
        QScrollArea,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    print("PySide6 required")
    sys.exit(1)

from copytalker.core.constants import SUPPORTED_LANGUAGES


class TranslationLanguageDialog(QDialog):
    """Dialog for selecting languages to download translation models for."""

    _DEFAULT_CHECKED = {"zh", "ja", "es", "fr", "de"}
    _EAST_ASIAN = {"zh", "ja", "ko"}
    _EUROPEAN = {"es", "fr", "de", "ru"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Translation Languages")
        self.setMinimumWidth(400)
        self.setMinimumHeight(350)

        self._checkboxes: dict[str, QCheckBox] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Quick-select buttons
        btn_row = QHBoxLayout()
        for label, action in [
            ("Select All", self._select_all),
            ("Deselect All", self._deselect_all),
            ("East Asian", self._select_east_asian),
            ("European", self._select_european),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(action)
            btn_row.addWidget(btn)
        layout.addLayout(btn_row)

        # Scrollable checkbox list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        container = QWidget()
        cb_layout = QVBoxLayout(container)

        for code, name in SUPPORTED_LANGUAGES:
            if code == "en":
                continue
            cb = QCheckBox(f"{name} ({code})")
            cb.setChecked(code in self._DEFAULT_CHECKED)
            self._checkboxes[code] = cb
            cb_layout.addWidget(cb)

        cb_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll, stretch=1)

        # OK / Cancel
        btn_row2 = QHBoxLayout()
        btn_row2.addStretch()
        dl_btn = QPushButton("Download")
        dl_btn.clicked.connect(self.accept)
        btn_row2.addWidget(dl_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row2.addWidget(cancel_btn)
        layout.addLayout(btn_row2)

    def selected_languages(self) -> list[str]:
        """Return list of checked language codes."""
        return [code for code, cb in self._checkboxes.items() if cb.isChecked()]

    def _select_all(self):
        for cb in self._checkboxes.values():
            cb.setChecked(True)

    def _deselect_all(self):
        for cb in self._checkboxes.values():
            cb.setChecked(False)

    def _select_east_asian(self):
        for code, cb in self._checkboxes.items():
            cb.setChecked(code in self._EAST_ASIAN)

    def _select_european(self):
        for code, cb in self._checkboxes.items():
            cb.setChecked(code in self._EUROPEAN)
