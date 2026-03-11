"""Conversation view with chat bubbles and push-to-talk bar."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from copytalker.gui.widgets.chat_bubble import ChatCanvas
from copytalker.gui.widgets.common import StatusIndicator
from copytalker.gui.widgets.ptt_bar import PushToTalkBar


class ConversationView(ttk.Frame):
    """
    Main conversation view showing chat-bubble style transcription/translation
    and a push-to-talk bar at the bottom.
    """

    def __init__(
        self,
        parent: tk.Widget,
        on_settings: Callable[[], None],
        **kwargs,
    ):
        super().__init__(parent, **kwargs)

        self._on_settings = on_settings

        self._build_top_bar()
        self._build_chat_area()
        self._build_ptt_bar()

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_top_bar(self) -> None:
        """Build the top bar with status indicator, title, and gear button."""
        bar = ttk.Frame(self)
        bar.pack(fill=tk.X, padx=16, pady=(12, 0))

        # Status indicator (left)
        self.status_indicator = StatusIndicator(bar)
        self.status_indicator.pack(side=tk.LEFT)

        # Gear button (right)
        gear_btn = ttk.Button(
            bar,
            text="\u2699 Settings",
            width=10,
            command=self._on_settings,
        )
        gear_btn.pack(side=tk.RIGHT)

        # Title (center)
        title = ttk.Label(
            bar,
            text="CopyTalker",
            font=("Helvetica", 16, "bold"),
            anchor="center",
        )
        title.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        # Thin separator
        sep = ttk.Separator(self, orient="horizontal")
        sep.pack(fill=tk.X, padx=16, pady=(8, 0))

    def _build_chat_area(self) -> None:
        """Build the scrollable chat canvas."""
        self.chat_canvas = ChatCanvas(self)
        self.chat_canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

    def _build_ptt_bar(self) -> None:
        """Build the push-to-talk status bar at the bottom."""
        self.ptt_bar = PushToTalkBar(self)
        self.ptt_bar.pack(fill=tk.X, padx=12, pady=(0, 12))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_transcription(self, text: str, language: str = "") -> None:
        """Add a transcription bubble (left side)."""
        label = f"[{language}]" if language else ""
        self.chat_canvas.add_bubble(text=text, side="left", label=label)

    def add_translation(self, text: str) -> None:
        """Add a translation bubble (right side)."""
        self.chat_canvas.add_bubble(text=text, side="right")

    def set_status(self, status: str, text: str | None = None) -> None:
        """Update the status indicator."""
        self.status_indicator.set_status(status, text)

    def set_recording_state(self, is_recording: bool) -> None:
        """Update PTT bar to reflect recording state."""
        if is_recording:
            self.ptt_bar.set_mode("recording")
        else:
            self.ptt_bar.set_mode("ready")

    def set_processing(self, is_processing: bool) -> None:
        """Update PTT bar to show processing state."""
        if is_processing:
            self.ptt_bar.set_mode("processing")
        else:
            self.ptt_bar.set_mode("ready")

    def set_pipeline_mode(self, capture_mode: str) -> None:
        """Set PTT bar mode based on pipeline capture mode."""
        if capture_mode == "vad":
            self.ptt_bar.set_mode("vad")
        else:
            self.ptt_bar.set_mode("ready")

    def set_idle(self) -> None:
        """Set PTT bar to idle (pipeline not running)."""
        self.ptt_bar.set_mode("idle")

    def clear_chat(self) -> None:
        """Remove all chat bubbles."""
        self.chat_canvas.clear()
