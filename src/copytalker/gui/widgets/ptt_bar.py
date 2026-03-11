"""Push-to-Talk bar widget for the conversation view."""

from __future__ import annotations

import tkinter as tk
from typing import Literal


class PushToTalkBar(tk.Frame):
    """
    A prominent status bar at the bottom of the conversation view.

    Shows the current input mode state and provides visual
    feedback during PTT recording.
    """

    MODES = {
        "idle": {
            "bg": "#ECEFF1",
            "fg": "#78909C",
            "text": "Configure settings to begin",
        },
        "ready": {
            "bg": "#E8EAF6",
            "fg": "#283593",
            "text": "Hold [Space] to Talk",
        },
        "recording": {
            "bg": "#FFCDD2",
            "fg": "#B71C1C",
            "text": "Recording...  Release [Space] to translate",
        },
        "processing": {
            "bg": "#FFF9C4",
            "fg": "#F57F17",
            "text": "Processing...",
        },
        "vad": {
            "bg": "#E8F5E9",
            "fg": "#2E7D32",
            "text": "Listening... (continuous mode)",
        },
    }

    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, bg=self.MODES["idle"]["bg"], height=64, **kwargs)
        self.pack_propagate(False)  # Fixed height

        self._mode = "idle"
        bg = self.MODES["idle"]["bg"]

        # Recording indicator dot
        self._dot_canvas = tk.Canvas(
            self,
            width=18,
            height=18,
            bg=bg,
            highlightthickness=0,
        )
        self._dot = self._dot_canvas.create_oval(3, 3, 15, 15, fill="#B0BEC5", outline="")
        self._dot_canvas.pack(side=tk.LEFT, padx=(20, 10))

        # Status text
        self._label = tk.Label(
            self,
            text=self.MODES["idle"]["text"],
            font=("Helvetica", 14),
            fg=self.MODES["idle"]["fg"],
            bg=bg,
            anchor="w",
        )
        self._label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Level meter (only visible during recording)
        self._level_canvas = tk.Canvas(
            self,
            width=100,
            height=14,
            bg=bg,
            highlightthickness=0,
        )
        self._level_bar = self._level_canvas.create_rectangle(
            0, 0, 0, 14, fill="#4CAF50", outline=""
        )
        self._level_canvas.pack(side=tk.RIGHT, padx=(0, 20))
        self._level_canvas.pack_forget()  # Hidden by default

    def set_mode(self, mode: Literal["idle", "ready", "recording", "processing", "vad"]) -> None:
        """Update the PTT bar to reflect the current state."""
        if mode not in self.MODES:
            return

        self._mode = mode
        style = self.MODES[mode]

        self.configure(bg=style["bg"])
        self._label.configure(text=style["text"], fg=style["fg"], bg=style["bg"])
        self._dot_canvas.configure(bg=style["bg"])

        dot_colors = {
            "idle": "#B0BEC5",
            "ready": "#5C6BC0",
            "recording": "#D32F2F",
            "processing": "#FFA000",
            "vad": "#43A047",
        }
        self._dot_canvas.itemconfig(self._dot, fill=dot_colors.get(mode, "#B0BEC5"))

        if mode == "recording":
            self._level_canvas.configure(bg=style["bg"])
            self._level_canvas.pack(side=tk.RIGHT, padx=(0, 20))
        else:
            self._level_canvas.pack_forget()

    def set_audio_level(self, level: float) -> None:
        """Update the audio level meter (0.0 - 1.0)."""
        level = max(0.0, min(1.0, level))
        bar_width = int(100 * level)

        if level > 0.8:
            color = "#f44336"
        elif level > 0.5:
            color = "#FF9800"
        else:
            color = "#4CAF50"

        self._level_canvas.coords(self._level_bar, 0, 0, bar_width, 14)
        self._level_canvas.itemconfig(self._level_bar, fill=color)

    @property
    def mode(self) -> str:
        """Current mode."""
        return self._mode
