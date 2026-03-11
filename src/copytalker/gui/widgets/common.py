"""
Custom Tkinter widgets for CopyTalker GUI.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional


class StatusIndicator(ttk.Frame):
    """
    A status indicator widget with icon and text.
    """

    COLORS = {
        "ready": "#4CAF50",  # Green
        "running": "#2196F3",  # Blue
        "error": "#f44336",  # Red
        "warning": "#FF9800",  # Orange
        "stopped": "#9E9E9E",  # Gray
    }

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # Status indicator dot
        self.canvas = tk.Canvas(self, width=16, height=16, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, padx=(0, 5))

        self._indicator = self.canvas.create_oval(4, 4, 12, 12, fill=self.COLORS["stopped"])

        # Status text
        self.label = ttk.Label(self, text="Ready")
        self.label.pack(side=tk.LEFT)

        self._current_status = "stopped"

    def set_status(self, status: str, text: Optional[str] = None) -> None:
        """
        Set the status.

        Args:
            status: Status type ('ready', 'running', 'error', 'warning', 'stopped')
            text: Optional status text
        """
        color = self.COLORS.get(status, self.COLORS["stopped"])
        self.canvas.itemconfig(self._indicator, fill=color)

        if text:
            self.label.config(text=text)

        self._current_status = status

    @property
    def status(self) -> str:
        """Get current status."""
        return self._current_status


class LevelMeter(ttk.Frame):
    """
    An audio level meter widget.
    """

    def __init__(self, parent, width: int = 200, height: int = 20, **kwargs):
        super().__init__(parent, **kwargs)

        self._width = width
        self._height = height

        self.canvas = tk.Canvas(
            self,
            width=width,
            height=height,
            highlightthickness=1,
            highlightbackground="#ccc",
        )
        self.canvas.pack()

        # Background
        self.canvas.create_rectangle(
            0,
            0,
            width,
            height,
            fill="#f0f0f0",
            outline="",
        )

        # Level bar
        self._bar = self.canvas.create_rectangle(
            0,
            0,
            0,
            height,
            fill="#4CAF50",
            outline="",
        )

        self._level = 0.0

    def set_level(self, level: float) -> None:
        """
        Set the audio level (0.0 - 1.0).

        Args:
            level: Audio level between 0.0 and 1.0
        """
        level = max(0.0, min(1.0, level))
        self._level = level

        # Calculate bar width
        bar_width = int(self._width * level)

        # Update color based on level
        if level > 0.8:
            color = "#f44336"  # Red
        elif level > 0.5:
            color = "#FF9800"  # Orange
        else:
            color = "#4CAF50"  # Green

        self.canvas.coords(self._bar, 0, 0, bar_width, self._height)
        self.canvas.itemconfig(self._bar, fill=color)

    @property
    def level(self) -> float:
        """Get current level."""
        return self._level


class TooltipLabel(ttk.Label):
    """
    Label with tooltip on hover.
    """

    def __init__(self, parent, tooltip_text: str = "", **kwargs):
        super().__init__(parent, **kwargs)

        self._tooltip_text = tooltip_text
        self._tooltip_window: Optional[tk.Toplevel] = None

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _on_enter(self, event) -> None:
        """Show tooltip on mouse enter."""
        if not self._tooltip_text:
            return

        x = event.x_root + 10
        y = event.y_root + 10

        self._tooltip_window = tw = tk.Toplevel(self)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = ttk.Label(
            tw,
            text=self._tooltip_text,
            padding=(5, 2),
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
        )
        label.pack()

    def _on_leave(self, event) -> None:
        """Hide tooltip on mouse leave."""
        if self._tooltip_window:
            self._tooltip_window.destroy()
            self._tooltip_window = None

    def set_tooltip(self, text: str) -> None:
        """Set tooltip text."""
        self._tooltip_text = text
