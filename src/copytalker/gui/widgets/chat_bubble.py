"""Chat bubble widgets for the conversation view."""

from __future__ import annotations

import sys
import time
import tkinter as tk
from tkinter import ttk
from typing import Literal

_IS_MACOS = sys.platform == "darwin"


class ChatBubble(tk.Frame):
    """A single chat bubble displaying text with a timestamp."""

    STYLES = {
        "left": {
            "bg": "#E3F2FD",
            "fg": "#1565C0",
            "text_fg": "#212121",
            "anchor": "w",
            "margin_near": 16,
            "margin_far": 100,
        },
        "right": {
            "bg": "#E8F5E9",
            "fg": "#2E7D32",
            "text_fg": "#1B5E20",
            "anchor": "e",
            "margin_near": 100,
            "margin_far": 16,
        },
    }

    def __init__(
        self,
        parent: tk.Widget,
        text: str,
        side: Literal["left", "right"] = "left",
        label: str = "",
        timestamp: str = "",
        **kwargs,
    ):
        parent_bg = parent.cget("bg")
        super().__init__(parent, bg=parent_bg, **kwargs)
        style = self.STYLES[side]

        padx_left = style["margin_near"] if side == "left" else style["margin_far"]
        padx_right = style["margin_far"] if side == "left" else style["margin_near"]

        # Outer container for alignment
        outer = tk.Frame(self, bg=parent_bg)
        outer.pack(fill=tk.X, padx=(padx_left, padx_right), pady=(3, 3))

        # Bubble frame with colored background
        bubble = tk.Frame(outer, bg=style["bg"], padx=14, pady=10)
        bubble.pack(anchor=style["anchor"])

        # Optional label (e.g. language code)
        if label:
            lbl = tk.Label(
                bubble,
                text=label,
                font=("Helvetica", 9, "bold"),
                fg=style["fg"],
                bg=style["bg"],
                anchor="w",
            )
            lbl.pack(fill=tk.X, anchor="w")

        # Message text
        self._msg = tk.Label(
            bubble,
            text=text,
            font=("Helvetica", 13),
            fg=style["text_fg"],
            bg=style["bg"],
            wraplength=450,
            justify=tk.LEFT,
            anchor="w",
        )
        self._msg.pack(fill=tk.X, anchor="w", pady=(2, 0))

        # Timestamp
        if timestamp:
            ts = tk.Label(
                bubble,
                text=timestamp,
                font=("Helvetica", 8),
                fg="#9E9E9E",
                bg=style["bg"],
                anchor="e",
            )
            ts.pack(fill=tk.X, anchor="e", pady=(4, 0))


class ChatCanvas(tk.Frame):
    """A scrollable canvas container for chat bubbles."""

    MAX_BUBBLES = 200

    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, **kwargs)

        self._bubble_count = 0

        # Canvas + Scrollbar
        self._canvas = tk.Canvas(
            self,
            bg="#FAFAFA",
            highlightthickness=0,
            borderwidth=0,
        )
        self._scrollbar = ttk.Scrollbar(
            self,
            orient=tk.VERTICAL,
            command=self._canvas.yview,
        )
        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Inner frame that holds the chat bubbles
        self._inner = tk.Frame(self._canvas, bg="#FAFAFA")
        self._inner_id = self._canvas.create_window(
            (0, 0),
            window=self._inner,
            anchor="nw",
        )

        # Bind resize/scroll events
        self._inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        # Mousewheel: Enter/Leave ownership pattern (no global bind_all)
        self._canvas.bind("<Enter>", self._on_enter)
        self._canvas.bind("<Leave>", self._on_leave)

    # ------------------------------------------------------------------
    # Mousewheel handling
    # ------------------------------------------------------------------

    def _on_enter(self, event) -> None:
        """Take mousewheel ownership when mouse enters chat area."""
        if _IS_MACOS:
            self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        else:
            self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)
            self._canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
            self._canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _on_leave(self, event) -> None:
        """Release mousewheel ownership when mouse leaves chat area."""
        self._canvas.unbind_all("<MouseWheel>")
        if not _IS_MACOS:
            self._canvas.unbind_all("<Button-4>")
            self._canvas.unbind_all("<Button-5>")

    def _on_inner_configure(self, event):
        """Update scroll region when inner frame changes size."""
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        """Match inner frame width to canvas width."""
        self._canvas.itemconfig(self._inner_id, width=event.width)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling (macOS/Windows)."""
        if _IS_MACOS:
            self._canvas.yview_scroll(-event.delta, "units")
        else:
            self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event):
        """Handle mouse wheel scrolling (Linux)."""
        if event.num == 4:
            self._canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self._canvas.yview_scroll(1, "units")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_bubble(
        self,
        text: str,
        side: Literal["left", "right"] = "left",
        label: str = "",
        timestamp: str = "",
    ) -> None:
        """Add a new chat bubble and auto-scroll to bottom."""
        if not timestamp:
            timestamp = time.strftime("%H:%M:%S")

        bubble = ChatBubble(
            self._inner,
            text=text,
            side=side,
            label=label,
            timestamp=timestamp,
        )
        bubble.pack(fill=tk.X, anchor="w")

        self._bubble_count += 1

        # Remove oldest bubbles if we exceed the limit
        if self._bubble_count > self.MAX_BUBBLES:
            children = self._inner.winfo_children()
            if children:
                children[0].destroy()
                self._bubble_count -= 1

        # Auto-scroll to bottom
        self._canvas.update_idletasks()
        self._canvas.yview_moveto(1.0)

    def clear(self) -> None:
        """Remove all chat bubbles."""
        for child in self._inner.winfo_children():
            child.destroy()
        self._bubble_count = 0

    def destroy(self):
        """Clean up global bindings before destroying."""
        try:
            self._canvas.unbind_all("<MouseWheel>")
            self._canvas.unbind_all("<Button-4>")
            self._canvas.unbind_all("<Button-5>")
        except tk.TclError:
            pass
        super().destroy()
