"""CopyTalkerApp – root orchestrator for the redesigned two-view GUI."""

from __future__ import annotations

import logging
import queue
import tkinter as tk
from tkinter import messagebox, ttk

from copytalker import __version__
from copytalker.gui.controllers.model_controller import ModelDownloadController
from copytalker.gui.controllers.pipeline_controller import PipelineController
from copytalker.gui.state import AppState
from copytalker.gui.views.conversation import ConversationView
from copytalker.gui.views.settings import SettingsView

logger = logging.getLogger(__name__)

# How often the main-thread polls the event queue (ms)
_POLL_MS = 100
# Minimum audio-level update interval (ms)
_LEVEL_POLL_MS = 50


class CopyTalkerApp:
    """
    Root orchestrator.

    * Creates AppState, controllers, and both views.
    * Switches between ConversationView and SettingsView via pack/pack_forget.
    * Polls the shared event queue and dispatches events to the active view.
    * Binds global <Space> key for push-to-talk with debounce.
    """

    WINDOW_TITLE = f"CopyTalker v{__version__}"
    WINDOW_SIZE = "800x700"
    MIN_WIDTH = 600
    MIN_HEIGHT = 500

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(self.WINDOW_TITLE)
        self.root.geometry(self.WINDOW_SIZE)
        self.root.minsize(self.MIN_WIDTH, self.MIN_HEIGHT)

        # Shared state + event queue --------------------------------
        self._state = AppState()
        self._event_queue: queue.Queue = queue.Queue()

        # Controllers -----------------------------------------------
        self._pipeline_ctrl = PipelineController(self._event_queue)
        self._model_ctrl = ModelDownloadController(self._event_queue)

        # Views (both are children of root, only one visible) -------
        self._conversation_view = ConversationView(
            self.root,
            on_settings=self.show_settings,
        )
        self._settings_view = SettingsView(
            self.root,
            state=self._state,
            event_queue=self._event_queue,
            model_ctrl=self._model_ctrl,
            on_back=self.show_conversation,
            on_start=self._on_start,
            on_stop=self._on_stop,
        )

        self._active_view: str = "settings"  # start in settings

        # Show settings first (user needs to configure before talking)
        self._settings_view.pack(fill=tk.BOTH, expand=True)

        # Key bindings (global) -------------------------------------
        self._space_down = False  # debounce flag
        self.root.bind("<KeyPress-space>", self._on_space_press)
        self.root.bind("<KeyRelease-space>", self._on_space_release)
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)

        # Audio-level polling id
        self._level_poll_id: str | None = None

        # Start event loop ------------------------------------------
        self._process_events()

    # ==================================================================
    # View switching
    # ==================================================================

    def show_conversation(self) -> None:
        """Switch to the conversation view."""
        self._settings_view.pack_forget()
        self._conversation_view.pack(fill=tk.BOTH, expand=True)
        self._active_view = "conversation"

    def show_settings(self) -> None:
        """Switch to the settings view."""
        self._conversation_view.pack_forget()
        self._settings_view.pack(fill=tk.BOTH, expand=True)
        self._settings_view.reset_scroll()
        self._active_view = "settings"

    # ==================================================================
    # Pipeline start / stop (callbacks from SettingsView)
    # ==================================================================

    def _on_start(self) -> None:
        """Start the translation pipeline with current settings."""
        self._settings_view.sync_to_state()
        self._pipeline_ctrl.start(self._state)

    def _on_stop(self) -> None:
        """Stop the translation pipeline."""
        self._pipeline_ctrl.stop()

    # ==================================================================
    # Push-to-talk key bindings
    # ==================================================================

    def _on_space_press(self, event: tk.Event) -> None:
        """Handle spacebar press – start PTT recording."""
        # Ignore if focus is on a text-entry widget
        focus = self.root.focus_get()
        if isinstance(focus, (tk.Entry, tk.Text, ttk.Entry)):
            return
        # Debounce: KeyPress fires repeatedly on most platforms
        if self._space_down:
            return
        if not self._pipeline_ctrl.is_running:
            return
        if self._state.capture_mode != "ptt":
            return

        self._space_down = True
        self._pipeline_ctrl.start_ptt_capture()

    def _on_space_release(self, event: tk.Event) -> None:
        """Handle spacebar release – stop PTT recording and inject audio."""
        focus = self.root.focus_get()
        if isinstance(focus, (tk.Entry, tk.Text, ttk.Entry)):
            return
        if not self._space_down:
            return

        self._space_down = False
        self._pipeline_ctrl.stop_ptt_capture()

    # ==================================================================
    # Audio-level polling (active during PTT recording)
    # ==================================================================

    def _start_level_polling(self) -> None:
        """Begin polling RMS level for the PTT bar meter."""
        self._stop_level_polling()
        self._poll_level()

    def _poll_level(self) -> None:
        """Read current RMS and update PTT bar."""
        if self._pipeline_ctrl.is_ptt_recording:
            level = self._pipeline_ctrl.get_ptt_rms_level()
            # Normalise: RMS level from VoiceRecorder is in 0..1 range
            self._conversation_view.ptt_bar.set_audio_level(level)
            self._level_poll_id = self.root.after(_LEVEL_POLL_MS, self._poll_level)
        else:
            self._level_poll_id = None

    def _stop_level_polling(self) -> None:
        """Cancel any pending level poll."""
        if self._level_poll_id is not None:
            self.root.after_cancel(self._level_poll_id)
            self._level_poll_id = None

    # ==================================================================
    # Event loop
    # ==================================================================

    def _process_events(self) -> None:
        """Drain the shared event queue and dispatch to views/controllers."""
        try:
            while True:
                event_type, data = self._event_queue.get_nowait()
                self._dispatch_event(event_type, data)
        except queue.Empty:
            pass

        # Schedule next tick
        self.root.after(_POLL_MS, self._process_events)

    def _dispatch_event(self, event_type: str, data) -> None:
        """Route a single event to the appropriate handler."""

        # -- Pipeline lifecycle events --
        if event_type == "started":
            capture_mode = data  # "ptt" or "vad"
            self._state.is_running = True
            self._conversation_view.set_status("running", "Running")
            self._conversation_view.set_pipeline_mode(capture_mode)
            # Auto-switch to conversation view when pipeline starts
            self.show_conversation()

        elif event_type == "stopped":
            self._state.is_running = False
            self._stop_level_polling()
            self._conversation_view.set_status("stopped", "Stopped")
            self._conversation_view.set_idle()
            self._settings_view.on_pipeline_stopped()

        # -- Transcription / Translation --
        elif event_type == "transcription":
            lang = getattr(data, "language", "")
            text = getattr(data, "text", str(data))
            self._conversation_view.add_transcription(text, lang)

        elif event_type == "translation":
            text = getattr(data, "translated_text", str(data))
            self._conversation_view.add_translation(text)

        # -- PTT events --
        elif event_type == "ptt_recording":
            is_recording = bool(data)
            self._conversation_view.set_recording_state(is_recording)
            if is_recording:
                self._start_level_polling()
            else:
                self._stop_level_polling()

        elif event_type == "ptt_processing":
            is_processing = bool(data)
            self._conversation_view.set_processing(is_processing)

        # -- Download progress --
        elif event_type == "dl_progress":
            self._settings_view.dl_progress_var.set(str(data))

        elif event_type == "download_done":
            self._settings_view.on_download_done()

        # -- Calibration --
        elif event_type == "calibration_done":
            self._settings_view.on_calibration_done(data)

        # -- Clone test --
        elif event_type == "clone_test_done":
            self._settings_view.on_clone_test_done()

        # -- Status / error --
        elif event_type == "status":
            msg = str(data)
            self._conversation_view.set_status("info", msg)

        elif event_type == "error":
            msg = str(data)
            self._conversation_view.set_status("error", msg)
            messagebox.showerror("Error", msg)

        # -- Synthesis done (reset processing indicator) --
        elif event_type == "synthesis":
            # After TTS finishes, reset PTT bar back to ready/vad
            if self._state.capture_mode == "ptt":
                self._conversation_view.ptt_bar.set_mode("ready")
            else:
                self._conversation_view.ptt_bar.set_mode("vad")

        else:
            logger.debug(f"Unhandled event: {event_type} -> {data}")

    # ==================================================================
    # Shutdown
    # ==================================================================

    def _on_quit(self) -> None:
        """Graceful shutdown."""
        self._stop_level_polling()
        if self._pipeline_ctrl.is_running:
            self._pipeline_ctrl.stop()
        self.root.quit()


# ======================================================================
# Entry point
# ======================================================================


def main() -> int:
    """Main entry point for the redesigned CopyTalker GUI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    try:
        root = tk.Tk()
        _app = CopyTalkerApp(root)
        root.mainloop()
        return 0
    except Exception as e:
        logger.error(f"GUI error: {e}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
