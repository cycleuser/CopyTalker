"""Settings view – single scrollable page with all configuration sections."""

from __future__ import annotations

import logging
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from copytalker.core.constants import (
    SUPPORTED_LANGUAGES,
    get_available_voices,
)
from copytalker.gui.controllers.model_controller import ModelDownloadController
from copytalker.gui.state import AppState

logger = logging.getLogger(__name__)

_IS_MACOS = sys.platform == "darwin"

EMOTIONS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "surprised",
    "whisper",
    "excited",
    "calm",
    "serious",
    "gentle",
    "contemptuous",
    "disgusted",
]

# Uniform layout constants
_SECTION_PAD = (0, 10)  # vertical pad between LabelFrame sections
_SECTION_INNER_PAD = 12  # padding inside each LabelFrame
_ROW_PAD = 5  # vertical pad between rows inside a section
_LABEL_WIDTH = 16  # consistent label column width


class SettingsView(ttk.Frame):
    """Settings view – single scrollable page, no tabs."""

    def __init__(
        self,
        parent: tk.Widget,
        state: AppState,
        event_queue: queue.Queue,
        model_ctrl: ModelDownloadController,
        on_back: Callable[[], None],
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        **kwargs,
    ):
        super().__init__(parent, **kwargs)

        self._state = state
        self._event_queue = event_queue
        self._model_ctrl = model_ctrl
        self._on_back = on_back
        self._on_start = on_start
        self._on_stop = on_stop

        # --- Tk variables ---
        self.source_var = tk.StringVar(value="Auto-detect")
        self.target_var = tk.StringVar()
        self.engine_var = tk.StringVar(value=state.tts_engine)
        self.voice_var = tk.StringVar(value=state.voice)
        self.capture_mode_var = tk.StringVar(value=state.capture_mode)
        self.ref_audio_var = tk.StringVar(value=state.ref_audio_path)
        self.emotion_var = tk.StringVar(value=state.emotion)
        self.trans_model_var = tk.StringVar(value=state.translation_model)
        self.trans_device_var = tk.StringVar(value=state.trans_device)
        self.tts_device_var = tk.StringVar(value=state.tts_device)
        self.noise_level_var = tk.StringVar(value="Not calibrated")
        self.dl_progress_var = tk.StringVar(value="")
        self.cache_info_var = tk.StringVar(value="")

        # Recording state
        self._recorder = None
        self._rec_timer_id = None
        self.rec_timer_var = tk.StringVar(value="0.0s")

        # Clone testing
        self.clone_engine_var = tk.StringVar(value="indextts")
        self.clone_lang_var = tk.StringVar(value="en")
        self.clone_text_var = tk.StringVar(value="Hello, this is a voice clone test.")
        self._saved_clone_var = tk.StringVar()
        self._upload_path_var = tk.StringVar()

        self._build_ui()
        self._set_default_target()

    # ==================================================================
    # UI Construction
    # ==================================================================

    def _build_ui(self) -> None:
        # ---- Fixed header ----
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=16, pady=(12, 0))

        ttk.Button(header, text="\u2190 Back", width=8, command=self._on_back).pack(side=tk.LEFT)
        ttk.Label(header, text="Settings", font=("Helvetica", 16, "bold")).pack(
            side=tk.LEFT, padx=16
        )

        ttk.Separator(self, orient="horizontal").pack(fill=tk.X, padx=16, pady=(8, 0))

        # ---- Scrollable content area ----
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=16, pady=4)

        self._canvas = tk.Canvas(container, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Inner frame for all sections
        self._inner = ttk.Frame(self._canvas, padding=(4, 12, 4, 12))
        self._inner_id = self._canvas.create_window((0, 0), window=self._inner, anchor="nw")

        self._inner.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas.bind(
            "<Configure>",
            lambda e: self._canvas.itemconfig(self._inner_id, width=e.width),
        )

        # Mousewheel: Enter/Leave ownership pattern
        self._canvas.bind("<Enter>", self._on_enter_canvas)
        self._canvas.bind("<Leave>", self._on_leave_canvas)

        # ---- Build all sections sequentially ----
        p = self._inner
        self._build_languages_section(p)
        self._build_input_mode_section(p)
        self._build_tts_section(p)
        self._build_ref_audio_section(p)
        self._build_record_section(p)
        self._build_clone_test_section(p)
        self._build_translation_section(p)
        self._build_tts_device_section(p)
        self._build_calibration_section(p)
        self._build_downloads_section(p)

        # ---- Fixed bottom action bar ----
        ttk.Separator(self, orient="horizontal").pack(fill=tk.X, padx=16, pady=(0, 0))

        action = ttk.Frame(self)
        action.pack(fill=tk.X, padx=16, pady=(8, 12))

        self.start_btn = ttk.Button(
            action, text="\u25b6  Start Translation", command=self._handle_start
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 12))

        self.stop_btn = ttk.Button(
            action, text="\u25a0  Stop", command=self._handle_stop, state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT)

        self.status_label = ttk.Label(action, text="", foreground="gray")
        self.status_label.pack(side=tk.RIGHT)

    # ------------------------------------------------------------------
    # Mousewheel handling (Enter/Leave ownership)
    # ------------------------------------------------------------------

    def _on_enter_canvas(self, event) -> None:
        if _IS_MACOS:
            self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        else:
            self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)
            self._canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
            self._canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _on_leave_canvas(self, event) -> None:
        self._canvas.unbind_all("<MouseWheel>")
        if not _IS_MACOS:
            self._canvas.unbind_all("<Button-4>")
            self._canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event) -> None:
        if _IS_MACOS:
            self._canvas.yview_scroll(-event.delta, "units")
        else:
            self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event) -> None:
        if event.num == 4:
            self._canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self._canvas.yview_scroll(1, "units")

    def reset_scroll(self) -> None:
        """Scroll back to top. Called from app.py on view switch."""
        self._canvas.yview_moveto(0)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_languages_section(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="Languages", padding=_SECTION_INNER_PAD)
        frame.pack(fill=tk.X, pady=_SECTION_PAD)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=_ROW_PAD)
        ttk.Label(row, text="Source language:", width=_LABEL_WIDTH).pack(side=tk.LEFT)
        lang_names = ["Auto-detect"] + [f"{n} ({c})" for c, n in SUPPORTED_LANGUAGES]
        self.source_combo = ttk.Combobox(
            row, textvariable=self.source_var, values=lang_names, state="readonly"
        )
        self.source_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=_ROW_PAD)
        ttk.Label(row, text="Target language:", width=_LABEL_WIDTH).pack(side=tk.LEFT)
        target_names = [f"{n} ({c})" for c, n in SUPPORTED_LANGUAGES]
        self.target_combo = ttk.Combobox(
            row, textvariable=self.target_var, values=target_names, state="readonly"
        )
        self.target_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.target_combo.bind("<<ComboboxSelected>>", self._on_target_changed)

    def _build_input_mode_section(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="Input Mode", padding=_SECTION_INNER_PAD)
        frame.pack(fill=tk.X, pady=_SECTION_PAD)

        ttk.Radiobutton(
            frame,
            text="Push-to-Talk  (hold Space to record, release to translate)",
            variable=self.capture_mode_var,
            value="ptt",
        ).pack(anchor="w", pady=_ROW_PAD)
        ttk.Radiobutton(
            frame,
            text="Continuous  (auto-detect voice activity)",
            variable=self.capture_mode_var,
            value="vad",
        ).pack(anchor="w", pady=_ROW_PAD)

    def _build_tts_section(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="Text-to-Speech", padding=_SECTION_INNER_PAD)
        frame.pack(fill=tk.X, pady=_SECTION_PAD)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=_ROW_PAD)
        ttk.Label(row, text="Engine:", width=_LABEL_WIDTH).pack(side=tk.LEFT)
        engines = ["auto", "kokoro", "edge-tts", "pyttsx3", "indextts", "fish-speech"]
        self.engine_combo = ttk.Combobox(
            row, textvariable=self.engine_var, values=engines, state="readonly"
        )
        self.engine_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.engine_combo.bind("<<ComboboxSelected>>", self._on_engine_changed)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=_ROW_PAD)
        ttk.Label(row, text="Voice:", width=_LABEL_WIDTH).pack(side=tk.LEFT)
        self.voice_combo = ttk.Combobox(row, textvariable=self.voice_var, state="readonly")
        self.voice_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Button(row, text="Preview", width=8, command=self._on_preview_voice).pack(side=tk.RIGHT)

        self._update_voices()

    def _build_ref_audio_section(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(
            parent, text="Voice Cloning – Reference Audio", padding=_SECTION_INNER_PAD
        )
        frame.pack(fill=tk.X, pady=_SECTION_PAD)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=_ROW_PAD)
        ttk.Label(row, text="File:", width=_LABEL_WIDTH).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.ref_audio_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8)
        )
        ttk.Button(row, text="Browse...", width=9, command=self._on_browse_ref_audio).pack(
            side=tk.RIGHT
        )

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=_ROW_PAD)
        ttk.Label(row, text="Saved clones:", width=_LABEL_WIDTH).pack(side=tk.LEFT)
        self.saved_clone_combo = ttk.Combobox(
            row, textvariable=self._saved_clone_var, state="readonly"
        )
        self.saved_clone_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.saved_clone_combo.bind("<<ComboboxSelected>>", self._on_saved_clone_selected)
        ttk.Button(row, text="Refresh", width=9, command=self._refresh_saved_clones).pack(
            side=tk.RIGHT
        )

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=_ROW_PAD)
        ttk.Label(row, text="Emotion:", width=_LABEL_WIDTH).pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.emotion_var, values=EMOTIONS, state="readonly").pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

    def _build_record_section(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(
            parent, text="Voice Cloning – Record / Import", padding=_SECTION_INNER_PAD
        )
        frame.pack(fill=tk.X, pady=_SECTION_PAD)

        # Record buttons
        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=_ROW_PAD)
        self.rec_start_btn = ttk.Button(btn_row, text="Start Recording", command=self._on_rec_start)
        self.rec_start_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.rec_stop_btn = ttk.Button(
            btn_row, text="Stop", command=self._on_rec_stop, state=tk.DISABLED
        )
        self.rec_stop_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.rec_play_btn = ttk.Button(
            btn_row, text="Play", command=self._on_rec_play, state=tk.DISABLED
        )
        self.rec_play_btn.pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(btn_row, textvariable=self.rec_timer_var, foreground="gray").pack(side=tk.LEFT)

        # Save row
        save_row = ttk.Frame(frame)
        save_row.pack(fill=tk.X, pady=_ROW_PAD)
        ttk.Label(save_row, text="Save as:", width=_LABEL_WIDTH).pack(side=tk.LEFT)
        self.rec_name_entry = ttk.Entry(save_row, width=24)
        self.rec_name_entry.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(save_row, text="Save", command=self._on_rec_save).pack(side=tk.LEFT)

        # Import row
        up_row = ttk.Frame(frame)
        up_row.pack(fill=tk.X, pady=_ROW_PAD)
        ttk.Label(up_row, text="Or import:", width=_LABEL_WIDTH).pack(side=tk.LEFT)
        ttk.Entry(up_row, textvariable=self._upload_path_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8)
        )
        ttk.Button(up_row, text="Browse...", width=9, command=self._on_upload_browse).pack(
            side=tk.RIGHT, padx=(0, 4)
        )
        ttk.Button(up_row, text="Import", width=8, command=self._on_upload_import).pack(
            side=tk.RIGHT
        )

        self._refresh_saved_clones()

    def _build_clone_test_section(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="Voice Cloning – Test", padding=_SECTION_INNER_PAD)
        frame.pack(fill=tk.X, pady=_SECTION_PAD)

        trow = ttk.Frame(frame)
        trow.pack(fill=tk.X, pady=_ROW_PAD)
        ttk.Label(trow, text="Engine:", width=_LABEL_WIDTH).pack(side=tk.LEFT)
        ttk.Combobox(
            trow,
            textvariable=self.clone_engine_var,
            values=["indextts", "fish-speech"],
            state="readonly",
            width=14,
        ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(trow, text="Language:").pack(side=tk.LEFT)
        ttk.Combobox(
            trow,
            textvariable=self.clone_lang_var,
            values=["en", "zh", "ja"],
            state="readonly",
            width=8,
        ).pack(side=tk.LEFT, padx=(4, 0))

        trow2 = ttk.Frame(frame)
        trow2.pack(fill=tk.X, pady=_ROW_PAD)
        ttk.Label(trow2, text="Text:", width=_LABEL_WIDTH).pack(side=tk.LEFT)
        ttk.Entry(trow2, textvariable=self.clone_text_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        trow3 = ttk.Frame(frame)
        trow3.pack(fill=tk.X, pady=(_ROW_PAD, 0))
        self.synth_play_btn = ttk.Button(
            trow3, text="Synthesize & Play", command=self._on_clone_test
        )
        self.synth_play_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.synth_save_btn = ttk.Button(
            trow3, text="Synthesize & Save", command=self._on_clone_save
        )
        self.synth_save_btn.pack(side=tk.LEFT)

    def _build_translation_section(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="Translation", padding=_SECTION_INNER_PAD)
        frame.pack(fill=tk.X, pady=_SECTION_PAD)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=_ROW_PAD)
        ttk.Label(row, text="Model:", width=_LABEL_WIDTH).pack(side=tk.LEFT)
        ttk.Combobox(
            row,
            textvariable=self.trans_model_var,
            values=["helsinki", "nllb"],
            state="readonly",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=_ROW_PAD)
        ttk.Label(row, text="Device:", width=_LABEL_WIDTH).pack(side=tk.LEFT)
        ttk.Combobox(
            row,
            textvariable=self.trans_device_var,
            values=["cpu", "cuda", "mps", "rocm"],
            state="readonly",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _build_tts_device_section(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="TTS Device", padding=_SECTION_INNER_PAD)
        frame.pack(fill=tk.X, pady=_SECTION_PAD)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=_ROW_PAD)
        ttk.Label(row, text="Device:", width=_LABEL_WIDTH).pack(side=tk.LEFT)
        ttk.Combobox(
            row,
            textvariable=self.tts_device_var,
            values=["cpu", "cuda", "mps", "rocm"],
            state="readonly",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _build_calibration_section(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="Noise Calibration", padding=_SECTION_INNER_PAD)
        frame.pack(fill=tk.X, pady=_SECTION_PAD)

        row = ttk.Frame(frame)
        row.pack(fill=tk.X, pady=_ROW_PAD)
        ttk.Button(row, text="Calibrate Noise (2s)", command=self._on_calibrate).pack(side=tk.LEFT)
        ttk.Label(row, textvariable=self.noise_level_var, foreground="gray").pack(
            side=tk.LEFT, padx=16
        )

    def _build_downloads_section(self, parent: tk.Widget) -> None:
        frame = ttk.LabelFrame(parent, text="Model Downloads", padding=_SECTION_INNER_PAD)
        frame.pack(fill=tk.X, pady=_SECTION_PAD)

        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, pady=_ROW_PAD)
        for label, cmd_arg in [
            ("IndexTTS v2", "indextts"),
            ("Fish-Speech", "fish_speech"),
            ("Kokoro TTS", "kokoro"),
            ("Whisper", "whisper"),
        ]:
            ttk.Button(
                row1, text=label, command=lambda a=cmd_arg: self._model_ctrl.download(a)
            ).pack(side=tk.LEFT, padx=(0, 6))

        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=_ROW_PAD)
        ttk.Button(row2, text="Translation Models...", command=self._on_dl_translation).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(
            row2, text="Download ALL", command=lambda: self._model_ctrl.download("all")
        ).pack(side=tk.LEFT)

        ttk.Label(frame, textvariable=self.dl_progress_var, foreground="gray", wraplength=550).pack(
            fill=tk.X, pady=(_ROW_PAD, 0)
        )

        cache_row = ttk.Frame(frame)
        cache_row.pack(fill=tk.X, pady=(_ROW_PAD, 0))
        ttk.Button(cache_row, text="Show Cache Info", command=self._refresh_cache).pack(
            side=tk.LEFT
        )
        ttk.Label(frame, textvariable=self.cache_info_var, foreground="gray", wraplength=550).pack(
            fill=tk.X, pady=(_ROW_PAD, 0)
        )

    # ==================================================================
    # State sync
    # ==================================================================

    def sync_to_state(self) -> None:
        """Write current widget values into AppState."""
        s = self._state

        sel = self.source_var.get()
        if sel == "Auto-detect":
            s.source_lang = "auto"
        else:
            for code, name in SUPPORTED_LANGUAGES:
                if f"{name} ({code})" == sel:
                    s.source_lang = code
                    break

        sel = self.target_var.get()
        for code, name in SUPPORTED_LANGUAGES:
            if f"{name} ({code})" == sel:
                s.target_lang = code
                break

        s.tts_engine = self.engine_var.get()
        s.voice = self.voice_var.get()
        s.capture_mode = self.capture_mode_var.get()
        s.ref_audio_path = self.ref_audio_var.get()
        s.emotion = self.emotion_var.get()
        s.translation_model = self.trans_model_var.get()
        s.trans_device = self.trans_device_var.get()
        s.tts_device = self.tts_device_var.get()

    # ==================================================================
    # Event handlers
    # ==================================================================

    def _set_default_target(self) -> None:
        for code, name in SUPPORTED_LANGUAGES:
            if code == self._state.target_lang:
                self.target_var.set(f"{name} ({code})")
                break
        self._update_voices()

    def _on_target_changed(self, event=None) -> None:
        self._update_voices()

    def _on_engine_changed(self, event=None) -> None:
        self._update_voices()

    def _update_voices(self) -> None:
        engine = self.engine_var.get()
        target_lang = "en"
        sel = self.target_var.get()
        for code, name in SUPPORTED_LANGUAGES:
            if f"{name} ({code})" == sel:
                target_lang = code
                break
        try:
            voices = get_available_voices(target_lang, engine)
        except Exception:
            voices = []
        self.voice_combo["values"] = voices
        if voices:
            self.voice_var.set(voices[0])
        else:
            self.voice_var.set("")

    def _on_preview_voice(self) -> None:
        self.sync_to_state()
        threading.Thread(target=self._preview_thread, daemon=True).start()

    def _preview_thread(self) -> None:
        try:
            from copytalker.audio.playback import AudioPlayer
            from copytalker.core.config import TTSConfig
            from copytalker.tts.base import get_tts_engine

            s = self._state
            tts_config = TTSConfig(
                engine=s.tts_engine, voice=s.voice, language=s.target_lang, device=s.tts_device
            )
            if s.ref_audio_path:
                tts_config.indextts_reference_audio = s.ref_audio_path
                tts_config.fish_speech_reference_audio = s.ref_audio_path
            engine = get_tts_engine(s.tts_engine, tts_config)
            audio, sr = engine.synthesize("Hello, this is a voice preview.", s.target_lang)
            if len(audio) > 0:
                player = AudioPlayer(default_sample_rate=sr)
                player.play(audio, sr, blocking=True)
                player.close()
        except Exception as e:
            self._event_queue.put(("error", f"Voice preview failed: {e}"))

    def _on_browse_ref_audio(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg"), ("All files", "*.*")]
        )
        if path:
            self.ref_audio_var.set(path)

    def _on_saved_clone_selected(self, event=None) -> None:
        from copytalker.audio.recorder import list_saved_voice_clones

        sel = self._saved_clone_var.get()
        for clone in list_saved_voice_clones():
            if clone["name"] == sel:
                self.ref_audio_var.set(clone["path"])
                break

    def _refresh_saved_clones(self) -> None:
        try:
            from copytalker.audio.recorder import list_saved_voice_clones

            clones = list_saved_voice_clones()
            names = [c["name"] for c in clones]
            self.saved_clone_combo["values"] = names
        except Exception:
            pass

    # Recording handlers
    def _on_rec_start(self) -> None:
        from copytalker.audio.recorder import VoiceRecorder

        self._recorder = VoiceRecorder()
        self._recorder.start()
        self.rec_start_btn.config(state=tk.DISABLED)
        self.rec_stop_btn.config(state=tk.NORMAL)
        self.rec_play_btn.config(state=tk.DISABLED)
        self._update_rec_timer()

    def _on_rec_stop(self) -> None:
        if self._recorder:
            self._recorder.stop()
        self.rec_start_btn.config(state=tk.NORMAL)
        self.rec_stop_btn.config(state=tk.DISABLED)
        self.rec_play_btn.config(state=tk.NORMAL)
        if self._rec_timer_id:
            self.after_cancel(self._rec_timer_id)
            self._rec_timer_id = None

    def _on_rec_play(self) -> None:
        if not self._recorder:
            return

        def play():
            try:
                from copytalker.audio.playback import AudioPlayer

                audio = self._recorder.get_audio_array()
                if len(audio) > 0:
                    player = AudioPlayer(default_sample_rate=16000)
                    player.play(audio, 16000, blocking=True)
                    player.close()
            except Exception as e:
                self._event_queue.put(("error", f"Playback error: {e}"))

        threading.Thread(target=play, daemon=True).start()

    def _on_rec_save(self) -> None:
        if not self._recorder:
            return
        name = self.rec_name_entry.get().strip()
        try:
            path = self._recorder.save(name=name if name else None)
            self.ref_audio_var.set(path)
            self._refresh_saved_clones()
            self._event_queue.put(("status", f"Recording saved: {path}"))
        except Exception as e:
            self._event_queue.put(("error", f"Save error: {e}"))

    def _update_rec_timer(self) -> None:
        if self._recorder and self._recorder.is_recording:
            self.rec_timer_var.set(f"{self._recorder.duration:.1f}s")
            self._rec_timer_id = self.after(100, self._update_rec_timer)

    # Upload handlers
    def _on_upload_browse(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg"), ("All files", "*.*")]
        )
        if path:
            self._upload_path_var.set(path)

    def _on_upload_import(self) -> None:
        path = self._upload_path_var.get().strip()
        if not path:
            return
        try:
            import shutil
            from pathlib import Path

            from copytalker.core.config import get_default_cache_dir

            dest_dir = get_default_cache_dir() / "voice_clones"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / Path(path).name
            shutil.copy2(path, dest)
            self.ref_audio_var.set(str(dest))
            self._refresh_saved_clones()
            self._event_queue.put(("status", f"Imported: {dest}"))
        except Exception as e:
            self._event_queue.put(("error", f"Import error: {e}"))

    # Clone test handlers
    def _on_clone_test(self) -> None:
        self._run_clone_test(play=True)

    def _on_clone_save(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV", "*.wav")])
        if path:
            self._run_clone_test(play=False, save_path=path)

    def _run_clone_test(self, play: bool = True, save_path: str | None = None) -> None:
        ref = self.ref_audio_var.get()
        if not ref:
            messagebox.showwarning("No Reference", "Select a reference audio first.")
            return
        text = self.clone_text_var.get()
        engine = self.clone_engine_var.get()
        lang = self.clone_lang_var.get()
        self.synth_play_btn.config(state=tk.DISABLED)
        self.synth_save_btn.config(state=tk.DISABLED)
        threading.Thread(
            target=self._clone_test_thread,
            args=(text, ref, engine, lang, play, save_path),
            daemon=True,
        ).start()

    def _clone_test_thread(self, text, ref, engine, lang, play=True, save_path=None) -> None:
        try:
            from copytalker.core.config import TTSConfig
            from copytalker.tts.base import get_tts_engine

            tts_config = TTSConfig(engine=engine, language=lang, device=self.tts_device_var.get())
            tts_config.indextts_reference_audio = ref
            tts_config.fish_speech_reference_audio = ref
            emotion = self.emotion_var.get()
            if emotion:
                tts_config.indextts_emotion = emotion
                tts_config.fish_speech_emotion = emotion
            tts = get_tts_engine(engine, tts_config)
            audio, sr = tts.synthesize(text, lang, ref)
            if len(audio) == 0:
                self._event_queue.put(("error", "Voice clone produced no audio"))
                return
            if save_path:
                from copytalker.api import _write_wav

                _write_wav(save_path, audio, sr)
                self._event_queue.put(("status", f"Saved: {save_path}"))
            if play:
                from copytalker.audio.playback import AudioPlayer

                player = AudioPlayer(default_sample_rate=sr)
                player.play(audio, sr, blocking=True)
                player.close()
                self._event_queue.put(("status", "Voice clone playback complete"))
        except Exception as e:
            logger.error(f"Clone test error: {e}")
            self._event_queue.put(("error", f"Clone test failed: {e}"))
        finally:
            self._event_queue.put(("clone_test_done", None))

    # Advanced handlers
    def _on_calibrate(self) -> None:
        self.noise_level_var.set("Calibrating... (2s)")
        threading.Thread(target=self._calibrate_thread, daemon=True).start()

    def _calibrate_thread(self) -> None:
        try:
            from copytalker.audio.capture import AudioCapturer
            from copytalker.core.config import AudioConfig

            capturer = AudioCapturer(AudioConfig())
            level = capturer.calibrate_noise(duration=2.0)
            self._state.calibrated_noise_level = level
            self._event_queue.put(("calibration_done", level))
        except Exception as e:
            self._event_queue.put(("error", f"Calibration failed: {e}"))

    # Translation model download dialog
    def _on_dl_translation(self) -> None:
        dialog = tk.Toplevel(self.winfo_toplevel())
        dialog.title("Select Translation Languages")
        dialog.geometry("500x600")
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()
        dialog.wait_visibility()

        lang_vars: dict[str, tk.BooleanVar] = {}

        ttk.Label(
            dialog, text="Select languages to download:", font=("Helvetica", 12, "bold")
        ).pack(pady=10)

        quick = ttk.Frame(dialog)
        quick.pack(pady=5)
        ttk.Button(
            quick, text="Select All", command=lambda: [v.set(True) for v in lang_vars.values()]
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            quick,
            text="Deselect All",
            command=lambda: [v.set(False) for v in lang_vars.values()],
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            quick,
            text="East Asian",
            command=lambda: [lang_vars[c].set(True) for c in ["zh", "ja", "ko"] if c in lang_vars],
        ).pack(side=tk.LEFT, padx=5)

        lf = ttk.Frame(dialog)
        lf.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        canvas = tk.Canvas(lf)
        sb = ttk.Scrollbar(lf, orient="vertical", command=canvas.yview)
        sf = ttk.Frame(canvas)
        sf.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=sf, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        for code, name in SUPPORTED_LANGUAGES:
            if code == "en":
                continue
            var = tk.BooleanVar(value=code in ["zh", "ja", "es", "fr", "de"])
            lang_vars[code] = var
            ttk.Checkbutton(sf, text=f"{name} ({code})", variable=var).pack(anchor="w", pady=2)

        btn = ttk.Frame(dialog)
        btn.pack(pady=15)

        def on_ok():
            selected = [c for c, v in lang_vars.items() if v.get()]
            if not selected:
                messagebox.showwarning("No Selection", "Please select at least one language.")
                return
            dialog.destroy()
            self._model_ctrl.download_translation_for_langs(selected)

        ttk.Button(btn, text="Download", command=on_ok).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=10)

    def _refresh_cache(self) -> None:
        info = self._model_ctrl.refresh_cache_info()
        self.cache_info_var.set(info)

    # Start / Stop handlers
    def _handle_start(self) -> None:
        self.sync_to_state()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._on_start()

    def _handle_stop(self) -> None:
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self._on_stop()

    # ==================================================================
    # External event handlers (called from app.py)
    # ==================================================================

    def on_pipeline_stopped(self) -> None:
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def on_clone_test_done(self) -> None:
        self.synth_play_btn.config(state=tk.NORMAL)
        self.synth_save_btn.config(state=tk.NORMAL)

    def on_calibration_done(self, level: float) -> None:
        self.noise_level_var.set(f"Noise level: {level:.4f}")

    def on_download_done(self) -> None:
        self.dl_progress_var.set("Download complete!")
