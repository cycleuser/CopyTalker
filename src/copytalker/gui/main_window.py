"""
Main Tkinter GUI window for CopyTalker.

Uses lazy imports for heavy dependencies to allow the GUI window to
display even when some dependencies are missing.
"""

import logging
import queue
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from typing import Optional, TYPE_CHECKING

from copytalker import __version__
from copytalker.core.constants import (
    SUPPORTED_LANGUAGES,
    get_available_voices,
    AUTO_DETECT_CODE,
)

if TYPE_CHECKING:
    from copytalker.core.config import AppConfig
    from copytalker.core.pipeline import TranslationPipeline

logger = logging.getLogger(__name__)


class CopyTalkerGUI:
    """Main GUI application for CopyTalker."""
    
    WINDOW_TITLE = "CopyTalker"
    WINDOW_SIZE = "650x800"
    UPDATE_INTERVAL_MS = 100
    
    def __init__(self, root: tk.Tk):
        """
        Initialize the GUI.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title(self.WINDOW_TITLE)
        self.root.geometry(self.WINDOW_SIZE)
        self.root.minsize(500, 600)
        
        # State
        self._pipeline: Optional[TranslationPipeline] = None
        self._is_running = False
        self._event_queue: queue.Queue = queue.Queue()
        self._calibrated_noise_level: float = 0.0
        
        # Build UI
        self._create_widgets()
        self._setup_bindings()
        
        # Start event processing
        self._process_events()
    
    def _create_widgets(self) -> None:
        """Create all GUI widgets."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="CopyTalker",
            font=("Helvetica", 18, "bold"),
        )
        title_label.pack(pady=(0, 5))
        
        subtitle_label = ttk.Label(
            main_frame,
            text="Real-time Speech Translation",
            font=("Helvetica", 10),
        )
        subtitle_label.pack(pady=(0, 15))
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Source language
        source_frame = ttk.Frame(settings_frame)
        source_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(source_frame, text="Source Language:", width=15).pack(side=tk.LEFT)
        
        self.source_var = tk.StringVar(value=AUTO_DETECT_CODE)
        source_values = ["Auto-detect"] + [f"{name} ({code})" for code, name in SUPPORTED_LANGUAGES]
        self.source_combo = ttk.Combobox(
            source_frame,
            textvariable=self.source_var,
            values=source_values,
            state="readonly",
            width=30,
        )
        self.source_combo.current(0)
        self.source_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Target language
        target_frame = ttk.Frame(settings_frame)
        target_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(target_frame, text="Target Language:", width=15).pack(side=tk.LEFT)
        
        self.target_var = tk.StringVar()
        target_values = [f"{name} ({code})" for code, name in SUPPORTED_LANGUAGES]
        self.target_combo = ttk.Combobox(
            target_frame,
            textvariable=self.target_var,
            values=target_values,
            state="readonly",
            width=30,
        )
        self.target_combo.current(1)  # Default to Chinese
        self.target_combo.pack(side=tk.LEFT, padx=(10, 0))
        self.target_combo.bind("<<ComboboxSelected>>", self._on_target_changed)
        
        # Voice selection
        voice_frame = ttk.Frame(settings_frame)
        voice_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(voice_frame, text="Voice:", width=15).pack(side=tk.LEFT)
        
        self.voice_var = tk.StringVar()
        self.voice_combo = ttk.Combobox(
            voice_frame,
            textvariable=self.voice_var,
            state="readonly",
            width=22,
        )
        self.voice_combo.pack(side=tk.LEFT, padx=(10, 0))
        self._update_voice_list()
        
        # Voice preview button
        self.preview_button = ttk.Button(
            voice_frame,
            text="Preview",
            command=self._on_preview_voice,
            width=8,
        )
        self.preview_button.pack(side=tk.LEFT, padx=(5, 0))
        
        # TTS Engine
        engine_frame = ttk.Frame(settings_frame)
        engine_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(engine_frame, text="TTS Engine:", width=15).pack(side=tk.LEFT)
        
        self.engine_var = tk.StringVar(value="auto")
        engine_combo = ttk.Combobox(
            engine_frame,
            textvariable=self.engine_var,
            values=["auto", "kokoro", "edge-tts", "pyttsx3"],
            state="readonly",
            width=30,
        )
        engine_combo.current(0)
        engine_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Advanced Settings frame
        advanced_frame = ttk.LabelFrame(main_frame, text="Advanced Settings", padding="10")
        advanced_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Translation Model
        trans_model_frame = ttk.Frame(advanced_frame)
        trans_model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(trans_model_frame, text="Translation Model:", width=18).pack(side=tk.LEFT)
        
        self.trans_model_var = tk.StringVar(value="helsinki")
        trans_model_combo = ttk.Combobox(
            trans_model_frame,
            textvariable=self.trans_model_var,
            values=["helsinki", "nllb"],
            state="readonly",
            width=27,
        )
        trans_model_combo.current(0)
        trans_model_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Translation Device
        trans_device_frame = ttk.Frame(advanced_frame)
        trans_device_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(trans_device_frame, text="Translation Device:", width=18).pack(side=tk.LEFT)
        
        self.trans_device_var = tk.StringVar(value="cuda")
        trans_device_combo = ttk.Combobox(
            trans_device_frame,
            textvariable=self.trans_device_var,
            values=["cuda", "cpu"],
            state="readonly",
            width=27,
        )
        trans_device_combo.current(0)
        trans_device_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # TTS Device
        tts_device_frame = ttk.Frame(advanced_frame)
        tts_device_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(tts_device_frame, text="TTS Device:", width=18).pack(side=tk.LEFT)
        
        self.tts_device_var = tk.StringVar(value="cpu")
        tts_device_combo = ttk.Combobox(
            tts_device_frame,
            textvariable=self.tts_device_var,
            values=["cpu", "cuda"],
            state="readonly",
            width=27,
        )
        tts_device_combo.current(0)  # Default to CPU to avoid GPU contention
        tts_device_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Calibration button
        calibrate_frame = ttk.Frame(advanced_frame)
        calibrate_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(calibrate_frame, text="Noise Calibration:", width=18).pack(side=tk.LEFT)
        
        self.calibrate_button = ttk.Button(
            calibrate_frame,
            text="Calibrate",
            command=self._on_calibrate,
            width=12,
        )
        self.calibrate_button.pack(side=tk.LEFT, padx=(10, 0))
        
        self.noise_level_var = tk.StringVar(value="Not calibrated")
        ttk.Label(
            calibrate_frame,
            textvariable=self.noise_level_var,
            width=15,
        ).pack(side=tk.LEFT, padx=(10, 0))
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            font=("Helvetica", 11),
        )
        status_label.pack()
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_button = ttk.Button(
            button_frame,
            text="Start Translation",
            command=self._on_start,
            width=20,
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(
            button_frame,
            text="Stop",
            command=self._on_stop,
            width=20,
            state=tk.DISABLED,
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.download_button = ttk.Button(
            button_frame,
            text="Download Models",
            command=self._on_download_models,
            width=20,
        )
        self.download_button.pack(side=tk.LEFT)
        
        # Transcription display
        trans_frame = ttk.LabelFrame(main_frame, text="Transcription (What you said)", padding="10")
        trans_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.transcription_text = scrolledtext.ScrolledText(
            trans_frame,
            height=6,
            wrap=tk.WORD,
            font=("Helvetica", 11),
            state=tk.DISABLED,
        )
        self.transcription_text.pack(fill=tk.BOTH, expand=True)
        
        # Translation display
        translation_frame = ttk.LabelFrame(main_frame, text="Translation", padding="10")
        translation_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.translation_text = scrolledtext.ScrolledText(
            translation_frame,
            height=6,
            wrap=tk.WORD,
            font=("Helvetica", 11),
            state=tk.DISABLED,
        )
        self.translation_text.pack(fill=tk.BOTH, expand=True)
        
        # Footer
        footer_label = ttk.Label(
            main_frame,
            text="Press Start to begin real-time translation",
            font=("Helvetica", 9),
        )
        footer_label.pack(pady=(5, 0))
    
    def _setup_bindings(self) -> None:
        """Setup keyboard bindings."""
        self.root.bind("<Control-q>", lambda e: self._on_quit())
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)
        
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
    
    def _show_about(self) -> None:
        """Show about dialog with version info."""
        messagebox.showinfo(
            "About CopyTalker",
            f"CopyTalker v{__version__}\n\n"
            "Real-time multilingual speech-to-speech translation\n\n"
            "https://github.com/cycleuser/CopyTalker"
        )
    
    def _update_voice_list(self) -> None:
        """Update voice list based on selected target language."""
        target_lang = self._get_target_lang()
        voices = get_available_voices(target_lang, "kokoro")
        
        self.voice_combo["values"] = voices
        if voices:
            self.voice_combo.current(0)
    
    def _on_target_changed(self, event=None) -> None:
        """Handle target language change."""
        self._update_voice_list()
    
    def _on_preview_voice(self) -> None:
        """Preview the selected voice with sample text."""
        voice = self.voice_var.get()
        target_lang = self._get_target_lang()
        engine = self.engine_var.get()
        device = self.tts_device_var.get()
        
        if not voice:
            messagebox.showwarning("Warning", "Please select a voice first")
            return
        
        # Sample text based on target language
        sample_texts = {
            "zh": "你好，这是语音预览测试。",
            "en": "Hello, this is a voice preview test.",
            "ja": "こんにちは、これは音声プレビューテストです。",
            "ko": "안녕하세요, 이것은 음성 미리보기 테스트입니다.",
            "es": "Hola, esta es una prueba de vista previa de voz.",
            "fr": "Bonjour, ceci est un test de prévisualisation vocale.",
            "de": "Hallo, dies ist ein Sprachvorschautest.",
            "ru": "Привет, это тест предварительного просмотра голоса.",
            "ar": "مرحبا، هذا اختبار معاينة الصوت.",
        }
        sample_text = sample_texts.get(target_lang, sample_texts["en"])
        
        self.preview_button.config(state=tk.DISABLED)
        self.status_var.set(f"Previewing voice: {voice}...")
        
        # Run preview in background thread
        thread = threading.Thread(
            target=self._preview_voice_thread,
            args=(sample_text, target_lang, voice, engine, device),
            daemon=True,
        )
        thread.start()
    
    def _preview_voice_thread(self, text: str, lang: str, voice: str, engine: str, device: str) -> None:
        """Run voice preview in background thread."""
        try:
            from copytalker.core.config import TTSConfig
            from copytalker.tts.base import get_tts_engine
            from copytalker.audio.playback import AudioPlayer
            
            # Create TTS config
            tts_config = TTSConfig(
                engine=engine,
                voice=voice,
                language=lang,
                device=device,
            )
            
            # Get TTS engine
            tts = get_tts_engine(engine, tts_config)
            
            # Synthesize
            audio, sample_rate = tts.synthesize(text, lang, voice)
            
            if len(audio) == 0:
                self._event_queue.put(("error", "TTS produced no audio"))
                return
            
            # Play audio
            player = AudioPlayer(default_sample_rate=sample_rate)
            player.play(audio, sample_rate, blocking=True)
            player.close()
            
            self._event_queue.put(("status", "Preview complete"))
            
        except Exception as e:
            logger.error(f"Preview error: {e}")
            self._event_queue.put(("error", f"Preview failed: {e}"))
        finally:
            # Re-enable button via event queue
            self._event_queue.put(("preview_done", None))
    
    def _on_calibrate(self) -> None:
        """Calibrate noise level."""
        self.calibrate_button.config(state=tk.DISABLED)
        self.status_var.set("Calibrating... Please stay quiet for 2 seconds")
        
        # Run calibration in background thread
        thread = threading.Thread(
            target=self._calibrate_thread,
            daemon=True,
        )
        thread.start()
    
    def _calibrate_thread(self) -> None:
        """Run calibration in background thread."""
        try:
            from copytalker.audio.capture import AudioCapturer
            from copytalker.core.config import AudioConfig
            
            config = AudioConfig()
            capturer = AudioCapturer(config)
            
            noise_level = capturer.calibrate_noise(duration_s=2.0)
            
            # Store the calibrated noise level for later use
            self._calibrated_noise_level = noise_level
            
            self._event_queue.put(("calibration_done", noise_level))
            self._event_queue.put(("status", f"Calibration complete: noise={noise_level:.4f}"))
            
        except Exception as e:
            logger.error(f"Calibration error: {e}")
            self._event_queue.put(("error", f"Calibration failed: {e}"))
            self._event_queue.put(("calibration_done", 0.0))
    
    def _on_download_models(self) -> None:
        """Download all required models."""
        self.download_button.config(state=tk.DISABLED)
        self.status_var.set("Downloading models... This may take a while")
        
        thread = threading.Thread(
            target=self._download_models_thread,
            daemon=True,
        )
        thread.start()
    
    def _download_models_thread(self) -> None:
        """Download models in background thread."""
        try:
            from copytalker.core.constants import SUPPORTED_LANGUAGES, DEFAULT_TRANSLATION_MODELS
            
            total_models = []
            
            # Collect Helsinki models for all defined pairs
            for key, models in DEFAULT_TRANSLATION_MODELS.items():
                if key == "multilingual":
                    continue
                for m in models:
                    if m.startswith("Helsinki-NLP/") and m not in total_models:
                        total_models.append(m)
            
            # Add NLLB default model
            nllb_model = "facebook/nllb-200-distilled-600M"
            if nllb_model not in total_models:
                total_models.append(nllb_model)
            
            downloaded = 0
            failed = []
            
            for i, model_name in enumerate(total_models):
                self._event_queue.put(("status", f"Downloading ({i+1}/{len(total_models)}): {model_name}"))
                try:
                    from transformers import AutoTokenizer, AutoModel
                    
                    if model_name.startswith("Helsinki-NLP/"):
                        from transformers import MarianTokenizer, MarianMTModel
                        MarianTokenizer.from_pretrained(model_name)
                        MarianMTModel.from_pretrained(model_name)
                    elif "nllb" in model_name:
                        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                        AutoTokenizer.from_pretrained(model_name)
                        AutoModelForSeq2SeqLM.from_pretrained(model_name)
                    else:
                        AutoTokenizer.from_pretrained(model_name)
                        AutoModel.from_pretrained(model_name)
                    
                    downloaded += 1
                    logger.info(f"Downloaded: {model_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to download {model_name}: {e}")
                    failed.append(model_name)
            
            # Also download Whisper model
            self._event_queue.put(("status", f"Downloading Whisper (small)..."))
            try:
                from faster_whisper import WhisperModel
                WhisperModel("small", device="cpu", compute_type="float32")
                downloaded += 1
            except Exception as e:
                logger.error(f"Failed to download Whisper: {e}")
                failed.append("whisper-small")
            
            if failed:
                msg = f"Downloaded {downloaded} models. Failed: {', '.join(failed)}"
            else:
                msg = f"All {downloaded} models downloaded successfully!"
            
            self._event_queue.put(("status", msg))
            self._event_queue.put(("download_done", None))
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            self._event_queue.put(("error", f"Download error: {e}"))
            self._event_queue.put(("download_done", None))
    
    def _get_source_lang(self) -> str:
        """Get selected source language code."""
        selection = self.source_combo.get()
        if selection == "Auto-detect":
            return AUTO_DETECT_CODE
        
        # Extract code from "Name (code)" format
        for code, name in SUPPORTED_LANGUAGES:
            if f"{name} ({code})" == selection:
                return code
        return AUTO_DETECT_CODE
    
    def _get_target_lang(self) -> str:
        """Get selected target language code."""
        selection = self.target_combo.get()
        
        for code, name in SUPPORTED_LANGUAGES:
            if f"{name} ({code})" == selection:
                return code
        return "en"
    
    def _on_start(self) -> None:
        """Start the translation pipeline."""
        if self._is_running:
            return
        
        # Lazy import AppConfig
        from copytalker.core.config import AppConfig
        
        # Build configuration
        config = AppConfig()
        config.stt.language = self._get_source_lang()
        config.translation.source_lang = self._get_source_lang()
        config.translation.target_lang = self._get_target_lang()
        config.translation.model_name = self.trans_model_var.get()
        config.translation.device = self.trans_device_var.get()
        config.tts.engine = self.engine_var.get()
        config.tts.language = self._get_target_lang()
        config.tts.voice = self.voice_var.get()
        config.tts.device = self.tts_device_var.get()
        
        # Apply calibrated noise level if available
        if self._calibrated_noise_level > 0:
            config.audio.calibrated_noise_level = self._calibrated_noise_level
        
        # Update UI
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Initializing...")
        
        # Clear displays
        self._clear_displays()
        
        # Start pipeline in background thread
        thread = threading.Thread(target=self._start_pipeline, args=(config,), daemon=True)
        thread.start()
    
    def _start_pipeline(self, config) -> None:
        """Start pipeline in background thread."""
        try:
            # Lazy import TranslationPipeline
            from copytalker.core.pipeline import TranslationPipeline
            
            self._pipeline = TranslationPipeline(config)
            
            # Register callbacks
            self._pipeline.register_callback("transcription", self._on_transcription)
            self._pipeline.register_callback("translation", self._on_translation)
            self._pipeline.register_callback("status", self._on_status)
            self._pipeline.register_callback("error", self._on_error)
            
            self._pipeline.start()
            self._is_running = True
            
            self._event_queue.put(("status", "Listening..."))
            
        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            self._event_queue.put(("error", str(e)))
            self._event_queue.put(("stopped", None))
    
    def _on_stop(self) -> None:
        """Stop the translation pipeline."""
        if not self._is_running:
            return
        
        self.status_var.set("Stopping...")
        
        # Stop in background
        thread = threading.Thread(target=self._stop_pipeline, daemon=True)
        thread.start()
    
    def _stop_pipeline(self) -> None:
        """Stop pipeline in background thread."""
        try:
            if self._pipeline:
                self._pipeline.stop()
                self._pipeline = None
        except Exception as e:
            logger.error(f"Error stopping pipeline: {e}")
        finally:
            self._is_running = False
            self._event_queue.put(("stopped", None))
    
    def _on_transcription(self, event) -> None:
        """Handle transcription event."""
        self._event_queue.put(("transcription", event.data))
    
    def _on_translation(self, event) -> None:
        """Handle translation event."""
        self._event_queue.put(("translation", event.data))
    
    def _on_status(self, event) -> None:
        """Handle status event."""
        self._event_queue.put(("status", event.data))
    
    def _on_error(self, event) -> None:
        """Handle error event."""
        self._event_queue.put(("error", event.data))
    
    def _process_events(self) -> None:
        """Process events from the queue (runs in main thread)."""
        try:
            while True:
                event_type, data = self._event_queue.get_nowait()
                
                if event_type == "transcription":
                    self._append_text(
                        self.transcription_text,
                        f"[{data.language}] {data.text}\n"
                    )
                elif event_type == "translation":
                    self._append_text(
                        self.translation_text,
                        f"{data.translated_text}\n"
                    )
                elif event_type == "status":
                    self.status_var.set(data)
                elif event_type == "error":
                    self.status_var.set(f"Error: {data}")
                    messagebox.showerror("Error", str(data))
                elif event_type == "stopped":
                    self.start_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)
                    self.status_var.set("Stopped")
                    self._is_running = False
                elif event_type == "preview_done":
                    self.preview_button.config(state=tk.NORMAL)
                elif event_type == "calibration_done":
                    self.calibrate_button.config(state=tk.NORMAL)
                    if data > 0:
                        self.noise_level_var.set(f"Level: {data:.4f}")
                    else:
                        self.noise_level_var.set("Failed")
                elif event_type == "download_done":
                    self.download_button.config(state=tk.NORMAL)
                    
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(self.UPDATE_INTERVAL_MS, self._process_events)
    
    def _append_text(self, text_widget: scrolledtext.ScrolledText, text: str) -> None:
        """Append text to a scrolled text widget."""
        text_widget.config(state=tk.NORMAL)
        text_widget.insert(tk.END, text)
        text_widget.see(tk.END)
        text_widget.config(state=tk.DISABLED)
    
    def _clear_displays(self) -> None:
        """Clear transcription and translation displays."""
        for widget in [self.transcription_text, self.translation_text]:
            widget.config(state=tk.NORMAL)
            widget.delete(1.0, tk.END)
            widget.config(state=tk.DISABLED)
    
    def _on_quit(self) -> None:
        """Handle quit event."""
        if self._is_running:
            self._on_stop()
        self.root.quit()


def main() -> int:
    """Main entry point for GUI."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    
    try:
        root = tk.Tk()
        app = CopyTalkerGUI(root)
        root.mainloop()
        return 0
    except Exception as e:
        logger.error(f"GUI error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
