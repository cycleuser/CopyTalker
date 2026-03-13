"""Qt settings dialog for CopyTalker."""

from __future__ import annotations

import logging
import os
import sys
import threading

try:
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QProgressBar,
        QPushButton,
        QRadioButton,
        QScrollArea,
        QSizePolicy,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    print("PySide6 required")
    sys.exit(1)

from copytalker.core.constants import SUPPORTED_LANGUAGES, get_available_voices
from copytalker.core.i18n import I18n, UI_LANGUAGES

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
]

WHISPER_MODELS = {
    "tiny": ("~75 MB", "fastest, lowest quality"),
    "base": ("~145 MB", "fast, good quality"),
    "small": ("~465 MB", "balanced"),
    "medium": ("~1.5 GB", "slow, high quality"),
    "large": ("~3 GB", "slowest, best quality"),
}

NLLB_MODELS = {
    "distilled-600M": ("~1.2 GB", "fastest"),
    "distilled-1.3B": ("~2.6 GB", "balanced"),
    "1.3B": ("~2.6 GB", "high quality"),
    "3.3B": ("~6.5 GB", "best quality"),
}

HELSINKI_LANG_PAIRS = [
    ("en-zh", "English → Chinese"),
    ("zh-en", "Chinese → English"),
    ("en-ja", "English → Japanese"),
    ("ja-en", "Japanese → English"),
    ("en-ko", "English → Korean"),
    ("ko-en", "Korean → English"),
    ("en-fr", "English → French"),
    ("fr-en", "French → English"),
    ("en-de", "English → German"),
    ("de-en", "German → English"),
    ("en-es", "English → Spanish"),
    ("es-en", "Spanish → English"),
    ("en-ru", "English → Russian"),
    ("ru-en", "Russian → English"),
    ("en-it", "English → Italian"),
    ("it-en", "Italian → English"),
    ("en-pt", "English → Portuguese"),
    ("pt-en", "Portuguese → English"),
    ("en-ar", "English → Arabic"),
    ("ar-en", "Arabic → English"),
]


class QtSettingsDialog(QDialog):
    start_requested = Signal()
    stop_requested = Signal()

    def __init__(self, state, event_queue, model_ctrl, parent=None, ui_lang: str = "en"):
        super().__init__(parent)
        self._i18n = I18n(ui_lang)
        self.setWindowTitle(self._i18n.settings)
        self.setMinimumSize(650, 750)
        self.resize(700, 800)

        self._state = state
        self._event_queue = event_queue
        self._model_ctrl = model_ctrl
        self._download_running = False
        self._download_queue = []

        self._setup_ui()
        self._set_default_target()

    def set_ui_language(self, lang: str):
        self._i18n.lang = lang
        self.setWindowTitle(self._i18n.settings)
        self._update_ui_text()

    def _update_ui_text(self):
        t = self._i18n
        self._lang_group.setTitle(t.languages)
        self._input_group.setTitle(t.input_mode)
        self._tts_group.setTitle(t.text_to_speech)
        self._trans_group.setTitle(t.translation)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_layout.addWidget(scroll, stretch=1)

        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(12)

        cl.addWidget(self._create_language_group())
        cl.addWidget(self._create_input_group())
        cl.addWidget(self._create_history_group())
        cl.addWidget(self._create_tts_group())
        cl.addWidget(self._create_translation_group())
        cl.addWidget(self._create_downloads_group())
        cl.addStretch()

        scroll.setWidget(content)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.hide)
        main_layout.addWidget(button_box)

    def _create_language_group(self):
        t = self._i18n
        self._lang_group = QGroupBox(t.languages)
        form = QFormLayout(self._lang_group)
        form.setSpacing(10)

        source_items = [t.auto_detect] + [f"{name} ({code})" for code, name in SUPPORTED_LANGUAGES]
        self._source_combo = QComboBox()
        self._source_combo.addItems(source_items)
        form.addRow(t.source_language, self._source_combo)

        target_items = [f"{name} ({code})" for code, name in SUPPORTED_LANGUAGES]
        self._target_combo = QComboBox()
        self._target_combo.addItems(target_items)
        self._target_combo.currentIndexChanged.connect(self._on_target_changed)
        form.addRow(t.target_language, self._target_combo)

        return self._lang_group

    def _create_input_group(self):
        t = self._i18n
        self._input_group = QGroupBox(t.input_mode)
        layout = QVBoxLayout(self._input_group)
        layout.setSpacing(8)

        self._ptt_radio = QRadioButton(t.push_to_talk)
        self._ptt_radio.setChecked(True)
        layout.addWidget(self._ptt_radio)

        self._vad_radio = QRadioButton(t.continuous)
        layout.addWidget(self._vad_radio)

        return self._input_group

    def _create_history_group(self):
        t = self._i18n
        self._history_group = QGroupBox("Conversation History")
        layout = QVBoxLayout(self._history_group)
        layout.setSpacing(8)

        self._history_enabled_cb = QCheckBox("Enable conversation history saving")
        self._history_enabled_cb.setChecked(True)
        layout.addWidget(self._history_enabled_cb)

        self._save_original_audio_cb = QCheckBox("Save original audio")
        self._save_original_audio_cb.setChecked(True)
        layout.addWidget(self._save_original_audio_cb)

        self._save_translated_audio_cb = QCheckBox("Save translated audio")
        self._save_translated_audio_cb.setChecked(True)
        layout.addWidget(self._save_translated_audio_cb)

        info_label = QLabel("History is saved to cache/history/ directory")
        info_label.setStyleSheet("color: gray;")
        layout.addWidget(info_label)

        return self._history_group

    def _create_tts_group(self):
        t = self._i18n
        self._tts_group = QGroupBox(t.text_to_speech)
        form = QFormLayout(self._tts_group)
        form.setSpacing(10)

        engines = ["auto", "edge-tts", "pyttsx3", "kokoro", "fish-speech"]
        if not _IS_MACOS:
            engines.insert(4, "indextts")

        self._engine_combo = QComboBox()
        self._engine_combo.addItems(engines)
        self._engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        form.addRow(t.tts_engine, self._engine_combo)

        voice_row = QWidget()
        vr_layout = QHBoxLayout(voice_row)
        vr_layout.setContentsMargins(0, 0, 0, 0)
        self._voice_combo = QComboBox()
        self._voice_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        vr_layout.addWidget(self._voice_combo)
        preview_btn = QPushButton(t.preview)
        preview_btn.clicked.connect(self._on_preview_voice)
        vr_layout.addWidget(preview_btn)
        form.addRow(t.voice, voice_row)

        self._emotion_combo = QComboBox()
        self._emotion_combo.addItems(EMOTIONS)
        form.addRow(t.emotion, self._emotion_combo)

        ref_row = QWidget()
        ref_layout = QHBoxLayout(ref_row)
        ref_layout.setContentsMargins(0, 0, 0, 0)
        self._ref_audio_edit = QLineEdit()
        self._ref_audio_edit.setReadOnly(True)
        self._ref_audio_edit.setPlaceholderText(t.reference_audio)
        ref_layout.addWidget(self._ref_audio_edit)
        browse_btn = QPushButton(t.browse)
        browse_btn.clicked.connect(self._on_browse_ref_audio)
        ref_layout.addWidget(browse_btn)
        form.addRow(t.reference_audio, ref_row)

        return self._tts_group

    def _create_translation_group(self):
        t = self._i18n
        self._trans_group = QGroupBox(t.translation)
        form = QFormLayout(self._trans_group)
        form.setSpacing(10)

        self._trans_model_combo = QComboBox()
        self._trans_model_combo.addItems(["helsinki", "nllb"])
        form.addRow(t.model, self._trans_model_combo)

        self._trans_device_combo = QComboBox()
        self._trans_device_combo.addItems(["cpu", "cuda", "mps", "rocm"])
        form.addRow(t.device, self._trans_device_combo)

        self._tts_device_combo = QComboBox()
        self._tts_device_combo.addItems(["cpu", "cuda", "mps", "rocm"])
        form.addRow(t.tts_device, self._tts_device_combo)

        return self._trans_group

    def _create_downloads_group(self):
        t = self._i18n
        self._download_group = QGroupBox(t.model_downloads)
        layout = QVBoxLayout(self._download_group)
        layout.setSpacing(16)

        tabs = QTabWidget()
        tabs.addTab(self._create_whisper_tab(), "Whisper (STT)")
        tabs.addTab(self._create_translation_tab(), "Translation")
        tabs.addTab(self._create_tts_tab(), "TTS")
        tabs.addTab(self._create_batch_tab(), "Batch Download")
        layout.addWidget(tabs)

        self._download_progress = QProgressBar()
        self._download_progress.setVisible(False)
        layout.addWidget(self._download_progress)

        self._download_label = QLabel("")
        self._download_label.setWordWrap(True)
        layout.addWidget(self._download_label)

        return self._download_group

    def _create_whisper_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)

        info_label = QLabel(
            "Whisper: Speech-to-Text model for transcribing audio.\nQuantization: int8 (fastest, smallest) > float16 (balanced) > float32 (slowest, best)"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        quant_row = QHBoxLayout()
        quant_label = QLabel("Quantization:")
        quant_label.setMinimumWidth(80)
        quant_row.addWidget(quant_label)

        self._whisper_quant_combo = QComboBox()
        self._whisper_quant_combo.addItems(
            ["int8 (fastest)", "float16 (balanced)", "float32 (best)"]
        )
        quant_row.addWidget(self._whisper_quant_combo)
        quant_row.addStretch()
        layout.addLayout(quant_row)

        for model_name, (size, desc) in WHISPER_MODELS.items():
            row = QHBoxLayout()
            label = QLabel(f"{model_name.capitalize()}")
            label.setMinimumWidth(80)
            row.addWidget(label)

            size_label = QLabel(f"{size}")
            size_label.setMinimumWidth(80)
            row.addWidget(size_label)

            desc_label = QLabel(f"({desc})")
            desc_label.setStyleSheet("color: gray;")
            row.addWidget(desc_label)

            row.addStretch()

            btn = QPushButton("Download")
            btn.setMinimumWidth(100)
            btn.clicked.connect(lambda checked, m=model_name: self._download_whisper(m))
            row.addWidget(btn)

            layout.addLayout(row)

        layout.addStretch()
        return widget

    def _create_translation_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)

        nllb_group = QGroupBox("NLLB-200 (Multilingual - All Language Pairs)")
        nllb_layout = QVBoxLayout(nllb_group)

        nllb_info = QLabel("NLLB supports translation between any pair of 200 languages.")
        nllb_info.setWordWrap(True)
        nllb_layout.addWidget(nllb_info)

        for model_name, (size, desc) in NLLB_MODELS.items():
            row = QHBoxLayout()
            label = QLabel(f"nllb-200-{model_name}")
            label.setMinimumWidth(150)
            row.addWidget(label)

            size_label = QLabel(f"{size}")
            size_label.setMinimumWidth(80)
            row.addWidget(size_label)

            desc_label = QLabel(f"({desc})")
            desc_label.setStyleSheet("color: gray;")
            row.addWidget(desc_label)

            row.addStretch()

            btn = QPushButton("Download")
            btn.setMinimumWidth(100)
            btn.clicked.connect(lambda checked, m=model_name: self._download_nllb(m))
            row.addWidget(btn)

            nllb_layout.addLayout(row)

        layout.addWidget(nllb_group)

        helsinki_group = QGroupBox("Helsinki-NLP (Language-Specific - Faster)")
        helsinki_layout = QVBoxLayout(helsinki_group)

        helsinki_info = QLabel(
            "Helsinki models are faster but only support specific language pairs."
        )
        helsinki_info.setWordWrap(True)
        helsinki_layout.addWidget(helsinki_info)

        self._helsinki_checkboxes = {}
        for pair, desc in HELSINKI_LANG_PAIRS:
            cb = QCheckBox(f"{desc} ({pair})")
            self._helsinki_checkboxes[pair] = cb
            helsinki_layout.addWidget(cb)

        btn_row = QHBoxLayout()
        btn_select_all = QPushButton("Select All")
        btn_select_all.clicked.connect(self._select_all_helsinki)
        btn_row.addWidget(btn_select_all)

        btn_clear = QPushButton("Clear Selection")
        btn_clear.clicked.connect(self._clear_helsinki_selection)
        btn_row.addWidget(btn_clear)

        btn_download = QPushButton("Download Selected")
        btn_download.clicked.connect(self._download_selected_helsinki)
        btn_row.addWidget(btn_download)

        btn_row.addStretch()
        helsinki_layout.addLayout(btn_row)

        layout.addWidget(helsinki_group)
        layout.addStretch()
        return widget

    def _create_tts_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)

        kokoro_group = QGroupBox("Kokoro (Neural TTS)")
        kokoro_layout = QVBoxLayout(kokoro_group)

        kokoro_info = QLabel("High-quality neural TTS. Supports English, Chinese, Japanese.")
        kokoro_info.setWordWrap(True)
        kokoro_layout.addWidget(kokoro_info)

        row = QHBoxLayout()
        label = QLabel("Kokoro-82M (~330 MB)")
        row.addWidget(label)
        row.addStretch()
        btn = QPushButton("Download")
        btn.clicked.connect(self._download_kokoro)
        row.addWidget(btn)
        kokoro_layout.addLayout(row)

        layout.addWidget(kokoro_group)

        edge_group = QGroupBox("Edge TTS (Cloud)")
        edge_layout = QVBoxLayout(edge_group)

        edge_info = QLabel(
            "Microsoft Azure TTS. No download required, uses cloud API.\nSupports all 11 target languages."
        )
        edge_info.setWordWrap(True)
        edge_layout.addWidget(edge_info)

        layout.addWidget(edge_group)

        fish_group = QGroupBox("Fish-Speech (Voice Cloning)")
        fish_layout = QVBoxLayout(fish_group)

        fish_info = QLabel("Supports 50+ emotion tags. Cloud API or local inference.")
        fish_info.setWordWrap(True)
        fish_layout.addWidget(fish_info)

        row = QHBoxLayout()
        label = QLabel("Fish-Speech-1.5 (~2 GB)")
        row.addWidget(label)
        row.addStretch()
        btn = QPushButton("Download")
        btn.clicked.connect(self._download_fish)
        row.addWidget(btn)
        fish_layout.addLayout(row)

        layout.addWidget(fish_group)

        if not _IS_MACOS:
            indextts_group = QGroupBox("IndexTTS (Voice Cloning)")
            indextts_layout = QVBoxLayout(indextts_group)

            indextts_info = QLabel("Emotional voice cloning. Not available on macOS.")
            indextts_info.setWordWrap(True)
            indextts_layout.addWidget(indextts_info)

            row = QHBoxLayout()
            label = QLabel("IndexTTS-v2 (~4 GB)")
            row.addWidget(label)
            row.addStretch()
            btn = QPushButton("Download")
            btn.clicked.connect(self._download_indextts)
            row.addWidget(btn)
            indextts_layout.addLayout(row)

            layout.addWidget(indextts_group)

        layout.addStretch()
        return widget

    def _create_batch_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)

        info_label = QLabel(
            "Select models to download in batch. Models will be downloaded one by one."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        whisper_group = QGroupBox("Whisper (Speech-to-Text)")
        whisper_layout = QVBoxLayout(whisper_group)

        self._whisper_checkboxes = {}
        for model_name, (size, desc) in WHISPER_MODELS.items():
            cb = QCheckBox(f"{model_name.capitalize()} {size} - {desc}")
            self._whisper_checkboxes[model_name] = cb
            whisper_layout.addWidget(cb)

        layout.addWidget(whisper_group)

        trans_group = QGroupBox("Translation Models")
        trans_layout = QVBoxLayout(trans_group)

        self._nllb_checkboxes = {}
        for model_name, (size, desc) in NLLB_MODELS.items():
            cb = QCheckBox(f"NLLB-200-{model_name} {size} - {desc}")
            self._nllb_checkboxes[model_name] = cb
            trans_layout.addWidget(cb)

        self._helsinki_batch_cb = QCheckBox("All Helsinki-NLP models (~50 models)")
        trans_layout.addWidget(self._helsinki_batch_cb)

        layout.addWidget(trans_group)

        tts_group = QGroupBox("TTS Models")
        tts_layout = QVBoxLayout(tts_group)

        self._kokoro_cb = QCheckBox("Kokoro-82M (~330 MB)")
        tts_layout.addWidget(self._kokoro_cb)

        if not _IS_MACOS:
            self._indextts_cb = QCheckBox("IndexTTS-v2 (~4 GB)")
            tts_layout.addWidget(self._indextts_cb)

        self._fish_cb = QCheckBox("Fish-Speech-1.5 (~2 GB)")
        tts_layout.addWidget(self._fish_cb)

        layout.addWidget(tts_group)

        btn_row = QHBoxLayout()
        btn_select_all = QPushButton("Select All")
        btn_select_all.clicked.connect(self._select_all_batch)
        btn_row.addWidget(btn_select_all)

        btn_clear = QPushButton("Clear Selection")
        btn_clear.clicked.connect(self._clear_batch_selection)
        btn_row.addWidget(btn_clear)

        btn_download = QPushButton("Download Selected")
        btn_download.setMinimumWidth(150)
        btn_download.clicked.connect(self._download_batch)
        btn_row.addWidget(btn_download)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        layout.addStretch()
        return widget

    def _select_all_helsinki(self):
        for cb in self._helsinki_checkboxes.values():
            cb.setChecked(True)

    def _clear_helsinki_selection(self):
        for cb in self._helsinki_checkboxes.values():
            cb.setChecked(False)

    def _select_all_batch(self):
        for cb in self._whisper_checkboxes.values():
            cb.setChecked(True)
        for cb in self._nllb_checkboxes.values():
            cb.setChecked(True)
        self._helsinki_batch_cb.setChecked(True)
        self._kokoro_cb.setChecked(True)
        self._fish_cb.setChecked(True)
        if not _IS_MACOS:
            self._indextts_cb.setChecked(True)

    def _clear_batch_selection(self):
        for cb in self._whisper_checkboxes.values():
            cb.setChecked(False)
        for cb in self._nllb_checkboxes.values():
            cb.setChecked(False)
        self._helsinki_batch_cb.setChecked(False)
        self._kokoro_cb.setChecked(False)
        self._fish_cb.setChecked(False)
        if not _IS_MACOS:
            self._indextts_cb.setChecked(False)

    def _download_whisper(self, size: str):
        self._start_download_thread(f"Whisper {size}", self._download_whisper_impl, size)

    def _download_whisper_impl(self, size: str):
        try:
            from faster_whisper import WhisperModel

            quant_text = self._whisper_quant_combo.currentText()
            if "int8" in quant_text:
                compute_type = "int8"
            elif "float16" in quant_text:
                compute_type = "float16"
            else:
                compute_type = "float32"

            self._update_download_status(f"Downloading Whisper {size} ({compute_type})...")
            WhisperModel(size, device="cpu", compute_type=compute_type)
            self._update_download_status(
                f"Whisper {size} ({compute_type}) downloaded successfully!"
            )
        except Exception as e:
            self._update_download_status(f"Error: {e}")

    def _download_nllb(self, size: str):
        self._start_download_thread(f"NLLB {size}", self._download_nllb_impl, size)

    def _download_nllb_impl(self, size: str):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            model_name = f"facebook/nllb-200-{size}"
            self._update_download_status(f"Downloading {model_name}...")
            AutoTokenizer.from_pretrained(model_name)
            AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self._update_download_status(f"NLLB {size} downloaded successfully!")
        except Exception as e:
            self._update_download_status(f"Error: {e}")

    def _download_selected_helsinki(self):
        selected = [pair for pair, cb in self._helsinki_checkboxes.items() if cb.isChecked()]
        if not selected:
            self._update_download_status("No models selected")
            return
        self._start_download_thread("Helsinki models", self._download_helsinki_impl, selected)

    def _download_helsinki_impl(self, pairs: list):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            for pair in pairs:
                model_name = f"Helsinki-NLP/opus-mt-{pair}"
                self._update_download_status(f"Downloading {model_name}...")
                try:
                    AutoTokenizer.from_pretrained(model_name)
                    AutoModelForSeq2SeqLM.from_pretrained(model_name)
                except Exception as e:
                    self._update_download_status(f"Skipping {pair}: {e}")
            self._update_download_status("Helsinki models downloaded!")
        except Exception as e:
            self._update_download_status(f"Error: {e}")

    def _download_kokoro(self):
        self._start_download_thread("Kokoro", self._download_kokoro_impl)

    def _download_kokoro_impl(self):
        try:
            from kokoro import KPipeline

            self._update_download_status("Downloading Kokoro model...")
            KPipeline(lang_code="a")
            self._update_download_status("Kokoro downloaded successfully!")
        except Exception as e:
            self._update_download_status(f"Error: {e}")

    def _download_indextts(self):
        self._update_download_status("IndexTTS requires manual download from GitHub")

    def _download_fish(self):
        self._update_download_status("Fish-Speech requires API key or manual setup")

    def _download_batch(self):
        queue = []

        for name, cb in self._whisper_checkboxes.items():
            if cb.isChecked():
                queue.append(("whisper", name))

        for name, cb in self._nllb_checkboxes.items():
            if cb.isChecked():
                queue.append(("nllb", name))

        if self._helsinki_batch_cb.isChecked():
            queue.append(("helsinki_all", None))

        if self._kokoro_cb.isChecked():
            queue.append(("kokoro", None))

        if not _IS_MACOS and hasattr(self, "_indextts_cb") and self._indextts_cb.isChecked():
            queue.append(("indextts", None))

        if self._fish_cb.isChecked():
            queue.append(("fish", None))

        if not queue:
            self._update_download_status("No models selected")
            return

        self._download_queue = queue
        self._start_download_thread("Batch download", self._download_batch_impl)

    def _download_batch_impl(self):
        total = len(self._download_queue)
        for i, (model_type, model_name) in enumerate(self._download_queue):
            self._update_download_status(f"Downloading {i + 1}/{total}: {model_type}...")
            try:
                if model_type == "whisper":
                    self._download_whisper_impl(model_name)
                elif model_type == "nllb":
                    self._download_nllb_impl(model_name)
                elif model_type == "helsinki_all":
                    all_pairs = [p for p, _ in HELSINKI_LANG_PAIRS]
                    self._download_helsinki_impl(all_pairs)
                elif model_type == "kokoro":
                    self._download_kokoro_impl()
            except Exception as e:
                self._update_download_status(f"Error with {model_type}: {e}")
        self._update_download_status("Batch download complete!")

    def _update_download_status(self, message: str):
        if hasattr(self, "_download_label"):
            self._download_label.setText(message)

    def _start_download_thread(self, name: str, func, *args):
        self._download_progress.setVisible(True)
        self._download_progress.setRange(0, 0)
        self._download_label.setText(f"Starting {name}...")
        threading.Thread(target=func, args=args, daemon=True).start()

    def sync_to_state(self):
        s = self._state

        sel = self._source_combo.currentText()
        if sel == self._i18n.auto_detect:
            s.sourceLang = "auto"
        else:
            code = self._extract_lang_code(sel)
            if code:
                s.sourceLang = code

        code = self._extract_lang_code(self._target_combo.currentText())
        if code:
            s.targetLang = code

        s.ttsEngine = self._engine_combo.currentText()
        s.voice = self._voice_combo.currentText()
        s.captureMode = "ptt" if self._ptt_radio.isChecked() else "vad"
        s.refAudioPath = self._ref_audio_edit.text()
        s.emotion = self._emotion_combo.currentText()
        s.translationModel = self._trans_model_combo.currentText()
        s.transDevice = self._trans_device_combo.currentText()
        s.ttsDevice = self._tts_device_combo.currentText()

        # History settings
        s.historyEnabled = self._history_enabled_cb.isChecked()
        s.saveOriginalAudio = self._save_original_audio_cb.isChecked()
        s.saveTranslatedAudio = self._save_translated_audio_cb.isChecked()

    @staticmethod
    def _extract_lang_code(text: str) -> str | None:
        for code, name in SUPPORTED_LANGUAGES:
            if f"{name} ({code})" == text:
                return code
        return None

    def _set_default_target(self):
        for i, (code, _name) in enumerate(SUPPORTED_LANGUAGES):
            if code == self._state.targetLang:
                self._target_combo.setCurrentIndex(i)
                break
        self._trans_device_combo.setCurrentText(self._state.transDevice)
        self._tts_device_combo.setCurrentText(self._state.ttsDevice)
        self._update_voices()

    def _on_target_changed(self):
        self._update_voices()

    def _on_engine_changed(self):
        self._update_voices()

    def _update_voices(self):
        engine = self._engine_combo.currentText()
        code = self._extract_lang_code(self._target_combo.currentText())
        target_lang = code if code else "en"

        try:
            voices = get_available_voices(target_lang, engine)
        except Exception:
            voices = []

        self._voice_combo.blockSignals(True)
        self._voice_combo.clear()
        if voices:
            self._voice_combo.addItems(voices)
            self._voice_combo.setCurrentIndex(0)
        self._voice_combo.blockSignals(False)

    def _on_preview_voice(self):
        self.sync_to_state()
        threading.Thread(target=self._preview_thread, daemon=True).start()

    def _preview_thread(self):
        try:
            from copytalker.audio.playback import AudioPlayer
            from copytalker.core.config import TTSConfig
            from copytalker.tts.base import get_tts_engine

            s = self._state
            tts_config = TTSConfig(
                engine=s.ttsEngine, voice=s.voice, language=s.targetLang, device=s.ttsDevice
            )
            if s.refAudioPath:
                tts_config.indextts_reference_audio = s.refAudioPath
                tts_config.fish_speech_reference_audio = s.refAudioPath
            engine = get_tts_engine(s.ttsEngine, tts_config)
            audio, sr = engine.synthesize("Hello, this is a voice preview.", s.targetLang)
            if len(audio) > 0:
                player = AudioPlayer(default_sample_rate=sr)
                player.play(audio, sr, blocking=True)
                player.close()
        except Exception as e:
            self._event_queue.put(("error", f"Voice preview failed: {e}"))

    def _on_browse_ref_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Audio",
            "",
            "Audio files (*.wav *.mp3 *.flac *.ogg);;All files (*)",
        )
        if path:
            self._ref_audio_edit.setText(path)

    def set_running_state(self, is_running: bool):
        pass

    def update_download_progress(self, message: str):
        self._download_label.setText(message)

    def on_download_finished(self):
        self._download_progress.setVisible(False)
        self._download_label.setText("Download complete!")

    def on_calibration_done(self, noise_level):
        pass

    def on_clone_test_done(self):
        pass

    def on_cache_info(self, text: str):
        pass

    def on_pipeline_stopped(self):
        pass
