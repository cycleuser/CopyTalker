"""
Unit tests for conversation mode (context-aware translation).

Covers:
- OllamaTranslator prompt building with prior turns
- UnifiedTranslator passing context through to backends
- MT backends (Helsinki/NLLB) accepting and ignoring context
- ConversationHistory.get_context windowing
- HistoryConfig.context_window configuration
- Pipeline wiring history context into the translator
- CLI / API flags for conversation mode
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from copytalker.core.config import AppConfig, HistoryConfig
from copytalker.core.history import ConversationHistory
from copytalker.core.types import TranslationResult
from copytalker.translation.helsinki import HelsinkiTranslator
from copytalker.translation.nllb import NLLBTranslator
from copytalker.translation.ollama import OllamaTranslator
from copytalker.translation.translator import UnifiedTranslator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx_pair(src_text: str, tgt_text: str, src="en", tgt="zh") -> TranslationResult:
    return TranslationResult(
        original_text=src_text,
        translated_text=tgt_text,
        source_lang=src,
        target_lang=tgt,
    )


# ---------------------------------------------------------------------------
# Ollama prompt building
# ---------------------------------------------------------------------------

class TestOllamaPromptWithContext:
    def test_prompt_without_context_has_no_history_section(self):
        t = OllamaTranslator()
        prompt = t._build_prompt("Hello", "en", "zh")
        assert "Previous turns" not in prompt
        assert "Source (English): Hello" in prompt
        assert "Translation (Chinese):" in prompt

    def test_prompt_with_context_includes_prior_turns(self):
        t = OllamaTranslator()
        ctx = [
            _ctx_pair("Hi", "你好"),
            _ctx_pair("How are you?", "你好吗？"),
        ]
        prompt = t._build_prompt("Goodbye", "en", "zh", context=ctx)
        assert "Previous turns" in prompt
        assert "Source (English): Hi" in prompt
        assert "Translation (Chinese): 你好" in prompt
        assert "Source (English): How are you?" in prompt
        assert "Translation (Chinese): 你好" in prompt  # first turn translation present
        # The new source is still present
        assert "Source (English): Goodbye" in prompt

    def test_prompt_context_empty_list_equals_no_context(self):
        t = OllamaTranslator()
        assert t._build_prompt("Hi", "en", "zh", context=[]) == t._build_prompt("Hi", "en", "zh")

    def test_prompt_context_none_equals_no_context(self):
        t = OllamaTranslator()
        assert t._build_prompt("Hi", "en", "zh", context=None) == t._build_prompt("Hi", "en", "zh")


# ---------------------------------------------------------------------------
# Ollama translate end-to-end (mocked HTTP)
# ---------------------------------------------------------------------------

class TestOllamaTranslateWithContext:
    def _make_translator(self, response_text: str) -> OllamaTranslator:
        t = OllamaTranslator()
        t._available = True  # bypass availability check
        return t

    @patch("copytalker.translation.ollama.requests.post")
    def test_translate_passes_context_to_prompt(self, mock_post):
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "再见"}
        mock_post.return_value = mock_resp

        t = self._make_translator("再见")
        ctx = [_ctx_pair("Hello", "你好")]
        result = t.translate("Bye", "en", "zh", context=ctx)

        assert result.translated_text == "再见"
        # The prompt sent to Ollama must contain the prior turn
        sent_prompt = mock_post.call_args.kwargs["json"]["prompt"]
        assert "Previous turns" in sent_prompt
        assert "Source (English): Hello" in sent_prompt

    @patch("copytalker.translation.ollama.requests.post")
    def test_translate_without_context_omits_history(self, mock_post):
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "你好"}
        mock_post.return_value = mock_resp

        t = self._make_translator("你好")
        t.translate("Hi", "en", "zh")

        sent_prompt = mock_post.call_args.kwargs["json"]["prompt"]
        assert "Previous turns" not in sent_prompt


# ---------------------------------------------------------------------------
# UnifiedTranslator context forwarding
# ---------------------------------------------------------------------------

class TestUnifiedTranslatorContextForwarding:
    def test_context_forwarded_to_ollama(self):
        translator = UnifiedTranslator(preferred_model="ollama")
        # Force ollama availability without HTTP
        translator._ollama = Mock(spec=OllamaTranslator)
        translator._ollama.is_available.return_value = True
        translator._ollama.translate.return_value = TranslationResult(
            original_text="Bye", translated_text="再见",
            source_lang="en", target_lang="zh", model_used="ollama/x",
        )

        ctx = [_ctx_pair("Hi", "你好")]
        translator.translate("Bye", "en", "zh", context=ctx)

        translator._ollama.translate.assert_called_once_with("Bye", "en", "zh", ctx)

    def test_context_forwarded_to_helsinki(self):
        translator = UnifiedTranslator(preferred_model="helsinki")
        translator._helsinki = Mock(spec=HelsinkiTranslator)
        translator._helsinki.translate.return_value = TranslationResult(
            original_text="Bye", translated_text="再见",
            source_lang="en", target_lang="zh", model_used="helsinki/x",
        )

        ctx = [_ctx_pair("Hi", "你好")]
        translator.translate("Bye", "en", "zh", context=ctx)

        translator._helsinki.translate.assert_called_once_with("Bye", "en", "zh", ctx)

    def test_context_forwarded_to_nllb(self):
        translator = UnifiedTranslator(preferred_model="nllb")
        translator._nllb = Mock(spec=NLLBTranslator)
        translator._nllb.translate.return_value = TranslationResult(
            original_text="Bye", translated_text="再见",
            source_lang="en", target_lang="zh", model_used="nllb/x",
        )

        ctx = [_ctx_pair("Hi", "你好")]
        translator.translate("Bye", "en", "zh", context=ctx)

        translator._nllb.translate.assert_called_once_with("Bye", "en", "zh", ctx)

    def test_no_context_passes_none(self):
        translator = UnifiedTranslator(preferred_model="nllb")
        translator._nllb = Mock(spec=NLLBTranslator)
        translator._nllb.translate.return_value = TranslationResult(
            original_text="Bye", translated_text="再见",
            source_lang="en", target_lang="zh",
        )
        translator.translate("Bye", "en", "zh")
        _, kwargs = translator._nllb.translate.call_args
        assert kwargs.get("context") is None


# ---------------------------------------------------------------------------
# MT backends accept context without error
# ---------------------------------------------------------------------------

class TestMTBackendsAcceptContext:
    def test_helsinki_same_language_with_context(self):
        t = HelsinkiTranslator()
        ctx = [_ctx_pair("Hi", "你好")]
        result = t.translate("Hello", "en", "en", context=ctx)
        assert result.translated_text == "Hello"

    def test_nllb_same_language_with_context(self):
        t = NLLBTranslator()
        ctx = [_ctx_pair("Hi", "你好")]
        result = t.translate("Hello", "en", "en", context=ctx)
        assert result.translated_text == "Hello"


# ---------------------------------------------------------------------------
# ConversationHistory.get_context
# ---------------------------------------------------------------------------

class TestConversationHistoryContext:
    def _seed(self, tmp_path: Path) -> ConversationHistory:
        h = ConversationHistory(history_dir=tmp_path / "hist")
        h.start_session()
        # Entry 1 complete
        h.create_entry()
        h.add_transcription("Hello", "en")
        h.add_translation("你好", "zh")
        # Entry 2 complete
        h.create_entry()
        h.add_transcription("How are you?", "en")
        h.add_translation("你好吗？", "zh")
        # Entry 3 incomplete (no translation)
        h.create_entry()
        h.add_transcription("Goodbye", "en")
        return h

    def test_get_context_window_zero_returns_empty(self, tmp_path):
        h = self._seed(tmp_path)
        assert h.get_context(0) == []

    def test_get_context_negative_returns_empty(self, tmp_path):
        h = self._seed(tmp_path)
        assert h.get_context(-1) == []

    def test_get_context_returns_only_completed_entries(self, tmp_path):
        h = self._seed(tmp_path)
        ctx = h.get_context(10)
        assert len(ctx) == 2  # entry 3 has no translation
        assert ctx[0].original_text == "Hello"
        assert ctx[1].original_text == "How are you?"

    def test_get_context_windows_last_n(self, tmp_path):
        h = self._seed(tmp_path)
        ctx = h.get_context(1)
        assert len(ctx) == 1
        assert ctx[0].original_text == "How are you?"

    def test_get_context_smaller_than_available(self, tmp_path):
        h = self._seed(tmp_path)
        assert len(h.get_context(2)) == 2
        assert len(h.get_context(1)) == 1


# ---------------------------------------------------------------------------
# HistoryConfig.context_window
# ---------------------------------------------------------------------------

class TestHistoryConfigContextWindow:
    def test_default_is_zero(self):
        assert HistoryConfig().context_window == 0

    def test_round_trip_through_appconfig(self, tmp_path):
        c = AppConfig()
        c.history.context_window = 4
        d = c.to_dict()
        assert d["history"]["context_window"] == 4
        loaded = AppConfig.from_dict(d)
        assert loaded.history.context_window == 4


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------

class TestPipelineContextWiring:
    def test_pipeline_passes_context_when_window_enabled(self):
        """When context_window > 0 the pipeline fetches history context and
        forwards it to the translator."""
        from copytalker.core.config import AppConfig
        from copytalker.core.pipeline import TranslationPipeline

        config = AppConfig()
        config.history.context_window = 3
        config.history.enabled = True

        pipeline = TranslationPipeline(config)
        # Inject a fake history with one completed turn
        fake_history = Mock(spec=ConversationHistory)
        fake_history.get_context.return_value = []
        pipeline._history = fake_history

        pipeline._translator = Mock(spec=UnifiedTranslator)
        pipeline._translator.translate.return_value = TranslationResult(
            original_text="Hi", translated_text="你好",
            source_lang="en", target_lang="zh",
        )

        from copytalker.core.types import TranscriptionResult
        transcription = TranscriptionResult(text="Hi", language="en", confidence=0.9)

        # Simulate one iteration of the translation loop via the queue
        pipeline._text_queue.put((transcription, 1))
        # Stop after one item
        def stop_after(*a, **kw):
            pipeline._stop_event.set()
            return pipeline._translator.translate.return_value
        pipeline._translator.translate.side_effect = stop_after

        pipeline._stop_event.clear()
        pipeline._translation_loop()

        # The translator must have been called with a context kwarg
        _, kwargs = pipeline._translator.translate.call_args
        assert "context" in kwargs
        # And history.get_context was queried with the configured window
        fake_history.get_context.assert_called_once_with(3)

    def test_pipeline_passes_none_when_window_zero(self):
        from copytalker.core.config import AppConfig
        from copytalker.core.pipeline import TranslationPipeline
        from copytalker.core.types import TranscriptionResult

        config = AppConfig()
        config.history.context_window = 0
        config.history.enabled = True

        pipeline = TranslationPipeline(config)
        pipeline._history = Mock(spec=ConversationHistory)
        pipeline._translator = Mock(spec=UnifiedTranslator)
        pipeline._translator.translate.return_value = TranslationResult(
            original_text="Hi", translated_text="你好",
            source_lang="en", target_lang="zh",
        )

        transcription = TranscriptionResult(text="Hi", language="en", confidence=0.9)
        pipeline._text_queue.put((transcription, 1))
        pipeline._translator.translate.side_effect = lambda *a, **kw: (
            pipeline._stop_event.set(),
            pipeline._translator.translate.return_value,
        )[1]
        pipeline._stop_event.clear()
        pipeline._translation_loop()

        _, kwargs = pipeline._translator.translate.call_args
        assert kwargs.get("context") is None
        pipeline._history.get_context.assert_not_called()


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------

class TestAPIConversationFlag:
    def test_translate_api_accepts_conversation_kwargs(self):
        from copytalker.api import translate

        # We don't run the pipeline; just verify the signature accepts the
        # new params and builds config without crashing by short-circuiting
        # via a monkeypatched AppConfig + Pipeline that stops immediately.
        with patch("copytalker.core.pipeline.TranslationPipeline") as MockPipe:
            instance = MockPipe.return_value
            instance.is_running = False  # exits the wait loop
            result = translate(target="zh", conversation=True, context_window=2)

        assert result.success is True


# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------

class TestCLIConversationFlags:
    def test_translate_parser_has_conversation_flags(self):
        from copytalker.cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(
            ["translate", "-t", "zh", "--conversation"]
        )
        assert args.conversation is True
        assert args.context_window == 0

    def test_translate_parser_context_window_value(self):
        from copytalker.cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(
            ["translate", "-t", "zh", "--context-window", "7"]
        )
        assert args.context_window == 7
        assert args.conversation is False

    def test_cmd_translate_conversation_sets_window(self):
        from copytalker.cli.main import create_parser, cmd_translate

        parser = create_parser()
        args = parser.parse_args(
            ["translate", "-t", "zh", "--conversation"]
        )

        # Patch pipeline to avoid real run
        with patch("copytalker.core.pipeline.TranslationPipeline") as MockPipe:
            instance = MockPipe.return_value
            instance.is_running = False
            cmd_translate(args)

        # Inspect the AppConfig passed to TranslationPipeline
        config = MockPipe.call_args.args[0]
        assert config.history.context_window == 5  # default for --conversation

    def test_cmd_translate_context_window_overrides(self):
        from copytalker.cli.main import create_parser, cmd_translate

        parser = create_parser()
        args = parser.parse_args(
            ["translate", "-t", "zh", "--conversation", "--context-window", "3"]
        )

        with patch("copytalker.core.pipeline.TranslationPipeline") as MockPipe:
            instance = MockPipe.return_value
            instance.is_running = False
            cmd_translate(args)

        config = MockPipe.call_args.args[0]
        assert config.history.context_window == 3

    def test_cmd_translate_context_window_alone_enables_history(self):
        from copytalker.cli.main import create_parser, cmd_translate

        parser = create_parser()
        args = parser.parse_args(
            ["translate", "-t", "zh", "--no-history", "--context-window", "4"]
        )

        with patch("copytalker.core.pipeline.TranslationPipeline") as MockPipe:
            instance = MockPipe.return_value
            instance.is_running = False
            cmd_translate(args)

        config = MockPipe.call_args.args[0]
        assert config.history.context_window == 4
        # --no-history sets enabled False, but context_window>0 re-enables it
        assert config.history.enabled is True


# ---------------------------------------------------------------------------
# tools.py schema
# ---------------------------------------------------------------------------

class TestToolsSchemaConversation:
    def test_translate_tool_has_conversation_params(self):
        from copytalker.tools import TOOLS

        tool = next(t for t in TOOLS if t["function"]["name"] == "copytalker_translate")
        props = tool["function"]["parameters"]["properties"]
        assert "conversation" in props
        assert props["conversation"]["type"] == "boolean"
        assert "context_window" in props
        assert props["context_window"]["type"] == "integer"


# ---------------------------------------------------------------------------
# GUI state wiring
# ---------------------------------------------------------------------------

class TestTkGuiConversationWiring:
    """AppState <-> build_app_config conversation mapping (Tk)."""

    def test_app_state_has_conversation_fields(self):
        from copytalker.gui.state import AppState

        s = AppState()
        assert s.conversation_mode is False
        assert s.context_window == 0

    def test_build_app_config_maps_conversation_on(self):
        from copytalker.gui.state import AppState, build_app_config

        s = AppState()
        s.conversation_mode = True
        s.context_window = 5
        cfg = build_app_config(s)
        assert cfg.history.context_window == 5

    def test_build_app_config_maps_conversation_off(self):
        from copytalker.gui.state import AppState, build_app_config

        s = AppState()
        s.conversation_mode = False
        s.context_window = 5  # ignored when mode off
        cfg = build_app_config(s)
        assert cfg.history.context_window == 0

    def test_tk_settings_view_syncs_conversation(self):
        """The Tk SettingsView widgets propagate to AppState via sync_to_state."""
        import queue
        import tkinter as tk
        from unittest.mock import Mock

        from copytalker.gui.state import AppState
        from copytalker.gui.views.settings import SettingsView

        root = tk.Tk()
        root.withdraw()
        try:
            s = AppState()
            mc = Mock()
            mc.refresh_cache_info.return_value = ""
            view = SettingsView(
                root, s, queue.Queue(), mc,
                on_back=lambda: None, on_start=lambda: None, on_stop=lambda: None,
            )
            view.conversation_mode_var.set(True)
            view.context_window_var.set(3)
            view.sync_to_state()
            assert s.conversation_mode is True
            assert s.context_window == 3
        finally:
            root.destroy()


# ---------------------------------------------------------------------------
# Qt GUI state wiring
# ---------------------------------------------------------------------------

class TestQtGuiConversationWiring:
    """QtAppState <-> build_app_config_from_qt conversation mapping."""

    def _make_state(self):
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from copytalker.gui.qt.state import QtAppState

        return QtAppState()

    def test_qt_state_has_conversation_fields(self):
        s = self._make_state()
        assert s.conversationMode is False
        assert s.contextWindow == 0

    def test_qt_build_app_config_maps_conversation_on(self):
        from copytalker.gui.qt.state import build_app_config_from_qt

        s = self._make_state()
        s.conversationMode = True
        s.contextWindow = 6
        cfg = build_app_config_from_qt(s)
        assert cfg.history.context_window == 6

    def test_qt_build_app_config_maps_conversation_off(self):
        from copytalker.gui.qt.state import build_app_config_from_qt

        s = self._make_state()
        s.conversationMode = False
        s.contextWindow = 6
        cfg = build_app_config_from_qt(s)
        assert cfg.history.context_window == 0

    def test_qt_settings_dialog_syncs_conversation(self):
        """The Qt settings dialog widgets propagate to QtAppState."""
        import os
        import queue

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from unittest.mock import Mock

        QApplication.instance() or QApplication([])
        from copytalker.gui.qt.state import QtAppState
        from copytalker.gui.qt.views.settings import QtSettingsDialog

        s = QtAppState()
        mc = Mock()
        mc.refresh_cache_info.return_value = ""
        dlg = QtSettingsDialog(s, queue.Queue(), mc, parent=None)
        dlg._conversation_mode_cb.setChecked(True)
        dlg._context_window_spin.setValue(4)
        dlg.sync_to_state()
        assert s.conversationMode is True
        assert s.contextWindow == 4