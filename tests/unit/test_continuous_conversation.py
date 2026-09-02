"""Tests for continuous conversation mode (Xiaoai-style turn-taking)."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from copytalker.core.config import AppConfig
from copytalker.core.conversation import ConversationFSM, ConversationState


# ---------------------------------------------------------------------------
# ConversationFSM
# ---------------------------------------------------------------------------

class TestConversationFSM:
    def test_initial_state_idle(self):
        assert ConversationFSM().state == ConversationState.IDLE

    def test_valid_transitions(self):
        fsm = ConversationFSM()
        assert fsm.transition_to(ConversationState.LISTENING)
        assert fsm.transition_to(ConversationState.PROCESSING)
        assert fsm.transition_to(ConversationState.SPEAKING)
        assert fsm.transition_to(ConversationState.LISTENING)

    def test_invalid_transition_rejected(self):
        fsm = ConversationFSM(ConversationState.LISTENING)
        # cannot jump LISTENING -> SPEAKING directly
        assert not fsm.transition_to(ConversationState.SPEAKING)
        assert fsm.state == ConversationState.LISTENING

    def test_same_state_is_noop_success(self):
        fsm = ConversationFSM(ConversationState.LISTENING)
        assert fsm.transition_to(ConversationState.LISTENING) is True

    def test_listener_notified(self):
        fsm = ConversationFSM()
        events = []
        fsm.add_listener(lambda s: events.append(s.name))
        fsm.transition_to(ConversationState.LISTENING)
        fsm.transition_to(ConversationState.PROCESSING)
        assert events == ["LISTENING", "PROCESSING"]

    def test_listener_exception_does_not_break_transition(self):
        fsm = ConversationFSM()
        fsm.add_listener(lambda s: (_ for _ in ()).throw(RuntimeError("boom")))
        # should not raise
        assert fsm.transition_to(ConversationState.LISTENING)
        assert fsm.state == ConversationState.LISTENING

    def test_reset_goes_to_idle(self):
        fsm = ConversationFSM(ConversationState.SPEAKING)
        fsm.reset()
        assert fsm.state == ConversationState.IDLE

    def test_barge_in_set_and_consume(self):
        fsm = ConversationFSM()
        fsm.request_barge_in()
        assert fsm.consume_barge_in()
        # consume does not clear; clear explicitly
        fsm.clear_barge_in()
        assert not fsm.consume_barge_in()

    def test_is_speaking_listening(self):
        assert ConversationFSM(ConversationState.SPEAKING).is_speaking
        assert ConversationFSM(ConversationState.LISTENING).is_listening
        assert not ConversationFSM(ConversationState.IDLE).is_speaking

    def test_can_accept_user_speech(self):
        assert ConversationFSM(ConversationState.LISTENING).can_accept_user_speech
        assert ConversationFSM(ConversationState.IDLE).can_accept_user_speech
        assert not ConversationFSM(ConversationState.SPEAKING).can_accept_user_speech
        assert not ConversationFSM(ConversationState.PROCESSING).can_accept_user_speech


# ---------------------------------------------------------------------------
# Pipeline continuous mode integration
# ---------------------------------------------------------------------------

class TestPipelineContinuousMode:
    def _make_pipeline(self, continuous: bool):
        from copytalker.core.pipeline import TranslationPipeline

        config = AppConfig()
        p = TranslationPipeline(config)
        p._continuous_mode = continuous
        return p

    def test_pipeline_has_fsm(self):
        p = self._make_pipeline(continuous=False)
        assert p.conversation_fsm is not None
        assert p.conversation_fsm.state == ConversationState.IDLE

    def test_start_continuous_enters_listening(self):
        from copytalker.core.pipeline import TranslationPipeline

        p = TranslationPipeline(AppConfig())
        with patch.object(p, "_initialize_components"), \
             patch("copytalker.core.pipeline.ThreadSafeAudioPlayer"), \
             patch("copytalker.core.pipeline.WhisperRecognizer"), \
             patch("copytalker.core.pipeline.UnifiedTranslator"), \
             patch("copytalker.core.pipeline.get_tts_engine"):
            p._audio_capturer = Mock()
            # Don't actually start threads — we only care about FSM state
            with patch.object(threading, "Thread"):
                p.start(capture_mode="vad", continuous=True)
            assert p.conversation_fsm.state == ConversationState.LISTENING
            assert p._continuous_mode is True
            p._stop_event.set()
            p._is_running = False

    def test_start_non_continuous_stays_idle(self):
        from copytalker.core.pipeline import TranslationPipeline

        p = TranslationPipeline(AppConfig())
        with patch.object(p, "_initialize_components"), \
             patch("copytalker.core.pipeline.ThreadSafeAudioPlayer"), \
             patch("copytalker.core.pipeline.WhisperRecognizer"), \
             patch("copytalker.core.pipeline.UnifiedTranslator"), \
             patch("copytalker.core.pipeline.get_tts_engine"):
            p._audio_capturer = Mock()
            with patch.object(threading, "Thread"):
                p.start(capture_mode="ptt", continuous=False)
            assert p.conversation_fsm.state == ConversationState.IDLE
            p._stop_event.set()
            p._is_running = False

    def test_stop_resets_fsm(self):
        from copytalker.core.pipeline import TranslationPipeline

        p = TranslationPipeline(AppConfig())
        p.conversation_fsm.transition_to(ConversationState.LISTENING)
        p.conversation_fsm.transition_to(ConversationState.PROCESSING)
        p._is_running = True  # fake running so stop() proceeds
        p._audio_capturer = Mock()
        p._audio_player = Mock()
        p._threads = {}
        p._history = None
        p.stop()
        assert p.conversation_fsm.state == ConversationState.IDLE


# ---------------------------------------------------------------------------
# Qt GUI continuous mode
# ---------------------------------------------------------------------------

class TestQtContinuousMode:
    def _state(self):
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        from copytalker.gui.qt.state import QtAppState

        return QtAppState()

    def test_default_continuous_on(self):
        s = self._state()
        assert s.continuousMode is True

    def test_settings_checkbox_syncs(self):
        import queue
        from unittest.mock import Mock

        from copytalker.gui.qt.views.settings import QtSettingsDialog

        s = self._state()
        mc = Mock()
        mc.refresh_cache_info.return_value = ""
        dlg = QtSettingsDialog(s, queue.Queue(), mc, parent=None)
        assert dlg._continuous_cb.isChecked() is True
        dlg._continuous_cb.setChecked(False)
        dlg.sync_to_state()
        assert s.continuousMode is False

    def test_build_config_propagates_continuous(self):
        # build_app_config_from_qt doesn't directly carry continuous, but
        # the pipeline reads it from state at start time. Verify the state
        # property round-trips.
        s = self._state()
        s.continuousMode = False
        assert s.continuousMode is False
        s.continuousMode = True
        assert s.continuousMode is True