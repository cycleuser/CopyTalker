"""
Integration tests for the full translation pipeline.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from copytalker.core.config import AppConfig
from copytalker.core.pipeline import TranslationPipeline


@pytest.mark.integration
class TestTranslationPipeline:
    """Integration tests for TranslationPipeline."""
    
    def test_initialization(self, app_config):
        """Test pipeline initialization."""
        pipeline = TranslationPipeline(app_config)
        
        assert pipeline.config == app_config
        assert pipeline.is_running is False
    
    def test_register_callback(self, app_config):
        """Test callback registration."""
        pipeline = TranslationPipeline(app_config)
        
        callback = Mock()
        pipeline.register_callback("transcription", callback)
        
        assert callback in pipeline._callbacks["transcription"]
    
    def test_register_invalid_callback_type(self, app_config):
        """Test registering callback with invalid type."""
        pipeline = TranslationPipeline(app_config)
        
        with pytest.raises(ValueError, match="Unknown event type"):
            pipeline.register_callback("invalid", Mock())
    
    def test_get_status(self, app_config):
        """Test getting pipeline status."""
        pipeline = TranslationPipeline(app_config)
        
        status = pipeline.get_status()
        
        assert "is_running" in status
        assert "source_lang" in status
        assert "target_lang" in status
        assert "tts_engine" in status
    
    @pytest.mark.timeout(10)
    @patch('copytalker.core.pipeline.AudioCapturer')
    @patch('copytalker.core.pipeline.ThreadSafeAudioPlayer')
    @patch('copytalker.core.pipeline.WhisperRecognizer')
    @patch('copytalker.core.pipeline.UnifiedTranslator')
    @patch('copytalker.core.pipeline.get_tts_engine')
    def test_start_stop(
        self,
        mock_tts,
        mock_translator,
        mock_recognizer,
        mock_player,
        mock_capturer,
        app_config,
    ):
        """Test starting and stopping pipeline."""
        # Setup mocks
        mock_capturer_instance = Mock()
        mock_capturer_instance.get_audio_segment.return_value = None
        mock_capturer.return_value = mock_capturer_instance
        
        mock_player_instance = Mock()
        mock_player.return_value = mock_player_instance
        
        mock_recognizer_instance = Mock()
        mock_recognizer.return_value = mock_recognizer_instance
        
        mock_translator_instance = Mock()
        mock_translator.return_value = mock_translator_instance
        
        mock_tts_instance = Mock()
        mock_tts.return_value = mock_tts_instance
        
        pipeline = TranslationPipeline(app_config)
        
        # Start
        pipeline.start()
        assert pipeline.is_running is True
        
        # Wait a bit
        time.sleep(0.2)
        
        # Stop
        pipeline.stop()
        assert pipeline.is_running is False
    
    @pytest.mark.timeout(10)
    def test_context_manager(self, app_config):
        """Test context manager usage."""
        with patch('copytalker.core.pipeline.AudioCapturer') as mock_cap:
            with patch('copytalker.core.pipeline.ThreadSafeAudioPlayer'):
                with patch('copytalker.core.pipeline.WhisperRecognizer'):
                    with patch('copytalker.core.pipeline.UnifiedTranslator'):
                        with patch('copytalker.core.pipeline.get_tts_engine'):
                            mock_cap_instance = Mock()
                            mock_cap_instance.get_audio_segment.return_value = None
                            mock_cap.return_value = mock_cap_instance
                            
                            with TranslationPipeline(app_config) as pipeline:
                                assert pipeline.is_running is True
                            
                            assert pipeline.is_running is False
    
    def test_callback_emission(self, app_config):
        """Test that callbacks are emitted correctly."""
        pipeline = TranslationPipeline(app_config)
        
        received_events = []
        
        def callback(event):
            received_events.append(event)
        
        pipeline.register_callback("status", callback)
        
        # Emit test event
        pipeline._emit_event("status", "Test status")
        
        assert len(received_events) == 1
        assert received_events[0].data == "Test status"
        assert received_events[0].event_type == "status"


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineIntegration:
    """Slow integration tests that may use real models."""
    
    @pytest.mark.skip(reason="Requires real models - run manually")
    def test_full_pipeline_with_models(self, sample_audio_speech, app_config):
        """Test full pipeline with real models."""
        # This test would use real models
        # Only run manually when models are available
        pass
