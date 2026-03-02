"""
Unit tests for speech recognition module.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from copytalker.core.config import STTConfig
from copytalker.core.types import TranscriptionResult
from copytalker.speech.recognizer import WhisperRecognizer


class TestWhisperRecognizer:
    """Tests for WhisperRecognizer."""
    
    def test_initialization(self):
        """Test WhisperRecognizer initialization."""
        config = STTConfig(model_size="tiny", device="cpu")
        recognizer = WhisperRecognizer(config)
        
        assert recognizer.config == config
        assert recognizer.is_loaded() is False
    
    def test_default_config(self):
        """Test default configuration is applied."""
        recognizer = WhisperRecognizer()
        
        assert recognizer.config.model_size == "small"
    
    @patch('copytalker.speech.recognizer.WhisperModel')
    def test_transcribe(self, mock_whisper_class, sample_audio_mono):
        """Test transcription."""
        audio, sample_rate = sample_audio_mono
        
        # Setup mock
        mock_segment = Mock()
        mock_segment.text = "Hello world"
        
        mock_info = Mock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        
        mock_model = Mock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_class.return_value = mock_model
        
        recognizer = WhisperRecognizer(STTConfig(device="cpu"))
        result = recognizer.transcribe(audio, sample_rate)
        
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.confidence == 0.95
    
    @patch('copytalker.speech.recognizer.WhisperModel')
    def test_transcribe_with_language_hint(self, mock_whisper_class, sample_audio_mono):
        """Test transcription with language hint."""
        audio, sample_rate = sample_audio_mono
        
        mock_segment = Mock()
        mock_segment.text = "Test"
        
        mock_info = Mock()
        mock_info.language = "zh"
        mock_info.language_probability = 0.9
        
        mock_model = Mock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_class.return_value = mock_model
        
        recognizer = WhisperRecognizer(STTConfig(device="cpu", language="zh"))
        result = recognizer.transcribe(audio, sample_rate, language="zh")
        
        # Verify language was passed to transcribe
        call_args = mock_model.transcribe.call_args
        assert call_args.kwargs.get('language') == "zh"
    
    @patch('copytalker.speech.recognizer.WhisperModel')
    def test_detect_language(self, mock_whisper_class, sample_audio_mono):
        """Test language detection."""
        audio, sample_rate = sample_audio_mono
        
        mock_info = Mock()
        mock_info.language = "ja"
        mock_info.language_probability = 0.88
        
        mock_model = Mock()
        mock_model.transcribe.return_value = ([], mock_info)
        mock_whisper_class.return_value = mock_model
        
        recognizer = WhisperRecognizer(STTConfig(device="cpu"))
        lang, confidence = recognizer.detect_language(audio, sample_rate)
        
        assert lang == "ja"
        assert confidence == 0.88
    
    @patch('copytalker.speech.recognizer.WhisperModel')
    def test_unload_model(self, mock_whisper_class):
        """Test model unloading."""
        mock_model = Mock()
        mock_whisper_class.return_value = mock_model
        
        recognizer = WhisperRecognizer(STTConfig(device="cpu"))
        recognizer._ensure_model()  # Load model
        
        assert recognizer.is_loaded() is True
        
        recognizer.unload()
        
        assert recognizer.is_loaded() is False
    
    @patch('copytalker.speech.recognizer.WhisperModel')
    def test_context_manager(self, mock_whisper_class, sample_audio_mono):
        """Test context manager usage."""
        audio, sample_rate = sample_audio_mono
        
        mock_segment = Mock()
        mock_segment.text = "Test"
        
        mock_info = Mock()
        mock_info.language = "en"
        mock_info.language_probability = 0.9
        
        mock_model = Mock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_class.return_value = mock_model
        
        with WhisperRecognizer(STTConfig(device="cpu")) as recognizer:
            assert recognizer.is_loaded() is True
            result = recognizer.transcribe(audio, sample_rate)
            assert result.text == "Test"


class TestTranscriptionResult:
    """Tests for TranscriptionResult."""
    
    def test_is_empty_with_text(self):
        """Test is_empty with text."""
        result = TranscriptionResult(
            text="Hello",
            language="en",
            confidence=0.9,
        )
        assert result.is_empty() is False
    
    def test_is_empty_without_text(self):
        """Test is_empty without text."""
        result = TranscriptionResult(
            text="",
            language="en",
            confidence=0.0,
        )
        assert result.is_empty() is True
    
    def test_is_empty_whitespace(self):
        """Test is_empty with whitespace only."""
        result = TranscriptionResult(
            text="   ",
            language="en",
            confidence=0.0,
        )
        assert result.is_empty() is True
