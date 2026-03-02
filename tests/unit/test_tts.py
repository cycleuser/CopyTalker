"""
Unit tests for TTS module.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from copytalker.core.config import TTSConfig
from copytalker.tts.base import TTSEngineBase, get_tts_engine
from copytalker.tts.kokoro import KokoroTTS
from copytalker.tts.edge import EdgeTTS
from copytalker.tts.pyttsx3_engine import Pyttsx3TTS


class TestTTSEngineBase:
    """Tests for TTS engine base class."""
    
    def test_get_default_voice(self):
        """Test getting default voice."""
        # Create a concrete implementation for testing
        class TestEngine(TTSEngineBase):
            @property
            def name(self):
                return "test"
            
            @property
            def is_available(self):
                return True
            
            def synthesize(self, text, language, voice=None, speed=1.0):
                return np.zeros(100, dtype=np.float32), 22050
            
            def get_available_voices(self, language):
                return ["voice1", "voice2"]
        
        engine = TestEngine()
        default = engine.get_default_voice("en")
        
        assert default == "voice1"


class TestKokoroTTS:
    """Tests for KokoroTTS engine."""
    
    def test_initialization(self):
        """Test KokoroTTS initialization."""
        config = TTSConfig(engine="kokoro")
        tts = KokoroTTS(config)
        
        assert tts.name == "kokoro"
    
    def test_get_available_voices(self):
        """Test getting available Kokoro voices."""
        tts = KokoroTTS()
        
        voices = tts.get_available_voices("en")
        assert isinstance(voices, list)
        assert "af_heart" in voices
        
        voices_zh = tts.get_available_voices("zh")
        assert "zf_xiaobei" in voices_zh
    
    def test_synthesize_empty_text(self):
        """Test synthesis with empty text."""
        tts = KokoroTTS()
        
        audio, sr = tts.synthesize("", "en")
        
        assert len(audio) == 0


class TestEdgeTTS:
    """Tests for EdgeTTS engine."""
    
    def test_initialization(self):
        """Test EdgeTTS initialization."""
        config = TTSConfig(engine="edge-tts")
        tts = EdgeTTS(config)
        
        assert tts.name == "edge-tts"
    
    def test_is_available(self):
        """Test availability check."""
        tts = EdgeTTS()
        
        # Should check if edge_tts is installed
        assert isinstance(tts.is_available, bool)
    
    def test_get_available_voices(self):
        """Test getting Edge TTS voices."""
        tts = EdgeTTS()
        
        voices = tts.get_available_voices("en")
        assert isinstance(voices, list)
        
        if voices:  # Only if edge_tts is available
            assert any("Neural" in v for v in voices)
    
    def test_voice_name_resolution(self):
        """Test voice name resolution."""
        tts = EdgeTTS()
        
        # Full voice name should pass through
        full_name = tts._get_voice_name("en", "en-US-AriaNeural")
        assert full_name == "en-US-AriaNeural"


class TestPyttsx3TTS:
    """Tests for Pyttsx3TTS engine."""
    
    def test_initialization(self):
        """Test Pyttsx3TTS initialization."""
        tts = Pyttsx3TTS()
        
        assert tts.name == "pyttsx3"
    
    def test_is_available(self):
        """Test availability check."""
        tts = Pyttsx3TTS()
        
        # Should check if pyttsx3 is working
        assert isinstance(tts.is_available, bool)


class TestGetTTSEngine:
    """Tests for get_tts_engine factory function."""
    
    def test_get_kokoro_engine(self):
        """Test getting Kokoro engine."""
        engine = get_tts_engine("kokoro")
        
        assert isinstance(engine, KokoroTTS)
    
    def test_get_edge_engine(self):
        """Test getting Edge TTS engine."""
        engine = get_tts_engine("edge-tts")
        
        assert isinstance(engine, EdgeTTS)
    
    def test_get_pyttsx3_engine(self):
        """Test getting pyttsx3 engine."""
        engine = get_tts_engine("pyttsx3")
        
        assert isinstance(engine, Pyttsx3TTS)
    
    def test_get_invalid_engine(self):
        """Test getting invalid engine raises error."""
        with pytest.raises(ValueError, match="Unknown TTS engine"):
            get_tts_engine("invalid")
    
    def test_auto_selection(self):
        """Test auto engine selection."""
        # This should return some engine without error
        engine = get_tts_engine("auto")
        
        assert hasattr(engine, 'synthesize')


class TestAudioOutput:
    """Tests for TTS audio output."""
    
    def test_output_format(self):
        """Test that synthesis returns correct format."""
        # Mock a simple synthesis
        class MockEngine(TTSEngineBase):
            @property
            def name(self):
                return "mock"
            
            @property
            def is_available(self):
                return True
            
            def synthesize(self, text, language, voice=None, speed=1.0):
                return np.random.randn(1000).astype(np.float32), 22050
            
            def get_available_voices(self, language):
                return ["voice1"]
        
        engine = MockEngine()
        audio, sample_rate = engine.synthesize("Test", "en")
        
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert isinstance(sample_rate, int)
        assert sample_rate > 0
