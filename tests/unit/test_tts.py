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
from copytalker.tts.indextts import IndexTTS, INDEXTTS_EMOTIONS, INDEXTTS_SAMPLE_RATE
from copytalker.tts.fish_speech import (
    FishSpeechTTS,
    FISH_SPEECH_EMOTION_TAGS,
    FISH_SPEECH_LANGUAGES,
    FISH_SPEECH_SAMPLE_RATE,
)


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
    
    def test_get_indextts_engine(self):
        """Test getting IndexTTS engine."""
        engine = get_tts_engine("indextts")
        
        assert isinstance(engine, IndexTTS)
    
    def test_get_fish_speech_engine(self):
        """Test getting Fish-Speech engine."""
        engine = get_tts_engine("fish-speech")
        
        assert isinstance(engine, FishSpeechTTS)
    
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


class TestIndexTTS:
    """Tests for IndexTTS engine."""
    
    def test_initialization(self):
        """Test IndexTTS initialization."""
        config = TTSConfig(engine="indextts")
        tts = IndexTTS(config)
        
        assert tts.name == "indextts"
    
    def test_is_available(self):
        """Test availability check."""
        tts = IndexTTS()
        
        # Should return bool (likely False without the model installed)
        assert isinstance(tts.is_available, bool)
    
    def test_synthesize_empty_text(self):
        """Test synthesis with empty text returns empty array."""
        tts = IndexTTS()
        
        audio, sr = tts.synthesize("", "en")
        
        assert len(audio) == 0
        assert sr == INDEXTTS_SAMPLE_RATE
    
    def test_get_supported_emotions(self):
        """Test getting supported emotion names."""
        emotions = IndexTTS.get_supported_emotions()
        
        assert isinstance(emotions, dict)
        assert "happy" in emotions
        assert "sad" in emotions
        assert "angry" in emotions
        assert "neutral" in emotions
        assert len(emotions) == 8
    
    def test_make_emotion_vector(self):
        """Test creating emotion vectors."""
        vec = IndexTTS.make_emotion_vector("happy", 0.8)
        
        assert len(vec) == 8
        assert vec[0] == 0.8  # happy is index 0
        assert sum(vec) == 0.8  # Only one emotion active
    
    def test_make_emotion_vector_invalid(self):
        """Test creating vector with invalid emotion raises error."""
        with pytest.raises(ValueError, match="Unknown emotion"):
            IndexTTS.make_emotion_vector("nonexistent")
    
    def test_make_emotion_vector_clamping(self):
        """Test that intensity is clamped to [0, 1]."""
        vec = IndexTTS.make_emotion_vector("sad", 1.5)
        assert vec[INDEXTTS_EMOTIONS["sad"]] == 1.0
        
        vec = IndexTTS.make_emotion_vector("sad", -0.5)
        assert vec[INDEXTTS_EMOTIONS["sad"]] == 0.0
    
    def test_emotion_constants(self):
        """Test emotion constants are consistent."""
        assert len(INDEXTTS_EMOTIONS) == 8
        expected = {
            "happy": 0, "angry": 1, "sad": 2, "fearful": 3,
            "surprised": 4, "disgusted": 5, "contemptuous": 6, "neutral": 7,
        }
        assert INDEXTTS_EMOTIONS == expected
    
    def test_close(self):
        """Test resource cleanup."""
        tts = IndexTTS()
        tts.close()
        
        assert tts._model is None


class TestFishSpeechTTS:
    """Tests for Fish-Speech TTS engine."""
    
    def test_initialization(self):
        """Test FishSpeechTTS initialization."""
        config = TTSConfig(engine="fish-speech")
        tts = FishSpeechTTS(config)
        
        assert tts.name == "fish-speech"
    
    def test_initialization_with_api_key(self):
        """Test FishSpeechTTS initialization with API key."""
        config = TTSConfig(
            engine="fish-speech",
            fish_speech_api_key="test_key_123",
        )
        tts = FishSpeechTTS(config)
        
        assert tts._api_mode is True
        assert tts._api_key == "test_key_123"
    
    def test_is_available(self):
        """Test availability check."""
        tts = FishSpeechTTS()
        
        assert isinstance(tts.is_available, bool)
    
    def test_synthesize_empty_text(self):
        """Test synthesis with empty text returns empty array."""
        tts = FishSpeechTTS()
        
        audio, sr = tts.synthesize("", "en")
        
        assert len(audio) == 0
        assert sr == FISH_SPEECH_SAMPLE_RATE
    
    def test_get_supported_emotions(self):
        """Test getting supported emotion tags."""
        emotions = FishSpeechTTS.get_supported_emotions()
        
        assert isinstance(emotions, list)
        assert "happy" in emotions
        assert "sad" in emotions
        assert "whisper" in emotions
        assert "excited" in emotions
        assert len(emotions) >= 50
    
    def test_get_supported_languages(self):
        """Test getting supported languages."""
        languages = FishSpeechTTS.get_supported_languages()
        
        assert isinstance(languages, list)
        assert "en" in languages
        assert "zh" in languages
        assert "ja" in languages
        assert len(languages) >= 12
    
    def test_format_emotion_text(self):
        """Test emotion text formatting."""
        result = FishSpeechTTS.format_emotion_text("Hello world", "happy")
        
        assert result == "(happy) Hello world"
    
    def test_format_emotion_text_whisper(self):
        """Test whisper emotion text formatting."""
        result = FishSpeechTTS.format_emotion_text("Secret message", "whisper")
        
        assert result == "(whisper) Secret message"
    
    def test_emotion_tags_constants(self):
        """Test emotion tags constants are properly defined."""
        assert isinstance(FISH_SPEECH_EMOTION_TAGS, list)
        assert len(FISH_SPEECH_EMOTION_TAGS) >= 50
        
        # Check some key emotions exist
        for emotion in ["happy", "sad", "angry", "whisper", "excited", "calm"]:
            assert emotion in FISH_SPEECH_EMOTION_TAGS
    
    def test_supported_languages_constants(self):
        """Test supported languages constants."""
        assert isinstance(FISH_SPEECH_LANGUAGES, list)
        assert len(FISH_SPEECH_LANGUAGES) >= 12
        
        for lang in ["en", "zh", "ja", "ko", "fr", "de"]:
            assert lang in FISH_SPEECH_LANGUAGES
    
    def test_close(self):
        """Test resource cleanup."""
        tts = FishSpeechTTS()
        tts.close()
        
        assert tts._model is None


class TestTTSConfigValidation:
    """Tests for TTS config validation with new engines."""
    
    def test_valid_indextts_config(self):
        """Test valid IndexTTS configuration."""
        config = TTSConfig(engine="indextts")
        config.validate()  # Should not raise
    
    def test_valid_fish_speech_config(self):
        """Test valid Fish-Speech configuration."""
        config = TTSConfig(engine="fish-speech")
        config.validate()  # Should not raise
    
    def test_invalid_engine_config(self):
        """Test invalid engine raises validation error."""
        config = TTSConfig(engine="nonexistent")
        
        with pytest.raises(ValueError, match="Invalid TTS engine"):
            config.validate()
    
    def test_all_valid_engines(self):
        """Test all valid engine names pass validation."""
        valid_engines = [
            "kokoro", "edge-tts", "pyttsx3", "indextts", "fish-speech", "auto",
        ]
        for engine_name in valid_engines:
            config = TTSConfig(engine=engine_name)
            config.validate()  # Should not raise
