"""
Unit tests for configuration module.
"""

import os
import pytest
from pathlib import Path

from copytalker.core.config import (
    AppConfig,
    AudioConfig,
    STTConfig,
    TranslationConfig,
    TTSConfig,
    CacheConfig,
    get_default_cache_dir,
    get_device,
)


class TestAudioConfig:
    """Tests for AudioConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = AudioConfig()
        
        assert config.sample_rate == 16000
        assert config.frame_duration_ms == 30
        assert config.vad_aggressiveness == 3
        assert config.channels == 1
    
    def test_frame_size_calculation(self):
        """Test frame size property calculation."""
        config = AudioConfig()
        expected = int(16000 * 30 / 1000)  # 480
        assert config.frame_size == expected
    
    def test_validate_sample_rate(self):
        """Test sample rate validation."""
        config = AudioConfig(sample_rate=44100)
        
        with pytest.raises(ValueError, match="Unsupported sample rate"):
            config.validate()
    
    def test_validate_vad_aggressiveness(self):
        """Test VAD aggressiveness validation."""
        config = AudioConfig(vad_aggressiveness=5)
        
        with pytest.raises(ValueError, match="VAD aggressiveness"):
            config.validate()
    
    def test_validate_frame_duration(self):
        """Test frame duration validation."""
        config = AudioConfig(frame_duration_ms=25)
        
        with pytest.raises(ValueError, match="Frame duration"):
            config.validate()


class TestSTTConfig:
    """Tests for STTConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = STTConfig()
        
        assert config.model_size == "small"
        assert config.beam_size == 5
        assert config.language == "auto"
    
    def test_auto_compute_type_cpu(self):
        """Test compute type auto-detection for CPU."""
        config = STTConfig(device="cpu")
        assert config.compute_type == "float32"
    
    def test_validate_model_size(self):
        """Test model size validation."""
        config = STTConfig(model_size="invalid")
        
        with pytest.raises(ValueError, match="Invalid model size"):
            config.validate()


class TestTranslationConfig:
    """Tests for TranslationConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = TranslationConfig()
        
        assert config.source_lang == "auto"
        assert config.target_lang == "en"
        assert config.max_length == 400
    
    def test_validate_max_length(self):
        """Test max_length validation."""
        config = TranslationConfig(max_length=0)
        
        with pytest.raises(ValueError, match="max_length"):
            config.validate()
        
        config2 = TranslationConfig(max_length=2000)
        
        with pytest.raises(ValueError, match="max_length"):
            config2.validate()


class TestTTSConfig:
    """Tests for TTSConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = TTSConfig()
        
        assert config.engine == "kokoro"
        assert config.speed == 1.0
    
    def test_validate_engine(self):
        """Test engine validation."""
        config = TTSConfig(engine="invalid")
        
        with pytest.raises(ValueError, match="Invalid TTS engine"):
            config.validate()
    
    def test_validate_speed(self):
        """Test speed validation."""
        config = TTSConfig(speed=0.1)
        
        with pytest.raises(ValueError, match="Speed"):
            config.validate()
        
        config2 = TTSConfig(speed=3.0)
        
        with pytest.raises(ValueError, match="Speed"):
            config2.validate()


class TestCacheConfig:
    """Tests for CacheConfig."""
    
    def test_default_cache_dir(self):
        """Test default cache directory."""
        config = CacheConfig()
        assert config.cache_dir == get_default_cache_dir()
    
    def test_subdirectories(self):
        """Test cache subdirectory properties."""
        config = CacheConfig()
        
        assert config.whisper_cache_dir == config.cache_dir / "whisper"
        assert config.translation_cache_dir == config.cache_dir / "translation"
        assert config.tts_cache_dir == config.cache_dir / "tts"


class TestAppConfig:
    """Tests for AppConfig."""
    
    def test_default_creation(self):
        """Test creating default AppConfig."""
        config = AppConfig()
        
        assert isinstance(config.audio, AudioConfig)
        assert isinstance(config.stt, STTConfig)
        assert isinstance(config.translation, TranslationConfig)
        assert isinstance(config.tts, TTSConfig)
        assert isinstance(config.cache, CacheConfig)
    
    def test_from_dict(self):
        """Test creating AppConfig from dictionary."""
        data = {
            "audio": {"sample_rate": 16000},
            "stt": {"model_size": "base"},
            "translation": {"target_lang": "zh"},
            "tts": {"engine": "edge-tts"},
            "debug": True,
        }
        
        config = AppConfig.from_dict(data)
        
        assert config.audio.sample_rate == 16000
        assert config.stt.model_size == "base"
        assert config.translation.target_lang == "zh"
        assert config.tts.engine == "edge-tts"
        assert config.debug is True
    
    def test_to_dict(self):
        """Test converting AppConfig to dictionary."""
        config = AppConfig()
        config.stt.model_size = "medium"
        
        data = config.to_dict()
        
        assert data["stt"]["model_size"] == "medium"
        assert "audio" in data
        assert "translation" in data
        assert "tts" in data
    
    def test_validate_all(self, app_config):
        """Test validating all configs."""
        # Should not raise
        app_config.validate()
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading config."""
        config_path = tmp_path / "config.yaml"
        
        config = AppConfig()
        config.stt.model_size = "medium"
        config.save(config_path)
        
        loaded = AppConfig.from_yaml(config_path)
        
        assert loaded.stt.model_size == "medium"


class TestEnvironmentVariables:
    """Tests for environment variable handling."""
    
    def test_cache_dir_from_env(self, monkeypatch, tmp_path):
        """Test cache directory from environment variable."""
        test_path = str(tmp_path / "custom_cache")
        monkeypatch.setenv("COPYTALKER_CACHE_DIR", test_path)
        
        cache_dir = get_default_cache_dir()
        assert cache_dir == Path(test_path)
    
    def test_device_from_env(self, monkeypatch):
        """Test device from environment variable."""
        monkeypatch.setenv("COPYTALKER_DEVICE", "cpu")
        
        device = get_device()
        assert device == "cpu"
