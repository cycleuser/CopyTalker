"""
Pytest configuration and fixtures for CopyTalker tests.
"""

import wave
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest


@pytest.fixture
def sample_audio_mono():
    """
    Generate sample mono audio data.

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    sample_rate = 16000
    duration = 1.0  # seconds
    frequency = 440  # Hz (A4 note)

    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    return audio.astype(np.float32), sample_rate


@pytest.fixture
def sample_audio_speech():
    """
    Generate longer audio simulating speech duration.

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    sample_rate = 16000
    duration = 3.0  # seconds

    # Multiple frequencies to simulate speech-like audio
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 400 * t)
        + 0.1 * np.sin(2 * np.pi * 800 * t)
    )
    # Add some amplitude variation
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
    audio = audio * envelope

    return audio.astype(np.float32), sample_rate


@pytest.fixture
def sample_wav_file(sample_audio_mono, tmp_path):
    """
    Create a temporary WAV file.

    Returns:
        Path to the WAV file
    """
    audio, sample_rate = sample_audio_mono
    audio_int16 = (audio * 32767).astype(np.int16)

    wav_path = tmp_path / "test_audio.wav"

    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    return wav_path


@pytest.fixture
def mock_whisper_model():
    """
    Create a mock Whisper model.
    """
    mock_model = Mock()

    # Mock transcription result
    mock_segment = Mock()
    mock_segment.text = "Hello world"

    mock_info = Mock()
    mock_info.language = "en"
    mock_info.language_probability = 0.95

    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    return mock_model


@pytest.fixture
def mock_translation_model():
    """
    Create a mock translation model.
    """
    mock_model = Mock()
    mock_tokenizer = Mock()

    # Mock tokenizer
    mock_tokenizer.return_value = {
        "input_ids": Mock(to=Mock(return_value=Mock())),
        "attention_mask": Mock(to=Mock(return_value=Mock())),
    }
    mock_tokenizer.decode.return_value = "Translated text"

    # Mock model
    mock_model.generate.return_value = Mock()
    mock_model.device = "cpu"

    return mock_model, mock_tokenizer


@pytest.fixture
def mock_sounddevice():
    """
    Create a mock sounddevice module.
    """
    mock_sd = MagicMock()
    return mock_sd


@pytest.fixture
def app_config():
    """
    Create a test AppConfig.
    """
    from copytalker.core.config import AppConfig

    config = AppConfig()
    config.stt.model_size = "tiny"
    config.stt.device = "cpu"
    config.translation.target_lang = "zh"
    config.tts.engine = "pyttsx3"

    return config


@pytest.fixture
def temp_cache_dir(tmp_path):
    """
    Create a temporary cache directory.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


# Markers
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "audio: mark test as requiring audio hardware")
    config.addinivalue_line("markers", "integration: mark test as integration test")
