"""
Unit tests for audio module.
"""

import numpy as np
import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from copytalker.core.config import AudioConfig
from copytalker.audio.capture import AudioCapturer
from copytalker.audio.playback import AudioPlayer, ThreadSafeAudioPlayer


class TestAudioCapturer:
    """Tests for AudioCapturer."""
    
    def test_initialization(self):
        """Test AudioCapturer initialization."""
        config = AudioConfig()
        capturer = AudioCapturer(config)
        
        assert capturer.config == config
        assert capturer.is_running is False
        assert capturer.audio_queue is not None
    
    def test_config_defaults(self):
        """Test default configuration is applied."""
        capturer = AudioCapturer()
        
        assert capturer.config.sample_rate == 16000
        assert capturer.config.vad_aggressiveness == 3
    
    @patch('copytalker.audio.capture.pyaudio.PyAudio')
    def test_start_stop(self, mock_pyaudio):
        """Test starting and stopping capture."""
        # Setup mock
        mock_pa = Mock()
        mock_stream = Mock()
        mock_stream.read.return_value = np.zeros(480, dtype=np.int16).tobytes()
        mock_pa.open.return_value = mock_stream
        mock_pyaudio.return_value = mock_pa
        
        capturer = AudioCapturer()
        
        # Start
        capturer.start()
        assert capturer.is_running is True
        
        # Wait a bit for thread to start
        time.sleep(0.1)
        
        # Stop
        capturer.stop()
        assert capturer.is_running is False
    
    def test_context_manager(self):
        """Test context manager usage."""
        with patch('copytalker.audio.capture.pyaudio.PyAudio') as mock_pa:
            mock_instance = Mock()
            mock_stream = Mock()
            mock_stream.read.return_value = np.zeros(480, dtype=np.int16).tobytes()
            mock_instance.open.return_value = mock_stream
            mock_pa.return_value = mock_instance
            
            with AudioCapturer() as capturer:
                assert capturer.is_running is True
            
            assert capturer.is_running is False


class TestAudioPlayer:
    """Tests for AudioPlayer."""
    
    def test_initialization(self):
        """Test AudioPlayer initialization."""
        player = AudioPlayer(default_sample_rate=22050)
        
        assert player._default_sample_rate == 22050
        assert player.is_playing is False
    
    def test_play_float32_audio(self, sample_audio_mono):
        """Test playing float32 audio data."""
        audio, sample_rate = sample_audio_mono
        
        with patch('copytalker.audio.playback.pyaudio.PyAudio') as mock_pa:
            mock_instance = Mock()
            mock_stream = Mock()
            mock_instance.open.return_value = mock_stream
            mock_pa.return_value = mock_instance
            
            player = AudioPlayer()
            player.play(audio, sample_rate)
            
            # Verify stream was opened and written to
            mock_instance.open.assert_called_once()
            mock_stream.write.assert_called_once()
    
    def test_play_int16_audio(self):
        """Test playing int16 audio data."""
        audio = np.array([0, 1000, -1000, 500], dtype=np.int16)
        
        with patch('copytalker.audio.playback.pyaudio.PyAudio') as mock_pa:
            mock_instance = Mock()
            mock_stream = Mock()
            mock_instance.open.return_value = mock_stream
            mock_pa.return_value = mock_instance
            
            player = AudioPlayer()
            player.play(audio, 16000)
            
            mock_stream.write.assert_called_once()
    
    def test_close(self):
        """Test closing player."""
        with patch('copytalker.audio.playback.pyaudio.PyAudio') as mock_pa:
            mock_instance = Mock()
            mock_pa.return_value = mock_instance
            
            player = AudioPlayer()
            player._pyaudio = mock_instance
            player.close()
            
            mock_instance.terminate.assert_called_once()


class TestThreadSafeAudioPlayer:
    """Tests for ThreadSafeAudioPlayer."""
    
    def test_has_playback_lock(self):
        """Test ThreadSafeAudioPlayer has playback lock."""
        player = ThreadSafeAudioPlayer()
        
        assert hasattr(player, 'playback_lock')
        assert player.playback_lock is not None
    
    def test_play_acquires_lock(self, sample_audio_mono):
        """Test play method acquires lock."""
        audio, sample_rate = sample_audio_mono
        
        with patch('copytalker.audio.playback.pyaudio.PyAudio') as mock_pa:
            mock_instance = Mock()
            mock_stream = Mock()
            mock_instance.open.return_value = mock_stream
            mock_pa.return_value = mock_instance
            
            player = ThreadSafeAudioPlayer()
            
            # Play should acquire lock
            player.play(audio, sample_rate)
            
            # Lock should be released after play
            assert player.playback_lock.acquire(blocking=False)
            player.playback_lock.release()


class TestAudioConversion:
    """Tests for audio format conversions."""
    
    def test_float32_to_int16_conversion(self):
        """Test float32 to int16 conversion preserves range."""
        # Create float32 audio in [-1, 1] range
        float_audio = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        
        # Convert
        int_audio = (float_audio * 32767).astype(np.int16)
        
        # Verify range
        assert int_audio[0] == -32767
        assert int_audio[2] == 0
        assert int_audio[4] == 32767
    
    def test_clipping_out_of_range(self):
        """Test values outside [-1, 1] are clipped."""
        float_audio = np.array([-2.0, 2.0], dtype=np.float32)
        
        # Clip then convert
        clipped = np.clip(float_audio, -1.0, 1.0)
        int_audio = (clipped * 32767).astype(np.int16)
        
        assert int_audio[0] == -32767
        assert int_audio[1] == 32767
