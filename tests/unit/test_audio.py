"""
Unit tests for audio module.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np

from copytalker.audio.capture import AudioCapturer
from copytalker.audio.playback import AudioPlayer, ThreadSafeAudioPlayer
from copytalker.core.config import AudioConfig


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

    def test_start_stop(self):
        """Test starting and stopping capture."""
        mock_sd = MagicMock()
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            capturer = AudioCapturer()
            capturer.start()
            assert capturer.is_running is True

            time.sleep(0.1)

            capturer.stop()
            assert capturer.is_running is False

    def test_context_manager(self):
        """Test context manager usage."""
        mock_sd = MagicMock()
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
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

        mock_sd = MagicMock()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            player = AudioPlayer()
            player.play(audio, sample_rate)

            mock_sd.play.assert_called_once()
            mock_sd.wait.assert_called_once()

    def test_play_int16_audio(self):
        """Test playing int16 audio data."""
        audio = np.array([0, 1000, -1000, 500], dtype=np.int16)

        mock_sd = MagicMock()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            player = AudioPlayer()
            player.play(audio, 16000)

            mock_sd.play.assert_called_once()

    def test_close(self):
        """Test closing player (no-op for sounddevice)."""
        player = AudioPlayer()
        player.close()  # should not raise


class TestThreadSafeAudioPlayer:
    """Tests for ThreadSafeAudioPlayer."""

    def test_has_playback_lock(self):
        """Test ThreadSafeAudioPlayer has playback lock."""
        player = ThreadSafeAudioPlayer()

        assert hasattr(player, "playback_lock")
        assert player.playback_lock is not None

    def test_play_acquires_lock(self, sample_audio_mono):
        """Test play method acquires lock."""
        audio, sample_rate = sample_audio_mono

        mock_sd = MagicMock()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            player = ThreadSafeAudioPlayer()
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
