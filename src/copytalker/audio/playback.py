"""
Audio playback functionality.

Uses ``sounddevice`` by default.  Falls back to ``pyaudio`` when
``sounddevice`` is not available.
"""

import logging
import threading
from typing import Optional

import numpy as np

from copytalker.core.exceptions import AudioError
from copytalker.core.types import AudioArray

logger = logging.getLogger(__name__)


class AudioPlayer:
    """
    Audio playback with thread-safe access.

    Supports playing numpy arrays and manages audio resources.
    """

    def __init__(self, default_sample_rate: int = 22050):
        """
        Initialize audio player.

        Args:
            default_sample_rate: Default sample rate for playback
        """
        self._default_sample_rate = default_sample_rate
        self._lock = threading.Lock()
        self._is_playing = False

    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._is_playing

    def play(
        self,
        audio_data: AudioArray,
        sample_rate: Optional[int] = None,
        blocking: bool = True,
    ) -> None:
        """
        Play audio data.

        Args:
            audio_data: Audio data as numpy array (float32 or int16)
            sample_rate: Sample rate in Hz (uses default if None)
            blocking: If True, wait for playback to complete
        """
        if sample_rate is None:
            sample_rate = self._default_sample_rate

        if blocking:
            self._play_blocking(audio_data, sample_rate)
        else:
            thread = threading.Thread(
                target=self._play_blocking,
                args=(audio_data, sample_rate),
                daemon=True,
            )
            thread.start()

    def _play_blocking(self, audio_data: AudioArray, sample_rate: int) -> None:
        """Play audio synchronously using the best available backend."""
        with self._lock:
            self._is_playing = True
            try:
                self._play_sd(audio_data, sample_rate)
            except ImportError:
                self._play_pa(audio_data, sample_rate)
            except Exception as e:
                logger.error(f"Error during audio playback: {e}")
                raise AudioError(f"Playback failed: {e}") from e
            finally:
                self._is_playing = False

    # ------------------------------------------------------------------
    # sounddevice backend
    # ------------------------------------------------------------------

    @staticmethod
    def _play_sd(audio_data: AudioArray, sample_rate: int) -> None:
        """Play via *sounddevice*."""
        import sounddevice as sd

        # Ensure float32 in [-1, 1]
        if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = np.clip(audio_data.astype(np.float32), -1.0, 1.0)

        sd.play(audio_float, samplerate=sample_rate)
        sd.wait()
        logger.debug(f"Played {len(audio_data)} samples at {sample_rate} Hz (sounddevice)")

    # ------------------------------------------------------------------
    # pyaudio fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _play_pa(audio_data: AudioArray, sample_rate: int) -> None:
        """Play via *pyaudio*."""
        import pyaudio

        # Convert to int16
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)
        elif audio_data.dtype != np.int16:
            audio_int16 = audio_data.astype(np.int16)
        else:
            audio_int16 = audio_data

        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True,
            )
            try:
                stream.write(audio_int16.tobytes())
            finally:
                stream.stop_stream()
                stream.close()
            logger.debug(f"Played {len(audio_data)} samples at {sample_rate} Hz (pyaudio)")
        finally:
            pa.terminate()

    # ------------------------------------------------------------------

    def play_with_lock(
        self,
        audio_data: AudioArray,
        sample_rate: Optional[int] = None,
    ) -> threading.Lock:
        """
        Play audio and return the lock for synchronization.

        Useful for pausing other operations during playback.

        Args:
            audio_data: Audio data to play
            sample_rate: Sample rate in Hz

        Returns:
            The playback lock
        """
        self.play(audio_data, sample_rate, blocking=True)
        return self._lock

    def wait_for_completion(self, timeout: float = 30.0) -> bool:
        """
        Wait for current playback to complete.

        Args:
            timeout: Maximum time to wait

        Returns:
            True if playback completed, False if timeout
        """
        import time

        start = time.time()
        while self._is_playing:
            if time.time() - start > timeout:
                return False
            time.sleep(0.1)
        return True

    def close(self) -> None:
        """Release audio resources (no-op for sounddevice)."""
        logger.debug("Audio player closed")

    def __enter__(self) -> "AudioPlayer":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


class ThreadSafeAudioPlayer(AudioPlayer):
    """
    Audio player that prevents audio capture interference.

    Provides a lock that can be acquired to prevent capture during playback.
    """

    def __init__(self, default_sample_rate: int = 22050):
        super().__init__(default_sample_rate)
        self._playback_lock = threading.Lock()

    @property
    def playback_lock(self) -> threading.Lock:
        """Get the playback lock for capture synchronization."""
        return self._playback_lock

    def play(
        self,
        audio_data: AudioArray,
        sample_rate: Optional[int] = None,
        blocking: bool = True,
    ) -> None:
        """
        Play audio with playback lock.

        The playback_lock is held during playback to allow
        synchronization with audio capture.
        """
        with self._playback_lock:
            super().play(audio_data, sample_rate, blocking)
