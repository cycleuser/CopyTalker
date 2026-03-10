"""
Audio recording utilities for voice cloning reference audio.

Provides simple recording to WAV file.

Uses ``sounddevice`` by default.  Falls back to ``pyaudio`` if available.

This module uses lazy imports to avoid triggering heavy dependencies
at import time, which allows it to be imported even when those
dependencies are not installed.
"""

import logging
import threading
import time
import wave
from typing import Optional

import numpy as np

from copytalker.core.config import get_default_cache_dir

logger = logging.getLogger(__name__)

# Recording defaults
DEFAULT_RECORD_SAMPLE_RATE = 16000
DEFAULT_RECORD_CHANNELS = 1
DEFAULT_RECORD_CHUNK_SIZE = 1024


class VoiceRecorder:
    """
    Records audio from the microphone for voice cloning reference.

    Captures audio to a WAV file that can be used as a reference
    for IndexTTS or Fish-Speech voice cloning.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_RECORD_SAMPLE_RATE,
        channels: int = DEFAULT_RECORD_CHANNELS,
        chunk_size: int = DEFAULT_RECORD_CHUNK_SIZE,
    ):
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = chunk_size

        self._frames: list[bytes] = []
        self._is_recording = False
        self._stop_event = threading.Event()
        self._record_thread: Optional[threading.Thread] = None
        self._duration: float = 0.0

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def duration(self) -> float:
        """Approximate duration of recorded audio in seconds."""
        return self._duration

    def start(self) -> None:
        """Start recording in a background thread."""
        if self._is_recording:
            logger.warning("Already recording")
            return

        self._frames = []
        self._duration = 0.0
        self._stop_event.clear()

        self._record_thread = threading.Thread(
            target=self._record_loop,
            name="VoiceRecorderThread",
            daemon=True,
        )
        self._record_thread.start()
        self._is_recording = True
        logger.info("Voice recording started")

    def stop(self) -> None:
        """Stop recording."""
        if not self._is_recording:
            return

        self._stop_event.set()
        if self._record_thread and self._record_thread.is_alive():
            self._record_thread.join(timeout=3.0)

        self._is_recording = False
        logger.info(f"Voice recording stopped ({self._duration:.1f}s)")

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    def _record_loop(self) -> None:
        """Pick the best available backend and record."""
        try:
            import sounddevice  # noqa: F401

            self._record_loop_sd()
        except ImportError:
            try:
                import pyaudio  # noqa: F401

                self._record_loop_pa()
            except ImportError:
                logger.error(
                    "Neither sounddevice nor pyaudio is installed. "
                    "Install one of them: pip install sounddevice"
                )
                self._is_recording = False

    # ------------------------------------------------------------------
    # sounddevice backend
    # ------------------------------------------------------------------

    def _record_loop_sd(self) -> None:
        """Record using *sounddevice*."""
        import queue as _queue

        import sounddevice as sd

        frame_q: _queue.Queue[bytes] = _queue.Queue()

        def _callback(indata, frames, time_info, status):  # noqa: ARG001
            int16 = (indata[:, 0] * 32768).clip(-32768, 32767).astype(np.int16)
            frame_q.put(int16.tobytes())

        stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="float32",
            blocksize=self._chunk_size,
            callback=_callback,
        )

        start_time = time.time()
        stream.start()
        try:
            while not self._stop_event.is_set():
                try:
                    data = frame_q.get(timeout=0.5)
                    self._frames.append(data)
                    self._duration = time.time() - start_time
                except _queue.Empty:
                    self._duration = time.time() - start_time
        except Exception as e:
            logger.error(f"Recording error (sounddevice): {e}")
        finally:
            stream.stop()
            stream.close()
            self._is_recording = False

    # ------------------------------------------------------------------
    # pyaudio fallback
    # ------------------------------------------------------------------

    def _record_loop_pa(self) -> None:
        """Record using *pyaudio*."""
        import pyaudio

        pa = None
        stream = None
        try:
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=self._channels,
                rate=self._sample_rate,
                input=True,
                frames_per_buffer=self._chunk_size,
            )

            start_time = time.time()
            while not self._stop_event.is_set():
                try:
                    data = stream.read(self._chunk_size, exception_on_overflow=False)
                    self._frames.append(data)
                    self._duration = time.time() - start_time
                except Exception as e:
                    if not self._stop_event.is_set():
                        logger.error(f"Recording read error: {e}")
                    break

        except Exception as e:
            logger.error(f"Failed to open recording stream: {e}")
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            if pa:
                try:
                    pa.terminate()
                except Exception:
                    pass
            self._is_recording = False

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None, name: Optional[str] = None) -> str:
        """
        Save the recorded audio to a WAV file.

        Args:
            path: Full file path. If None, saves to voice_clones cache dir.
            name: File name (without extension). Used when path is None.

        Returns:
            Path to the saved WAV file.
        """
        if not self._frames:
            raise RuntimeError("No audio has been recorded")

        if path is None:
            save_dir = get_default_cache_dir() / "voice_clones"
            save_dir.mkdir(parents=True, exist_ok=True)
            if name is None:
                name = f"voice_{int(time.time())}"
            path = str(save_dir / f"{name}.wav")

        with wave.open(path, "wb") as wf:
            wf.setnchannels(self._channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self._sample_rate)
            wf.writeframes(b"".join(self._frames))

        logger.info(f"Voice recording saved: {path} ({self._duration:.1f}s)")
        return path

    def get_audio_array(self) -> np.ndarray:
        """
        Get the recorded audio as a float32 numpy array.

        Returns:
            Audio data normalised to [-1, 1].
        """
        if not self._frames:
            return np.array([], dtype=np.float32)

        raw = b"".join(self._frames)
        audio = np.frombuffer(raw, dtype=np.int16)
        return audio.astype(np.float32) / 32768.0

    def get_rms_level(self) -> float:
        """Get current RMS level of the recording (for a level meter)."""
        if not self._frames:
            return 0.0
        # Use the last frame
        last = np.frombuffer(self._frames[-1], dtype=np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(last**2)))

    def clear(self) -> None:
        """Discard the current recording."""
        self._frames = []
        self._duration = 0.0


def list_saved_voice_clones() -> list[dict]:
    """
    List all saved voice clone reference audio files.

    Returns:
        List of dicts with 'name', 'path', and 'size_kb' keys.
    """
    clones_dir = get_default_cache_dir() / "voice_clones"
    if not clones_dir.exists():
        return []

    results = []
    for wav_file in sorted(clones_dir.glob("*.wav")):
        results.append({
            "name": wav_file.stem,
            "path": str(wav_file),
            "size_kb": wav_file.stat().st_size / 1024,
        })
    return results
