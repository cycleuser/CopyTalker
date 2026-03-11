"""
Audio capture with Voice Activity Detection (VAD).

Uses ``sounddevice`` by default (ships pre-built binaries for macOS, Linux,
Windows).  Falls back to ``pyaudio`` if ``sounddevice`` is unavailable.
"""

import logging
import queue
import threading
import time
from collections import deque
from typing import Callable, Optional

import numpy as np
import webrtcvad

from copytalker.core.config import AudioConfig
from copytalker.core.types import AudioArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thin backend helpers – isolate the difference between sounddevice / pyaudio
# ---------------------------------------------------------------------------

def _open_input_stream_sd(config: AudioConfig):
    """Open an input stream via *sounddevice*."""
    import sounddevice as sd

    q: queue.Queue[bytes] = queue.Queue()

    def _callback(indata, frames, time_info, status):  # noqa: ARG001
        # indata is a numpy array (float32 by default); convert to int16 bytes
        # for webrtcvad compatibility.
        int16 = (indata[:, 0] * 32768).clip(-32768, 32767).astype(np.int16)
        q.put(int16.tobytes())

    stream = sd.InputStream(
        samplerate=config.sample_rate,
        channels=config.channels,
        dtype="float32",
        blocksize=config.frame_size,
        callback=_callback,
    )
    return stream, q


def _open_input_stream_pa(config: AudioConfig):
    """Open an input stream via *pyaudio*."""
    import pyaudio

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=config.channels,
        rate=config.sample_rate,
        input=True,
        frames_per_buffer=config.frame_size,
    )
    return pa, stream


class AudioCapturer:
    """
    Captures audio from microphone with Voice Activity Detection.

    Segments audio into speech chunks based on VAD and silence detection.
    """

    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        on_audio_segment: Optional[Callable[[AudioArray], None]] = None,
    ):
        """
        Initialize audio capturer.

        Args:
            config: Audio configuration (uses defaults if None)
            on_audio_segment: Callback for completed audio segments
        """
        self.config = config or AudioConfig()
        self.config.validate()

        self._on_audio_segment = on_audio_segment
        self._audio_queue: queue.Queue[AudioArray] = queue.Queue()

        # VAD setup
        self._vad = webrtcvad.Vad(self.config.vad_aggressiveness)

        # Audio buffer
        self._audio_buffer: deque[np.ndarray] = deque(maxlen=self.config.buffer_size)

        # Thread control
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None

        # State tracking
        self._is_running = False
        self._is_playing_back = False  # For echo cancellation

    @property
    def audio_queue(self) -> queue.Queue:
        """Get the audio segment queue."""
        return self._audio_queue

    @property
    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._is_running

    def start(self) -> None:
        """Start audio capture in a background thread."""
        if self._is_running:
            logger.warning("Audio capture is already running")
            return

        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="AudioCaptureThread",
            daemon=True,
        )
        self._capture_thread.start()
        self._is_running = True
        logger.info("Audio capture started")

    def stop(self, timeout: float = 2.0) -> None:
        """
        Stop audio capture.

        Args:
            timeout: Maximum time to wait for thread to stop
        """
        if not self._is_running:
            return

        self._stop_event.set()

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=timeout)
            if self._capture_thread.is_alive():
                logger.warning("Audio capture thread did not stop in time")

        self._is_running = False
        logger.info("Audio capture stopped")

    # ------------------------------------------------------------------
    # sounddevice-based capture loop
    # ------------------------------------------------------------------

    def _capture_loop_sd(self) -> None:
        """Capture loop using *sounddevice*."""

        stream, frame_q = _open_input_stream_sd(self.config)

        logger.info(
            f"Listening (sounddevice) sample_rate={self.config.sample_rate}, "
            f"frame_size={self.config.frame_size}"
        )

        voice_buffer: list = []
        is_speaking = False
        last_voice_time = time.time()

        stream.start()
        try:
            while not self._stop_event.is_set():
                try:
                    frame = frame_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                audio_data = np.frombuffer(frame, dtype=np.int16)

                is_voice = self._vad.is_speech(frame, self.config.sample_rate)
                current_time = time.time()

                if is_voice:
                    voice_buffer.append(audio_data)
                    is_speaking = True
                    last_voice_time = current_time
                else:
                    silence_duration = current_time - last_voice_time
                    if is_speaking and silence_duration > self.config.silence_threshold_s:
                        self._flush_voice_buffer(voice_buffer)
                        voice_buffer = []
                        is_speaking = False
                    elif is_speaking:
                        voice_buffer.append(audio_data)

                self._audio_buffer.append(audio_data)
        finally:
            stream.stop()
            stream.close()
            logger.debug("sounddevice stream closed")

    # ------------------------------------------------------------------
    # pyaudio-based capture loop (fallback)
    # ------------------------------------------------------------------

    def _capture_loop_pa(self) -> None:
        """Capture loop using *pyaudio* (legacy fallback)."""
        import pyaudio

        pa = None
        pa_stream = None
        try:
            pa = pyaudio.PyAudio()
            pa_stream = pa.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.frame_size,
            )
        except Exception as e:
            logger.error(f"Failed to open audio stream (pyaudio): {e}")
            self._is_running = False
            return

        logger.info(
            f"Listening (pyaudio) sample_rate={self.config.sample_rate}, "
            f"frame_size={self.config.frame_size}"
        )

        voice_buffer: list = []
        is_speaking = False
        last_voice_time = time.time()

        while not self._stop_event.is_set():
            try:
                frame = pa_stream.read(
                    self.config.frame_size,
                    exception_on_overflow=False,
                )
                audio_data = np.frombuffer(frame, dtype=np.int16)

                is_voice = self._vad.is_speech(frame, self.config.sample_rate)
                current_time = time.time()

                if is_voice:
                    voice_buffer.append(audio_data)
                    is_speaking = True
                    last_voice_time = current_time
                else:
                    silence_duration = current_time - last_voice_time
                    if is_speaking and silence_duration > self.config.silence_threshold_s:
                        self._flush_voice_buffer(voice_buffer)
                        voice_buffer = []
                        is_speaking = False
                    elif is_speaking:
                        voice_buffer.append(audio_data)

                self._audio_buffer.append(audio_data)
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"Audio capture error: {e}")
                break

        # Cleanup
        try:
            if pa_stream:
                pa_stream.stop_stream()
                pa_stream.close()
        except Exception as e:
            logger.debug(f"Error closing pyaudio stream: {e}")
        try:
            if pa:
                pa.terminate()
        except Exception as e:
            logger.debug(f"Error terminating PyAudio: {e}")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _flush_voice_buffer(self, voice_buffer: list) -> None:
        """Convert a collected voice buffer into a segment and enqueue it."""
        min_frames = int(
            self.config.min_speech_duration_s
            * self.config.sample_rate
            / self.config.frame_size
        )
        if len(voice_buffer) <= max(5, min_frames):
            return

        audio_segment = np.concatenate(voice_buffer)
        audio_segment = audio_segment.astype(np.float32) / 32768.0

        rms_energy = np.sqrt(np.mean(audio_segment**2))

        effective_threshold = max(
            self.config.min_energy_threshold,
            self.config.calibrated_noise_level * 2.0,
        )

        if rms_energy >= effective_threshold:
            self._audio_queue.put(audio_segment.copy())
            if self._on_audio_segment:
                self._on_audio_segment(audio_segment)
            logger.debug(
                f"Audio segment queued: {len(audio_segment)} samples, "
                f"energy={rms_energy:.4f}"
            )
        else:
            logger.debug(
                f"Audio segment filtered (low energy): "
                f"{rms_energy:.4f} < {effective_threshold:.4f}"
            )

    def _capture_loop(self) -> None:
        """Main capture loop – pick the best available backend."""
        try:
            import sounddevice  # noqa: F401

            self._capture_loop_sd()
        except ImportError:
            logger.info("sounddevice not available, falling back to pyaudio")
            try:
                self._capture_loop_pa()
            except ImportError:
                logger.error(
                    "Neither sounddevice nor pyaudio is installed. "
                    "Install one of them: pip install sounddevice"
                )
                self._is_running = False

    def get_audio_segment(self, timeout: float = 1.0) -> Optional[AudioArray]:
        """
        Get next audio segment from queue.

        Args:
            timeout: Maximum time to wait for segment

        Returns:
            Audio segment as float32 array, or None if timeout
        """
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def calibrate_noise(self, duration_s: float = 2.0) -> float:
        """
        Calibrate noise floor by measuring ambient noise.

        Should be called when there's no speech.

        Args:
            duration_s: Duration to measure noise

        Returns:
            Measured noise level (RMS)
        """
        logger.info(f"Calibrating noise floor for {duration_s}s...")

        try:
            return self._calibrate_sd(duration_s)
        except ImportError:
            pass

        try:
            return self._calibrate_pa(duration_s)
        except ImportError:
            logger.error(
                "Neither sounddevice nor pyaudio is installed for calibration."
            )
            return 0.0

    def _calibrate_sd(self, duration_s: float) -> float:
        """Calibrate using *sounddevice*."""
        import sounddevice as sd

        num_samples = int(duration_s * self.config.sample_rate)
        recording = sd.rec(
            num_samples,
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype="float32",
        )
        sd.wait()

        noise_level = float(np.sqrt(np.mean(recording**2)))
        self.config.calibrated_noise_level = noise_level
        logger.info(f"Noise calibration complete (sounddevice): {noise_level:.4f}")
        return noise_level

    def _calibrate_pa(self, duration_s: float) -> float:
        """Calibrate using *pyaudio*."""
        import pyaudio

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.frame_size,
        )

        samples = []
        num_frames = int(duration_s * self.config.sample_rate / self.config.frame_size)
        for _ in range(num_frames):
            frame = stream.read(self.config.frame_size, exception_on_overflow=False)
            audio_data = np.frombuffer(frame, dtype=np.int16)
            samples.append(audio_data)

        stream.stop_stream()
        stream.close()
        pa.terminate()

        all_audio = np.concatenate(samples).astype(np.float32) / 32768.0
        noise_level = float(np.sqrt(np.mean(all_audio**2)))

        self.config.calibrated_noise_level = noise_level
        logger.info(f"Noise calibration complete (pyaudio): {noise_level:.4f}")
        return noise_level

    def __enter__(self) -> "AudioCapturer":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
