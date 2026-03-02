"""
Audio capture with Voice Activity Detection (VAD).
"""

import logging
import queue
import threading
import time
from collections import deque
from typing import Optional, Callable, Deque

import numpy as np
import pyaudio
import webrtcvad

from copytalker.core.config import AudioConfig
from copytalker.core.exceptions import AudioError
from copytalker.core.types import AudioArray

logger = logging.getLogger(__name__)


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
        self._audio_buffer: Deque[np.ndarray] = deque(maxlen=self.config.buffer_size)
        
        # Thread control
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._pyaudio: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        
        # State tracking
        self._is_running = False
        
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
    
    def _capture_loop(self) -> None:
        """Main capture loop running in background thread."""
        try:
            self._pyaudio = pyaudio.PyAudio()
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.frame_size,
            )
        except Exception as e:
            logger.error(f"Failed to open audio stream: {e}")
            self._is_running = False
            return
        
        logger.info(f"Listening for audio input (sample_rate={self.config.sample_rate}, "
                   f"frame_size={self.config.frame_size})")
        
        voice_buffer: list = []
        is_speaking = False
        last_voice_time = time.time()
        
        while not self._stop_event.is_set():
            try:
                frame = self._stream.read(
                    self.config.frame_size,
                    exception_on_overflow=False,
                )
                audio_data = np.frombuffer(frame, dtype=np.int16)
                
                # Check for voice activity
                is_voice = self._vad.is_speech(frame, self.config.sample_rate)
                
                current_time = time.time()
                
                if is_voice:
                    voice_buffer.append(audio_data)
                    is_speaking = True
                    last_voice_time = current_time
                else:
                    silence_duration = current_time - last_voice_time
                    
                    if is_speaking and silence_duration > self.config.silence_threshold_s:
                        # End of speech segment
                        min_frames = int(self.config.min_speech_duration_s * self.config.sample_rate / self.config.frame_size)
                        if len(voice_buffer) > max(5, min_frames):
                            audio_segment = np.concatenate(voice_buffer)
                            # Convert to float32 normalized
                            audio_segment = audio_segment.astype(np.float32) / 32768.0
                            
                            # Calculate RMS energy
                            rms_energy = np.sqrt(np.mean(audio_segment ** 2))
                            
                            # Filter by energy threshold
                            effective_threshold = max(
                                self.config.min_energy_threshold,
                                self.config.calibrated_noise_level * 2.0  # 2x noise floor
                            )
                            
                            if rms_energy >= effective_threshold:
                                # Put in queue
                                self._audio_queue.put(audio_segment.copy())
                                
                                # Call callback if set
                                if self._on_audio_segment:
                                    self._on_audio_segment(audio_segment)
                                
                                logger.debug(f"Audio segment queued: {len(audio_segment)} samples, energy={rms_energy:.4f}")
                            else:
                                logger.debug(f"Audio segment filtered (low energy): {rms_energy:.4f} < {effective_threshold:.4f}")
                        
                        voice_buffer = []
                        is_speaking = False
                    elif is_speaking:
                        # Still in speech, keep collecting
                        voice_buffer.append(audio_data)
                
                # Update rolling buffer
                self._audio_buffer.append(audio_data)
                
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"Audio capture error: {e}")
                break
        
        # Cleanup
        self._cleanup()
    
    def _cleanup(self) -> None:
        """Clean up audio resources."""
        try:
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
                self._stream = None
        except Exception as e:
            logger.debug(f"Error closing stream: {e}")
        
        try:
            if self._pyaudio:
                self._pyaudio.terminate()
                self._pyaudio = None
        except Exception as e:
            logger.debug(f"Error terminating PyAudio: {e}")
        
        logger.debug("Audio resources cleaned up")
    
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
            
            # Calculate RMS of all samples
            all_audio = np.concatenate(samples).astype(np.float32) / 32768.0
            noise_level = np.sqrt(np.mean(all_audio ** 2))
            
            # Update config
            self.config.calibrated_noise_level = noise_level
            
            logger.info(f"Noise calibration complete: {noise_level:.4f}")
            return noise_level
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return 0.0
    
    def __enter__(self) -> "AudioCapturer":
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
