"""
IndexTTS engine - Bilibili's zero-shot voice cloning TTS.

Supports:
- Zero-shot voice cloning from a single reference audio clip
- Emotion control via audio prompt, emotion vector, or text description
- Precise millisecond-level duration control
- Chinese, English, and Japanese
- Polyphone disambiguation via mixed character-pinyin input
"""

import logging
import tempfile
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from copytalker.core.config import TTSConfig, get_default_cache_dir
from copytalker.core.exceptions import TTSError, TTSEngineNotAvailableError
from copytalker.core.types import AudioArray
from copytalker.tts.base import TTSEngineBase

logger = logging.getLogger(__name__)

# IndexTTS default sample rate (24kHz via BigVGAN2 vocoder)
INDEXTTS_SAMPLE_RATE = 24000

# Supported emotion names mapped to vector indices
# Vector order: [happy, angry, sad, fearful, surprised, disgusted, contemptuous, neutral]
INDEXTTS_EMOTIONS = {
    "happy": 0,
    "angry": 1,
    "sad": 2,
    "fearful": 3,
    "surprised": 4,
    "disgusted": 5,
    "contemptuous": 6,
    "neutral": 7,
}


class IndexTTS(TTSEngineBase):
    """
    IndexTTS engine - Zero-shot voice cloning with emotion and duration control.

    Developed by Bilibili. Supports Chinese, English, and Japanese.
    Uses a Conformer encoder + FSQ speech codec + BigVGAN2 decoder pipeline.

    Features:
    - Zero-shot voice cloning from ~5-10 second reference audio
    - 8 basic emotions with three control methods (audio, vector, text)
    - Precise duration control (millisecond-level)
    - Polyphone disambiguation via character-pinyin mixed input
    - Two generation modes: fixed-duration and free-form autoregressive
    """

    # HuggingFace model IDs
    HF_MODEL_V1 = "IndexTeam/IndexTTS"
    HF_MODEL_V2 = "IndexTeam/IndexTTS-2"

    def __init__(self, config: Optional[TTSConfig] = None):
        super().__init__(config)

        self._model = None
        self._is_available = None
        self._model_dir: Optional[Path] = None
        self._model_version: str = "v2"

        # Check for custom model path from config
        model_path = getattr(self.config, "indextts_model_path", None)
        if model_path:
            self._model_dir = Path(model_path)
        else:
            # Check common locations
            common_paths = [
                get_default_cache_dir() / "indextts",
                Path.home() / "index-tts" / "checkpoints",
                Path("checkpoints"),
            ]
            for path in common_paths:
                if path.exists() and (path / "config.yaml").exists():
                    self._model_dir = path
                    break

    @property
    def name(self) -> str:
        return "indextts"

    @property
    def is_available(self) -> bool:
        """Check if IndexTTS package and model are available."""
        if self._is_available is not None:
            return self._is_available

        try:
            # Try v2 first, then v1
            try:
                from indextts.infer_v2 import IndexTTS2  # noqa: F401

                self._model_version = "v2"
            except ImportError:
                from indextts.infer import IndexTTS as IndexTTSV1  # noqa: F401

                self._model_version = "v1"

            self._is_available = True
        except ImportError:
            logger.debug("IndexTTS package not installed")
            self._is_available = False
            return False

        # Check if model directory exists
        if self._model_dir is None or not self._model_dir.exists():
            logger.debug("IndexTTS model directory not found")
            self._is_available = False
            return False

        return self._is_available

    def _ensure_model(self) -> None:
        """Initialize the IndexTTS model."""
        if self._model is not None:
            return

        if not self.is_available:
            raise TTSEngineNotAvailableError("IndexTTS is not available")

        device = self.config.device

        # Force CPU if CUDA is not available
        try:
            import torch

            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
                logger.info("CUDA not available, using CPU for IndexTTS")
        except ImportError:
            device = "cpu"

        logger.info(
            f"Initializing IndexTTS {self._model_version} "
            f"(model_dir={self._model_dir}, device={device})"
        )

        try:
            cfg_path = str(self._model_dir / "config.yaml")

            if self._model_version == "v2":
                from indextts.infer_v2 import IndexTTS2

                self._model = IndexTTS2(
                    model_dir=str(self._model_dir),
                    cfg_path=cfg_path,
                )
            else:
                from indextts.infer import IndexTTS as IndexTTSV1

                self._model = IndexTTSV1(
                    model_dir=str(self._model_dir),
                    cfg_path=cfg_path,
                )

            logger.info(f"IndexTTS {self._model_version} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize IndexTTS: {e}")
            raise TTSError(f"IndexTTS initialization failed: {e}") from e

    def synthesize(
        self,
        text: str,
        language: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[AudioArray, int]:
        """
        Synthesize text to speech using IndexTTS.

        The ``voice`` parameter should be a path to a reference audio file
        for zero-shot voice cloning. If not provided, a default reference
        audio will be used if available.

        Args:
            text: Text to synthesize
            language: Target language code
            voice: Path to reference audio file for voice cloning
            speed: Speech speed multiplier (unused in IndexTTS, use duration control)

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if not text or not text.strip():
            return np.array([], dtype=np.float32), INDEXTTS_SAMPLE_RATE

        self._ensure_model()

        # Resolve reference audio
        ref_audio = self._resolve_reference_audio(voice)
        if ref_audio is None:
            raise TTSError(
                "IndexTTS requires a reference audio file for voice cloning. "
                "Pass a .wav file path as the 'voice' parameter."
            )

        logger.debug(f"IndexTTS synthesizing: '{text[:50]}...' (ref={ref_audio})")

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_path = tmp.name

            self._model.infer(
                spk_audio_prompt=str(ref_audio),
                text=text,
                output_path=output_path,
            )

            audio_array = self._read_wav(output_path)

            # Clean up
            import os

            try:
                os.unlink(output_path)
            except OSError:
                pass

            if len(audio_array) == 0:
                logger.warning("IndexTTS produced no audio")
                return np.array([], dtype=np.float32), INDEXTTS_SAMPLE_RATE

            logger.debug(f"IndexTTS generated {len(audio_array)} samples")
            return audio_array, INDEXTTS_SAMPLE_RATE

        except Exception as e:
            logger.error(f"IndexTTS synthesis error: {e}")
            raise TTSError(f"IndexTTS synthesis failed: {e}") from e

    def synthesize_with_emotion(
        self,
        text: str,
        reference_audio: str,
        *,
        emotion_audio: Optional[str] = None,
        emotion_vector: Optional[List[float]] = None,
        use_emotion_text: bool = False,
    ) -> Tuple[AudioArray, int]:
        """
        Synthesize with emotion control (IndexTTS v2 only).

        Three methods for controlling emotion:
        1. Provide an emotion reference audio clip
        2. Provide a numeric vector for 8 emotions
        3. Enable text-based emotion inference

        Args:
            text: Text to synthesize
            reference_audio: Path to speaker reference audio
            emotion_audio: Path to emotion reference audio
            emotion_vector: 8-element list [happy, angry, sad, fearful,
                           surprised, disgusted, contemptuous, neutral]
            use_emotion_text: Infer emotion from text content

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if self._model_version != "v2":
            raise TTSError("Emotion control requires IndexTTS v2")

        self._ensure_model()

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_path = tmp.name

            kwargs: Dict = {
                "spk_audio_prompt": str(reference_audio),
                "text": text,
                "output_path": output_path,
            }

            if emotion_audio:
                kwargs["emo_audio_prompt"] = str(emotion_audio)
            if emotion_vector:
                kwargs["emo_vector"] = emotion_vector
            if use_emotion_text:
                kwargs["use_emo_text"] = True

            self._model.infer(**kwargs)

            audio_array = self._read_wav(output_path)

            import os

            try:
                os.unlink(output_path)
            except OSError:
                pass

            return audio_array, INDEXTTS_SAMPLE_RATE

        except Exception as e:
            logger.error(f"IndexTTS emotion synthesis error: {e}")
            raise TTSError(f"IndexTTS emotion synthesis failed: {e}") from e

    def synthesize_with_duration(
        self,
        text: str,
        reference_audio: str,
        target_duration: float,
    ) -> Tuple[AudioArray, int]:
        """
        Synthesize with precise duration control (IndexTTS v2 only).

        Args:
            text: Text to synthesize
            reference_audio: Path to speaker reference audio
            target_duration: Target duration in seconds

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if self._model_version != "v2":
            raise TTSError("Duration control requires IndexTTS v2")

        self._ensure_model()

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_path = tmp.name

            self._model.infer(
                spk_audio_prompt=str(reference_audio),
                text=text,
                output_path=output_path,
                target_dur=target_duration,
            )

            audio_array = self._read_wav(output_path)

            import os

            try:
                os.unlink(output_path)
            except OSError:
                pass

            return audio_array, INDEXTTS_SAMPLE_RATE

        except Exception as e:
            logger.error(f"IndexTTS duration synthesis error: {e}")
            raise TTSError(f"IndexTTS duration synthesis failed: {e}") from e

    def _resolve_reference_audio(self, voice: Optional[str]) -> Optional[Path]:
        """Resolve the reference audio path for voice cloning."""
        # 1. Explicit voice parameter
        if voice is not None:
            path = Path(voice)
            if path.exists():
                return path
            # Check in model directory
            if self._model_dir:
                ref = self._model_dir / voice
                if ref.exists():
                    return ref
            # Check in voice_clones cache directory
            clones_dir = get_default_cache_dir() / "voice_clones"
            if clones_dir.exists():
                ref = clones_dir / voice
                if ref.exists():
                    return ref
                # Try adding .wav extension
                ref = clones_dir / f"{voice}.wav"
                if ref.exists():
                    return ref

        # 2. Config reference audio
        config_ref = getattr(self.config, "indextts_reference_audio", None)
        if config_ref:
            path = Path(config_ref)
            if path.exists():
                return path

        # 3. Look for default reference audio in model directory
        if self._model_dir:
            for candidate in ["reference.wav", "default.wav", "spk.wav"]:
                ref = self._model_dir / candidate
                if ref.exists():
                    return ref

        # 4. Check voice_clones for any available WAV
        clones_dir = get_default_cache_dir() / "voice_clones"
        if clones_dir.exists():
            wavs = sorted(clones_dir.glob("*.wav"))
            if wavs:
                return wavs[0]

        return None

    def _read_wav(self, wav_path: str) -> AudioArray:
        """Read WAV file to float32 numpy array."""
        try:
            with wave.open(wav_path, "rb") as wav_file:
                sample_width = wav_file.getsampwidth()
                n_channels = wav_file.getnchannels()
                n_frames = wav_file.getnframes()
                raw_data = wav_file.readframes(n_frames)
        except Exception:
            # Fallback: try soundfile
            try:
                import soundfile as sf

                data, sr = sf.read(wav_path, dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                return data
            except ImportError:
                raise TTSError(
                    "Cannot read output WAV file. "
                    "Install soundfile for better format support: pip install soundfile"
                )

        if sample_width == 2:
            dtype = np.int16
            max_val = 32768.0
        elif sample_width == 4:
            dtype = np.int32
            max_val = 2147483648.0
        else:
            dtype = np.uint8
            max_val = 128.0

        audio = np.frombuffer(raw_data, dtype=dtype)

        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)

        return audio.astype(np.float32) / max_val

    def get_available_voices(self, language: str) -> List[str]:
        """
        Get available voices (reference audio files) for a language.

        IndexTTS uses reference audio files for voice cloning,
        so this returns any .wav files found in the model directory.
        """
        voices: List[str] = []
        if self._model_dir:
            for wav_file in sorted(self._model_dir.glob("*.wav")):
                voices.append(str(wav_file))
        return voices

    @staticmethod
    def get_supported_emotions() -> Dict[str, int]:
        """Get supported emotion names and their vector indices."""
        return dict(INDEXTTS_EMOTIONS)

    @staticmethod
    def make_emotion_vector(emotion: str, intensity: float = 1.0) -> List[float]:
        """
        Create an emotion vector for a single named emotion.

        Args:
            emotion: Emotion name (happy, angry, sad, fearful,
                     surprised, disgusted, contemptuous, neutral)
            intensity: Emotion intensity (0.0 to 1.0)

        Returns:
            8-element emotion vector
        """
        if emotion not in INDEXTTS_EMOTIONS:
            raise ValueError(
                f"Unknown emotion: {emotion}. "
                f"Must be one of: {list(INDEXTTS_EMOTIONS.keys())}"
            )

        vector = [0.0] * 8
        vector[INDEXTTS_EMOTIONS[emotion]] = max(0.0, min(1.0, intensity))
        return vector

    def close(self) -> None:
        """Release resources."""
        self._model = None
        logger.debug("IndexTTS closed")
