"""
Fish-Speech (OpenAudio) TTS engine - LLM-based voice cloning and synthesis.

Supports:
- Zero-shot and few-shot voice cloning from 10-30 second reference audio
- 13 languages with cross-lingual synthesis
- 50+ emotion/expression markers via text tags
- Speech parameter control (speed, pitch, tone)
- Streaming audio generation
- Multiple output formats (WAV, MP3, PCM, Opus)
- Local inference and cloud API modes
"""

import io
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

# Fish-Speech default sample rate
FISH_SPEECH_SAMPLE_RATE = 44100

# Supported emotion/expression markers (text tags)
# These can be inserted into the input text like: "(happy) Hello world!"
FISH_SPEECH_EMOTION_TAGS = [
    "happy", "sad", "angry", "surprised", "excited", "whisper",
    "shouting", "crying", "laughing", "fearful", "disgusted",
    "calm", "serious", "gentle", "cheerful", "melancholic",
    "dramatic", "sarcastic", "nervous", "confident", "bored",
    "anxious", "proud", "shy", "romantic", "nostalgic",
    "hopeful", "desperate", "relieved", "confused", "amused",
    "contemptuous", "sympathetic", "determined", "playful",
    "mysterious", "soothing", "energetic", "solemn", "tender",
    "irritated", "enthusiastic", "indifferent", "passionate",
    "mocking", "pleading", "thoughtful", "warm", "cold",
    "authoritative",
]

# Language codes supported by Fish-Speech
FISH_SPEECH_LANGUAGES = [
    "en", "zh", "ja", "ko", "fr", "de", "it", "ar", "es", "pt", "hi", "ru",
]


class FishSpeechTTS(TTSEngineBase):
    """
    Fish-Speech (OpenAudio) TTS engine.

    A state-of-the-art LLM-based TTS engine using a Dual Autoregressive
    architecture (Slow + Fast Transformers) with Firefly-GAN vocoder.

    Features:
    - Zero-shot voice cloning from 10-30 seconds of reference audio
    - 13 languages with cross-lingual support
    - 50+ emotion/expression markers controllable via text tags
    - Speech speed and pitch control
    - Streaming audio generation
    - No G2P/phoneme alignment required
    - Local inference or cloud API mode
    """

    # HuggingFace model IDs
    HF_MODEL_S1_MINI = "fishaudio/fish-speech-1.5"

    def __init__(self, config: Optional[TTSConfig] = None):
        super().__init__(config)

        self._model = None
        self._is_available = None
        self._model_dir: Optional[Path] = None
        self._api_mode = False

        # Check for API key (cloud mode)
        api_key = getattr(self.config, "fish_speech_api_key", None)
        if api_key:
            self._api_mode = True
            self._api_key = api_key
            self._api_base_url = getattr(
                self.config, "fish_speech_api_url", "https://api.fish.audio"
            )

        # Check for local model path
        if not self._api_mode:
            model_path = getattr(self.config, "fish_speech_model_path", None)
            if model_path:
                self._model_dir = Path(model_path)
            else:
                common_paths = [
                    get_default_cache_dir() / "fish-speech",
                    Path.home() / "fish-speech" / "checkpoints",
                    Path("checkpoints"),
                ]
                for path in common_paths:
                    if path.exists():
                        self._model_dir = path
                        break

    @property
    def name(self) -> str:
        return "fish-speech"

    @property
    def is_available(self) -> bool:
        """Check if Fish-Speech is available (local or API mode)."""
        if self._is_available is not None:
            return self._is_available

        # API mode: check if SDK is available and key is set
        if self._api_mode:
            try:
                import fish_audio_sdk  # noqa: F401

                self._is_available = True
            except ImportError:
                # Fallback: we can use httpx/requests directly
                try:
                    import httpx  # noqa: F401

                    self._is_available = True
                except ImportError:
                    try:
                        import requests  # noqa: F401

                        self._is_available = True
                    except ImportError:
                        logger.debug(
                            "Fish-Speech API mode requires "
                            "fish-audio-sdk, httpx, or requests"
                        )
                        self._is_available = False
            return self._is_available

        # Local mode: check if fish_speech package is available
        try:
            from fish_speech.inference import TTSInference  # noqa: F401

            self._is_available = True
        except ImportError:
            # Try alternative import path
            try:
                from tools.llama.generate import generate  # noqa: F401

                self._is_available = True
            except ImportError:
                logger.debug("Fish-Speech package not installed for local inference")
                self._is_available = False
                return False

        if self._model_dir is None or not self._model_dir.exists():
            logger.debug("Fish-Speech model directory not found")
            self._is_available = False
            return False

        return self._is_available

    def _ensure_model(self) -> None:
        """Initialize the Fish-Speech model for local inference."""
        if self._api_mode or self._model is not None:
            return

        if not self.is_available:
            raise TTSEngineNotAvailableError("Fish-Speech is not available")

        device = self.config.device

        try:
            import torch

            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
                logger.info("CUDA not available, using CPU for Fish-Speech")
        except ImportError:
            device = "cpu"

        logger.info(
            f"Initializing Fish-Speech (model_dir={self._model_dir}, device={device})"
        )

        try:
            from fish_speech.inference import TTSInference

            self._model = TTSInference(
                checkpoint_path=str(self._model_dir),
                device=device,
            )
            logger.info("Fish-Speech initialized successfully")
        except ImportError:
            logger.warning(
                "fish_speech.inference not available, "
                "using API fallback if configured"
            )
            self._is_available = False
            raise TTSEngineNotAvailableError(
                "Fish-Speech local inference not available"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Fish-Speech: {e}")
            raise TTSError(f"Fish-Speech initialization failed: {e}") from e

    def synthesize(
        self,
        text: str,
        language: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[AudioArray, int]:
        """
        Synthesize text to speech using Fish-Speech.

        The ``voice`` parameter can be:
        - A path to a reference audio file for voice cloning
        - A voice/speaker ID (for API mode)
        - None for default voice

        Emotion tags can be embedded in the text, e.g.:
        "(happy) Hello, how are you today?"

        Args:
            text: Text to synthesize (may include emotion tags)
            language: Target language code
            voice: Reference audio path or voice ID
            speed: Speech speed multiplier (0.8 - 1.2)

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if not text or not text.strip():
            return np.array([], dtype=np.float32), FISH_SPEECH_SAMPLE_RATE

        if self._api_mode:
            return self._synthesize_api(text, language, voice, speed)
        else:
            return self._synthesize_local(text, language, voice, speed)

    def _synthesize_api(
        self,
        text: str,
        language: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[AudioArray, int]:
        """Synthesize using Fish Audio cloud API."""
        logger.debug(f"Fish-Speech API synthesizing: '{text[:50]}...'")

        try:
            # Try using the official SDK first
            try:
                from fish_audio_sdk import Session, TTSRequest

                session = Session(self._api_key)

                request_kwargs: Dict = {"text": text}
                if voice:
                    request_kwargs["reference_id"] = voice
                if speed != 1.0:
                    request_kwargs["speed"] = speed

                request = TTSRequest(**request_kwargs)
                result = session.tts(request)

                audio_bytes = b"".join(result)
                return self._decode_audio_bytes(audio_bytes), FISH_SPEECH_SAMPLE_RATE

            except ImportError:
                pass

            # Fallback: direct HTTP request
            return self._synthesize_api_http(text, voice, speed)

        except Exception as e:
            logger.error(f"Fish-Speech API synthesis error: {e}")
            raise TTSError(f"Fish-Speech API synthesis failed: {e}") from e

    def _synthesize_api_http(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[AudioArray, int]:
        """Synthesize using direct HTTP request to Fish Audio API."""
        import json

        url = f"{self._api_base_url}/v1/tts"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict = {"text": text}
        if voice:
            payload["reference_id"] = voice
        if speed != 1.0:
            payload["speed"] = speed

        try:
            import httpx

            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                audio_bytes = response.content
        except ImportError:
            import requests

            response = requests.post(
                url, headers=headers, json=json.dumps(payload), timeout=60
            )
            response.raise_for_status()
            audio_bytes = response.content

        return self._decode_audio_bytes(audio_bytes), FISH_SPEECH_SAMPLE_RATE

    def _synthesize_local(
        self,
        text: str,
        language: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[AudioArray, int]:
        """Synthesize using local model inference."""
        self._ensure_model()

        ref_audio = self._resolve_reference_audio(voice)

        logger.debug(
            f"Fish-Speech local synthesizing: '{text[:50]}...' "
            f"(ref={ref_audio})"
        )

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_path = tmp.name

            kwargs: Dict = {"text": text, "output_path": output_path}

            if ref_audio:
                kwargs["reference_audio"] = str(ref_audio)
            if speed != 1.0:
                kwargs["speed"] = speed

            self._model.synthesize(**kwargs)

            audio_array = self._read_wav(output_path)

            import os

            try:
                os.unlink(output_path)
            except OSError:
                pass

            if len(audio_array) == 0:
                logger.warning("Fish-Speech produced no audio")
                return np.array([], dtype=np.float32), FISH_SPEECH_SAMPLE_RATE

            logger.debug(f"Fish-Speech generated {len(audio_array)} samples")
            return audio_array, FISH_SPEECH_SAMPLE_RATE

        except Exception as e:
            logger.error(f"Fish-Speech local synthesis error: {e}")
            raise TTSError(f"Fish-Speech local synthesis failed: {e}") from e

    def synthesize_with_emotion(
        self,
        text: str,
        emotion: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        language: str = "en",
    ) -> Tuple[AudioArray, int]:
        """
        Synthesize with an emotion tag prepended to the text.

        Fish-Speech supports 50+ emotion markers as text tags.

        Args:
            text: Text to synthesize
            emotion: Emotion tag name (e.g., 'happy', 'sad', 'whisper')
            voice: Reference audio path or voice ID
            speed: Speech speed multiplier
            language: Target language code

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if emotion not in FISH_SPEECH_EMOTION_TAGS:
            logger.warning(
                f"Unknown emotion tag: {emotion}. "
                f"Using it anyway as Fish-Speech may support it."
            )

        tagged_text = f"({emotion}) {text}"
        return self.synthesize(tagged_text, language, voice, speed)

    def synthesize_streaming(
        self,
        text: str,
        language: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        chunk_size: int = 4096,
    ):
        """
        Generate audio in streaming mode (API mode only).

        Yields audio chunks as they become available.

        Args:
            text: Text to synthesize
            language: Target language code
            voice: Voice ID or reference audio path
            speed: Speech speed multiplier
            chunk_size: Size of each audio chunk in bytes

        Yields:
            bytes: Raw audio chunks (PCM format)
        """
        if not self._api_mode:
            raise TTSError("Streaming synthesis is only available in API mode")

        try:
            try:
                from fish_audio_sdk import Session, TTSRequest

                session = Session(self._api_key)

                request_kwargs: Dict = {"text": text}
                if voice:
                    request_kwargs["reference_id"] = voice
                if speed != 1.0:
                    request_kwargs["speed"] = speed

                request = TTSRequest(**request_kwargs)

                for chunk in session.tts(request):
                    yield chunk

            except ImportError:
                # Fallback: direct HTTP streaming
                url = f"{self._api_base_url}/v1/tts"
                headers = {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                }
                payload: Dict = {"text": text}
                if voice:
                    payload["reference_id"] = voice

                try:
                    import httpx

                    with httpx.Client(timeout=60.0) as client:
                        with client.stream(
                            "POST", url, headers=headers, json=payload
                        ) as response:
                            response.raise_for_status()
                            for chunk in response.iter_bytes(chunk_size):
                                yield chunk
                except ImportError:
                    # Non-streaming fallback via requests
                    import requests

                    response = requests.post(
                        url, headers=headers, json=payload, stream=True, timeout=60
                    )
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        yield chunk

        except Exception as e:
            logger.error(f"Fish-Speech streaming error: {e}")
            raise TTSError(f"Fish-Speech streaming failed: {e}") from e

    def register_speaker(
        self,
        name: str,
        reference_audio: str,
        reference_text: Optional[str] = None,
    ) -> Optional[str]:
        """
        Register a speaker for reuse (API mode).

        Args:
            name: Speaker name
            reference_audio: Path to reference audio file
            reference_text: Transcript of the reference audio (optional,
                           auto-transcribed if not provided)

        Returns:
            Speaker/voice ID for later use, or None if registration fails
        """
        if not self._api_mode:
            logger.info(
                "Speaker registration is only available in API mode. "
                "For local mode, just pass the reference audio path directly."
            )
            return None

        try:
            try:
                from fish_audio_sdk import Session

                session = Session(self._api_key)

                with open(reference_audio, "rb") as f:
                    audio_data = f.read()

                result = session.create_model(
                    title=name,
                    voices=[audio_data],
                    texts=[reference_text] if reference_text else None,
                )
                speaker_id = result.id if hasattr(result, "id") else str(result)
                logger.info(f"Registered speaker '{name}' with ID: {speaker_id}")
                return speaker_id

            except ImportError:
                logger.warning(
                    "fish-audio-sdk not installed, "
                    "speaker registration not available"
                )
                return None

        except Exception as e:
            logger.error(f"Speaker registration failed: {e}")
            return None

    def _resolve_reference_audio(self, voice: Optional[str]) -> Optional[Path]:
        """Resolve the reference audio path."""
        if voice is None:
            # Look for default reference in model directory
            if self._model_dir:
                for candidate in [
                    "reference.wav", "default.wav", "spk.wav", "speaker.wav",
                ]:
                    ref = self._model_dir / candidate
                    if ref.exists():
                        return ref
                # Also check speakers subdirectory
                speakers_dir = self._model_dir / "speakers"
                if speakers_dir.exists():
                    wavs = list(speakers_dir.glob("*.wav"))
                    if wavs:
                        return wavs[0]
            return None

        path = Path(voice)
        if path.exists():
            return path

        if self._model_dir:
            ref = self._model_dir / voice
            if ref.exists():
                return ref
            # Check speakers subdirectory
            ref = self._model_dir / "speakers" / voice
            if ref.exists():
                return ref

        return None

    def _decode_audio_bytes(self, audio_bytes: bytes) -> AudioArray:
        """Decode audio bytes (MP3/WAV/PCM) to numpy array."""
        # Try WAV first
        try:
            buf = io.BytesIO(audio_bytes)
            with wave.open(buf, "rb") as wav_file:
                sample_width = wav_file.getsampwidth()
                n_channels = wav_file.getnchannels()
                n_frames = wav_file.getnframes()
                raw_data = wav_file.readframes(n_frames)

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

        except Exception:
            pass

        # Try MP3 decoding via pydub
        try:
            from pydub import AudioSegment

            buf = io.BytesIO(audio_bytes)
            audio_seg = AudioSegment.from_file(buf)
            audio_seg = audio_seg.set_channels(1)

            samples = np.array(audio_seg.get_array_of_samples())
            if audio_seg.sample_width == 2:
                return samples.astype(np.float32) / 32768.0
            elif audio_seg.sample_width == 4:
                return samples.astype(np.float32) / 2147483648.0
            return samples.astype(np.float32) / 128.0

        except ImportError:
            pass

        # Try librosa as last resort
        try:
            import librosa

            buf = io.BytesIO(audio_bytes)
            audio, sr = librosa.load(buf, sr=FISH_SPEECH_SAMPLE_RATE)
            return audio.astype(np.float32)

        except ImportError:
            raise TTSError(
                "Cannot decode Fish-Speech audio output. "
                "Install pydub or librosa: pip install pydub librosa"
            )

    def _read_wav(self, wav_path: str) -> AudioArray:
        """Read WAV file to float32 numpy array."""
        try:
            with wave.open(wav_path, "rb") as wav_file:
                sample_width = wav_file.getsampwidth()
                n_channels = wav_file.getnchannels()
                n_frames = wav_file.getnframes()
                raw_data = wav_file.readframes(n_frames)
        except Exception:
            try:
                import soundfile as sf

                data, sr = sf.read(wav_path, dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                return data
            except ImportError:
                raise TTSError(
                    "Cannot read output WAV file. "
                    "Install soundfile: pip install soundfile"
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
        Get available voices for a language.

        In API mode, returns registered speaker IDs.
        In local mode, returns reference audio files from the model directory.
        """
        voices: List[str] = []

        if self._api_mode:
            try:
                from fish_audio_sdk import Session

                session = Session(self._api_key)
                models = session.list_models()
                for model in models:
                    voice_id = model.id if hasattr(model, "id") else str(model)
                    voices.append(voice_id)
            except Exception as e:
                logger.debug(f"Could not list API voices: {e}")
        else:
            if self._model_dir:
                # Check for reference audio files
                for wav_file in sorted(self._model_dir.glob("*.wav")):
                    voices.append(str(wav_file))
                # Check speakers subdirectory
                speakers_dir = self._model_dir / "speakers"
                if speakers_dir.exists():
                    for wav_file in sorted(speakers_dir.glob("*.wav")):
                        voices.append(str(wav_file))

        return voices

    @staticmethod
    def get_supported_emotions() -> List[str]:
        """Get list of supported emotion/expression tags."""
        return list(FISH_SPEECH_EMOTION_TAGS)

    @staticmethod
    def get_supported_languages() -> List[str]:
        """Get list of supported language codes."""
        return list(FISH_SPEECH_LANGUAGES)

    @staticmethod
    def format_emotion_text(text: str, emotion: str) -> str:
        """
        Format text with an emotion tag for Fish-Speech.

        Args:
            text: Original text
            emotion: Emotion tag (e.g., 'happy', 'whisper')

        Returns:
            Formatted text with emotion tag: "(happy) Hello!"
        """
        return f"({emotion}) {text}"

    def close(self) -> None:
        """Release resources."""
        self._model = None
        logger.debug("Fish-Speech TTS closed")
