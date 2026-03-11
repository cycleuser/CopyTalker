"""
Microsoft Edge TTS engine - Cloud-based fallback.
"""

import asyncio
import io
import logging
from typing import List, Optional, Tuple

import numpy as np

from copytalker.core.config import TTSConfig
from copytalker.core.constants import EDGE_TTS_VOICE_MAP
from copytalker.core.exceptions import TTSError, TTSEngineNotAvailableError
from copytalker.core.types import AudioArray
from copytalker.tts.base import TTSEngineBase

logger = logging.getLogger(__name__)


class EdgeTTS(TTSEngineBase):
    """
    Microsoft Edge TTS engine.
    
    Cloud-based TTS with good quality and many voices.
    Requires internet connection.
    """
    
    DEFAULT_SAMPLE_RATE = 24000
    
    def __init__(self, config: Optional[TTSConfig] = None):
        """
        Initialize Edge TTS engine.
        
        Args:
            config: TTS configuration
        """
        super().__init__(config)
        self._is_available = None
    
    @property
    def name(self) -> str:
        return "edge-tts"
    
    @property
    def is_available(self) -> bool:
        """Check if edge-tts package is installed."""
        if self._is_available is not None:
            return self._is_available
        
        try:
            import edge_tts
            self._is_available = True
        except ImportError:
            logger.debug("edge-tts package not installed")
            self._is_available = False
        
        return self._is_available
    
    def _get_voice_name(self, language: str, voice: Optional[str]) -> str:
        """Get the full voice name for Edge TTS."""
        if voice and "-" in voice:
            # Already a full voice name
            return voice
        
        voices = self.get_available_voices(language)
        
        if voice:
            # Find matching voice
            for v in voices:
                if voice.lower() in v.lower():
                    return v
        
        # Return first available voice
        return voices[0] if voices else f"{language}-US-AriaNeural"
    
    async def _synthesize_async(
        self,
        text: str,
        voice: str,
        rate: str = "+0%",
    ) -> bytes:
        """Async synthesis using edge-tts."""
        import edge_tts
        
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        
        audio_data = io.BytesIO()
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
        
        return audio_data.getvalue()
    
    def synthesize(
        self,
        text: str,
        language: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[AudioArray, int]:
        """
        Synthesize text to speech using Edge TTS.
        
        Args:
            text: Text to synthesize
            language: Target language code
            voice: Voice name
            speed: Speech speed multiplier
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if not self.is_available:
            raise TTSEngineNotAvailableError("Edge TTS is not available")
        
        if not text or not text.strip():
            return np.array([], dtype=np.float32), self.DEFAULT_SAMPLE_RATE
        
        voice_name = self._get_voice_name(language, voice)
        
        # Convert speed to percentage
        rate_percent = int((speed - 1.0) * 100)
        rate = f"+{rate_percent}%" if rate_percent >= 0 else f"{rate_percent}%"
        
        logger.debug(f"Edge TTS synthesizing: '{text[:50]}...' (voice={voice_name})")
        
        try:
            # Run async synthesis
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                mp3_data = loop.run_until_complete(
                    self._synthesize_async(text, voice_name, rate)
                )
            finally:
                loop.close()
            
            if not mp3_data:
                logger.warning("Edge TTS produced no audio")
                return np.array([], dtype=np.float32), self.DEFAULT_SAMPLE_RATE
            
            # Convert MP3 to numpy array
            audio_array = self._decode_mp3(mp3_data)
            
            logger.debug(f"Edge TTS generated {len(audio_array)} samples")
            
            return audio_array, self.DEFAULT_SAMPLE_RATE
            
        except Exception as e:
            logger.error(f"Edge TTS synthesis error: {e}")
            raise TTSError(f"Edge TTS synthesis failed: {e}") from e
    
    def _decode_mp3(self, mp3_data: bytes) -> AudioArray:
        """Decode MP3 data to numpy array, with multiple fallback decoders."""
        # Method 1: pydub (needs ffmpeg binary)
        try:
            from pydub import AudioSegment
            
            audio = AudioSegment.from_mp3(io.BytesIO(mp3_data))
            audio = audio.set_channels(1)
            
            samples = np.array(audio.get_array_of_samples())
            
            # Normalize to float32
            if audio.sample_width == 2:
                samples = samples.astype(np.float32) / 32768.0
            elif audio.sample_width == 4:
                samples = samples.astype(np.float32) / 2147483648.0
            
            return samples
            
        except Exception as e:
            logger.debug(f"pydub MP3 decode failed: {e}")
        
        # Method 2: librosa
        try:
            import librosa
            
            audio, sr = librosa.load(io.BytesIO(mp3_data), sr=self.DEFAULT_SAMPLE_RATE)
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.debug(f"librosa MP3 decode failed: {e}")
        
        # Method 3: ffmpeg subprocess (direct, works if ffmpeg binary is installed)
        try:
            import subprocess
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as mp3_f:
                mp3_f.write(mp3_data)
                mp3_path = mp3_f.name
            
            wav_path = mp3_path.replace('.mp3', '.wav')
            
            try:
                subprocess.run(
                    ['ffmpeg', '-y', '-i', mp3_path, '-ar', str(self.DEFAULT_SAMPLE_RATE),
                     '-ac', '1', '-f', 'wav', wav_path],
                    capture_output=True, timeout=30, check=True,
                )
                
                import wave
                with wave.open(wav_path, 'rb') as wf:
                    raw = wf.readframes(wf.getnframes())
                    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                
                return samples
            finally:
                for p in (mp3_path, wav_path):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
                        
        except Exception as e:
            logger.debug(f"ffmpeg subprocess MP3 decode failed: {e}")
        
        raise TTSError(
            "MP3 decoding failed. Install ffmpeg for audio processing:\n"
            "  macOS:  brew install ffmpeg\n"
            "  Linux:  sudo apt install ffmpeg\n"
            "  Or install pydub: pip install pydub"
        )
    
    def get_available_voices(self, language: str) -> List[str]:
        """Get available Edge TTS voices for a language."""
        return EDGE_TTS_VOICE_MAP.get(language, EDGE_TTS_VOICE_MAP.get("en", []))
