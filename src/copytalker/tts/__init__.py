"""
Text-to-Speech module supporting multiple engines.
"""

from copytalker.tts.base import TTSEngineBase, get_tts_engine
from copytalker.tts.kokoro import KokoroTTS
from copytalker.tts.edge import EdgeTTS
from copytalker.tts.pyttsx3_engine import Pyttsx3TTS
from copytalker.tts.indextts import IndexTTS
from copytalker.tts.fish_speech import FishSpeechTTS

__all__ = [
    "TTSEngineBase",
    "KokoroTTS",
    "EdgeTTS",
    "Pyttsx3TTS",
    "IndexTTS",
    "FishSpeechTTS",
    "get_tts_engine",
]
