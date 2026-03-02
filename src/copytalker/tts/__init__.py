"""
Text-to-Speech module supporting multiple engines.
"""

from copytalker.tts.base import TTSEngineBase, get_tts_engine
from copytalker.tts.kokoro import KokoroTTS
from copytalker.tts.edge import EdgeTTS
from copytalker.tts.pyttsx3_engine import Pyttsx3TTS

__all__ = [
    "TTSEngineBase",
    "KokoroTTS",
    "EdgeTTS",
    "Pyttsx3TTS",
    "get_tts_engine",
]
