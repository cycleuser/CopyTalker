"""
CopyTalker - OpenAI function-calling tool definitions.

Provides TOOLS list and dispatch() for LLM agent integration.
"""

from __future__ import annotations

import json
from typing import Any

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "copytalker_translate",
            "description": (
                "Start a real-time speech-to-speech translation session. "
                "Listens to audio input, transcribes with Whisper, translates, "
                "and speaks the translation using TTS."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Target language code (e.g. 'en', 'zh', 'ja', 'ko').",
                    },
                    "source": {
                        "type": "string",
                        "description": "Source language code, or 'auto' for auto-detect.",
                        "default": "auto",
                    },
                    "voice": {
                        "type": "string",
                        "description": (
                            "TTS voice name (Kokoro/Edge-TTS) or path to reference "
                            "audio file (IndexTTS/Fish-Speech voice cloning)."
                        ),
                    },
                    "tts_engine": {
                        "type": "string",
                        "enum": [
                            "kokoro", "edge-tts", "pyttsx3",
                            "indextts", "fish-speech", "auto",
                        ],
                        "description": "TTS engine to use.",
                        "default": "auto",
                    },
                    "whisper_model": {
                        "type": "string",
                        "enum": ["tiny", "base", "small", "medium", "large"],
                        "description": "Whisper model size.",
                        "default": "small",
                    },
                    "device": {
                        "type": "string",
                        "enum": ["cpu", "cuda", "auto"],
                        "description": "Compute device.",
                        "default": "auto",
                    },
                    "duration": {
                        "type": "number",
                        "description": "Run for this many seconds then stop.",
                    },
                },
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "copytalker_tts_synthesize",
            "description": (
                "Synthesize text to speech audio using any supported TTS engine. "
                "Supports voice cloning (IndexTTS/Fish-Speech), emotion control, "
                "duration control, and emotion text tags."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "Text to synthesize. For Fish-Speech, you can embed "
                            "emotion tags like '(happy) Hello!'."
                        ),
                    },
                    "language": {
                        "type": "string",
                        "description": "Target language code (e.g. 'en', 'zh', 'ja').",
                        "default": "en",
                    },
                    "engine": {
                        "type": "string",
                        "enum": [
                            "kokoro", "edge-tts", "pyttsx3",
                            "indextts", "fish-speech", "auto",
                        ],
                        "description": "TTS engine to use.",
                        "default": "auto",
                    },
                    "voice": {
                        "type": "string",
                        "description": (
                            "Voice name or path to reference audio for voice cloning."
                        ),
                    },
                    "speed": {
                        "type": "number",
                        "description": "Speech speed multiplier (0.5-2.0).",
                        "default": 1.0,
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output WAV file path.",
                    },
                    "emotion": {
                        "type": "string",
                        "description": (
                            "Emotion name for IndexTTS (happy, sad, angry, etc.) "
                            "or emotion tag for Fish-Speech (happy, whisper, etc.)."
                        ),
                    },
                    "emotion_audio": {
                        "type": "string",
                        "description": (
                            "Path to emotion reference audio (IndexTTS v2 only)."
                        ),
                    },
                    "target_duration": {
                        "type": "number",
                        "description": (
                            "Target audio duration in seconds (IndexTTS v2 only)."
                        ),
                    },
                    "reference_audio": {
                        "type": "string",
                        "description": (
                            "Path to speaker reference audio for voice cloning "
                            "(IndexTTS/Fish-Speech)."
                        ),
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "copytalker_list_voices",
            "description": (
                "List available TTS voices, optionally filtered by language."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "Filter by language code.",
                    },
                    "engine": {
                        "type": "string",
                        "enum": [
                            "kokoro", "edge-tts", "indextts", "fish-speech",
                        ],
                        "description": "TTS engine.",
                        "default": "kokoro",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "copytalker_list_languages",
            "description": "List all supported languages for translation.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "copytalker_list_emotions",
            "description": (
                "List available emotion tags for TTS engines that support "
                "emotion control (IndexTTS and Fish-Speech)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "engine": {
                        "type": "string",
                        "enum": ["indextts", "fish-speech"],
                        "description": "TTS engine to query emotions for.",
                        "default": "fish-speech",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "copytalker_clone_voice",
            "description": (
                "Clone a voice from a reference audio file using IndexTTS or "
                "Fish-Speech. Returns synthesized audio of the given text in "
                "the cloned voice."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to speak in the cloned voice.",
                    },
                    "reference_audio": {
                        "type": "string",
                        "description": "Path to reference audio file (5-30 seconds).",
                    },
                    "engine": {
                        "type": "string",
                        "enum": ["indextts", "fish-speech"],
                        "description": "Voice cloning engine to use.",
                        "default": "indextts",
                    },
                    "language": {
                        "type": "string",
                        "description": "Target language code.",
                        "default": "en",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output WAV file path.",
                    },
                    "emotion": {
                        "type": "string",
                        "description": "Optional emotion to apply.",
                    },
                },
                "required": ["text", "reference_audio"],
            },
        },
    },
]


def dispatch(name: str, arguments: dict[str, Any] | str) -> dict:
    """Dispatch a tool call to the appropriate API function."""
    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    if name == "copytalker_translate":
        from .api import translate

        result = translate(**arguments)
        return result.to_dict()

    if name == "copytalker_tts_synthesize":
        from .api import tts_synthesize

        result = tts_synthesize(**arguments)
        return result.to_dict()

    if name == "copytalker_list_voices":
        from .api import list_voices

        result = list_voices(**arguments)
        return result.to_dict()

    if name == "copytalker_list_languages":
        from .api import list_languages

        result = list_languages()
        return result.to_dict()

    if name == "copytalker_list_emotions":
        from .api import list_emotions

        result = list_emotions(**arguments)
        return result.to_dict()

    if name == "copytalker_clone_voice":
        from .api import clone_voice

        result = clone_voice(**arguments)
        return result.to_dict()

    raise ValueError(f"Unknown tool: {name}")
