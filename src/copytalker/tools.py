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
                        "description": "TTS voice name.",
                    },
                    "tts_engine": {
                        "type": "string",
                        "enum": ["kokoro", "edge-tts", "pyttsx3", "auto"],
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
                        "enum": ["kokoro", "edge-tts"],
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
]


def dispatch(name: str, arguments: dict[str, Any] | str) -> dict:
    """Dispatch a tool call to the appropriate API function."""
    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    if name == "copytalker_translate":
        from .api import translate

        result = translate(**arguments)
        return result.to_dict()

    if name == "copytalker_list_voices":
        from .api import list_voices

        result = list_voices(**arguments)
        return result.to_dict()

    if name == "copytalker_list_languages":
        from .api import list_languages

        result = list_languages()
        return result.to_dict()

    raise ValueError(f"Unknown tool: {name}")
