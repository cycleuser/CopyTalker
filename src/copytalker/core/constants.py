"""
Language mappings and constants for CopyTalker.

This module centralizes all language codes, voice mappings, and model configurations.
"""

from typing import Dict, List, Tuple

# Supported languages: (code, display_name)
SUPPORTED_LANGUAGES: List[Tuple[str, str]] = [
    ("en", "English"),
    ("zh", "Chinese (Simplified)"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("de", "German"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("ru", "Russian"),
    ("ar", "Arabic"),
]

# Language code to display name mapping
LANGUAGE_NAMES: Dict[str, str] = {code: name for code, name in SUPPORTED_LANGUAGES}

# Whisper language code normalization
# Maps various language codes to standard codes
WHISPER_LANG_MAP: Dict[str, str] = {
    "zh-cn": "zh",
    "zh-tw": "zh",
    "cmn": "zh",
    "spa": "es",
    "fra": "fr",
    "deu": "de",
    "jpn": "ja",
    "kor": "ko",
    "rus": "ru",
    "ara": "ar",
    "eng": "en",
    # Identity mappings for standard codes
    "en": "en",
    "zh": "zh",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "ja": "ja",
    "ko": "ko",
    "ru": "ru",
    "ar": "ar",
}

# NLLB-200 language codes
NLLB_LANG_CODE_MAP: Dict[str, str] = {
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab",
}

# Kokoro TTS language mapping
KOKORO_LANG_MAP: Dict[str, str] = {
    "en": "a",  # American English
    "zh": "z",  # Chinese
    "es": "a",  # Use English for unsupported
    "fr": "a",
    "de": "a",
    "ja": "j",  # Japanese
    "ko": "a",
    "ru": "a",
    "ar": "a",
}

# Kokoro TTS voice mapping per language
KOKORO_VOICE_MAP: Dict[str, List[str]] = {
    "en": [
        "af_heart", "af_sky", "af_alloy", "af_aoede", "af_bella",
        "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river",
        "af_sarah", "am_adam", "am_echo", "am_eric", "am_fenrir",
        "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
    ],
    "zh": [
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
        "zm_yunjian", "zm_yunxia", "zm_yunxi", "zm_yunyang",
    ],
    "ja": [
        "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
    ],
    "es": ["af_heart"],
    "fr": ["af_heart"],
    "de": ["af_heart"],
    "ko": ["af_heart"],
    "ru": ["af_heart"],
    "ar": ["af_heart"],
}

# Edge TTS voice mapping per language
EDGE_TTS_VOICE_MAP: Dict[str, List[str]] = {
    "en": [
        "en-US-AriaNeural",
        "en-US-GuyNeural",
        "en-US-JennyNeural",
        "en-GB-SoniaNeural",
        "en-GB-RyanNeural",
    ],
    "zh": [
        "zh-CN-XiaoxiaoNeural",
        "zh-CN-YunxiNeural",
        "zh-CN-YunjianNeural",
        "zh-TW-HsiaoChenNeural",
    ],
    "ja": [
        "ja-JP-NanamiNeural",
        "ja-JP-KeitaNeural",
    ],
    "ko": [
        "ko-KR-SunHiNeural",
        "ko-KR-InJoonNeural",
    ],
    "es": [
        "es-ES-ElviraNeural",
        "es-MX-DaliaNeural",
    ],
    "fr": [
        "fr-FR-DeniseNeural",
        "fr-FR-HenriNeural",
    ],
    "de": [
        "de-DE-KatjaNeural",
        "de-DE-ConradNeural",
    ],
    "ru": [
        "ru-RU-SvetlanaNeural",
        "ru-RU-DmitryNeural",
    ],
    "ar": [
        "ar-SA-ZariyahNeural",
        "ar-SA-HamedNeural",
    ],
}

# Default translation models for language pairs
DEFAULT_TRANSLATION_MODELS: Dict[str, List[str]] = {
    "en->zh": ["Helsinki-NLP/opus-mt-en-zh"],
    "zh->en": ["Helsinki-NLP/opus-mt-zh-en"],
    "en->de": ["Helsinki-NLP/opus-mt-en-de"],
    "de->en": ["Helsinki-NLP/opus-mt-de-en"],
    "en->fr": ["Helsinki-NLP/opus-mt-en-fr"],
    "fr->en": ["Helsinki-NLP/opus-mt-fr-en"],
    "en->es": ["Helsinki-NLP/opus-mt-en-es"],
    "es->en": ["Helsinki-NLP/opus-mt-es-en"],
    "en->ru": ["Helsinki-NLP/opus-mt-en-ru"],
    "ru->en": ["Helsinki-NLP/opus-mt-ru-en"],
    "en->ar": ["Helsinki-NLP/opus-mt-en-ar"],
    "ar->en": ["Helsinki-NLP/opus-mt-ar-en"],
    "en->ja": ["Helsinki-NLP/opus-mt-en-jap"],
    "ja->en": ["Helsinki-NLP/opus-mt-ja-en"],
    "en->ko": ["Helsinki-NLP/opus-mt-en-ko" ],
    "ko->en": ["Helsinki-NLP/opus-mt-ko-en"],
    # Pairs without Helsinki models use NLLB automatically:
    # ja->zh, zh->ja, ko->zh, zh->ko, ja->ko, ko->ja,
    # fr->de, de->fr, es->fr, fr->es, etc.
    # Multilingual fallback
    "multilingual": [
        "facebook/nllb-200-distilled-600M",
        "facebook/nllb-200-distilled-1.3B",
        "facebook/nllb-200-1.3B",
        "facebook/nllb-200-3.3B",
    ],
}

# Audio configuration defaults
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FRAME_DURATION_MS = 30
DEFAULT_VAD_AGGRESSIVENESS = 3
DEFAULT_SILENCE_THRESHOLD_S = 0.8
DEFAULT_AUDIO_BUFFER_SIZE = 10

# Kokoro TTS sample rate
KOKORO_SAMPLE_RATE = 22050

# Model sizes (for user information)
MODEL_SIZES: Dict[str, str] = {
    "whisper-tiny": "~75 MB",
    "whisper-base": "~145 MB",
    "whisper-small": "~465 MB",
    "whisper-medium": "~1.5 GB",
    "whisper-large": "~3 GB",
    "nllb-200-distilled-600M": "~1.2 GB",
    "nllb-200-distilled-1.3B": "~2.6 GB",
    "kokoro-82M": "~330 MB",
}

# Auto-detect language code
AUTO_DETECT_CODE = "auto"


def get_language_name(code: str) -> str:
    """Get display name for a language code."""
    return LANGUAGE_NAMES.get(code, code)


def normalize_language_code(code: str) -> str:
    """Normalize a language code to standard format."""
    if code is None:
        return "en"
    code_lower = code.lower()
    return WHISPER_LANG_MAP.get(code_lower, code_lower)


def get_nllb_code(lang: str) -> str:
    """Get NLLB language code for a standard language code."""
    return NLLB_LANG_CODE_MAP.get(lang, "eng_Latn")


def get_kokoro_lang_code(lang: str) -> str:
    """Get Kokoro language code for a standard language code."""
    return KOKORO_LANG_MAP.get(lang, "a")


def get_available_voices(lang: str, engine: str = "kokoro") -> List[str]:
    """Get available voices for a language and TTS engine."""
    if engine == "kokoro":
        return KOKORO_VOICE_MAP.get(lang, KOKORO_VOICE_MAP["en"])
    elif engine == "edge-tts":
        return EDGE_TTS_VOICE_MAP.get(lang, EDGE_TTS_VOICE_MAP["en"])
    return []


def get_default_voice(lang: str, engine: str = "kokoro") -> str:
    """Get default voice for a language and TTS engine."""
    voices = get_available_voices(lang, engine)
    return voices[0] if voices else ""


def get_translation_models(source_lang: str, target_lang: str) -> List[str]:
    """Get available translation models for a language pair."""
    key = f"{source_lang}->{target_lang}"
    models = DEFAULT_TRANSLATION_MODELS.get(key, [])
    if not models:
        # Fall back to multilingual models
        models = DEFAULT_TRANSLATION_MODELS.get("multilingual", [])
    return models


def is_language_supported(code: str) -> bool:
    """Check if a language code is supported."""
    normalized = normalize_language_code(code)
    return normalized in LANGUAGE_NAMES
