"""
Unit tests for constants module.
"""

import pytest

from copytalker.core.constants import (
    SUPPORTED_LANGUAGES,
    LANGUAGE_NAMES,
    WHISPER_LANG_MAP,
    NLLB_LANG_CODE_MAP,
    KOKORO_LANG_MAP,
    KOKORO_VOICE_MAP,
    EDGE_TTS_VOICE_MAP,
    DEFAULT_TRANSLATION_MODELS,
    get_language_name,
    normalize_language_code,
    get_nllb_code,
    get_kokoro_lang_code,
    get_available_voices,
    get_default_voice,
    get_translation_models,
    is_language_supported,
    AUTO_DETECT_CODE,
)


class TestLanguageMappings:
    """Tests for language mappings."""
    
    def test_supported_languages_structure(self):
        """Test SUPPORTED_LANGUAGES structure."""
        assert isinstance(SUPPORTED_LANGUAGES, list)
        assert len(SUPPORTED_LANGUAGES) > 0
        
        for item in SUPPORTED_LANGUAGES:
            assert isinstance(item, tuple)
            assert len(item) == 2
            code, name = item
            assert isinstance(code, str)
            assert isinstance(name, str)
    
    def test_language_names_match(self):
        """Test LANGUAGE_NAMES matches SUPPORTED_LANGUAGES."""
        for code, name in SUPPORTED_LANGUAGES:
            assert code in LANGUAGE_NAMES
            assert LANGUAGE_NAMES[code] == name
    
    def test_whisper_lang_map_coverage(self):
        """Test Whisper language map covers supported languages."""
        for code, _ in SUPPORTED_LANGUAGES:
            assert code in WHISPER_LANG_MAP
    
    def test_nllb_lang_code_coverage(self):
        """Test NLLB code map covers supported languages."""
        for code, _ in SUPPORTED_LANGUAGES:
            assert code in NLLB_LANG_CODE_MAP
    
    def test_kokoro_lang_map_coverage(self):
        """Test Kokoro language map covers supported languages."""
        for code, _ in SUPPORTED_LANGUAGES:
            assert code in KOKORO_LANG_MAP


class TestNormalization:
    """Tests for language code normalization."""
    
    def test_normalize_standard_codes(self):
        """Test normalizing standard language codes."""
        assert normalize_language_code("en") == "en"
        assert normalize_language_code("zh") == "zh"
        assert normalize_language_code("ja") == "ja"
    
    def test_normalize_alternative_codes(self):
        """Test normalizing alternative language codes."""
        assert normalize_language_code("zh-cn") == "zh"
        assert normalize_language_code("zh-tw") == "zh"
        assert normalize_language_code("cmn") == "zh"
        assert normalize_language_code("jpn") == "ja"
        assert normalize_language_code("spa") == "es"
    
    def test_normalize_case_insensitive(self):
        """Test case-insensitive normalization."""
        assert normalize_language_code("EN") == "en"
        assert normalize_language_code("ZH-CN") == "zh"
    
    def test_normalize_unknown_code(self):
        """Test normalizing unknown codes returns as-is."""
        assert normalize_language_code("xyz") == "xyz"
    
    def test_normalize_none(self):
        """Test normalizing None returns default."""
        assert normalize_language_code(None) == "en"


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_get_language_name(self):
        """Test getting language display name."""
        assert get_language_name("en") == "English"
        assert get_language_name("zh") == "Chinese (Simplified)"
        assert get_language_name("ja") == "Japanese"
    
    def test_get_language_name_unknown(self):
        """Test getting name for unknown language returns code."""
        assert get_language_name("xyz") == "xyz"
    
    def test_get_nllb_code(self):
        """Test getting NLLB language codes."""
        assert get_nllb_code("en") == "eng_Latn"
        assert get_nllb_code("zh") == "zho_Hans"
        assert get_nllb_code("ja") == "jpn_Jpan"
    
    def test_get_kokoro_lang_code(self):
        """Test getting Kokoro language codes."""
        assert get_kokoro_lang_code("en") == "a"
        assert get_kokoro_lang_code("zh") == "z"
        assert get_kokoro_lang_code("ja") == "j"
    
    def test_is_language_supported(self):
        """Test language support check."""
        assert is_language_supported("en") is True
        assert is_language_supported("zh") is True
        assert is_language_supported("xyz") is False


class TestVoiceFunctions:
    """Tests for voice-related functions."""
    
    def test_get_available_voices_kokoro(self):
        """Test getting Kokoro voices."""
        voices = get_available_voices("en", "kokoro")
        assert isinstance(voices, list)
        assert len(voices) > 0
        assert "af_heart" in voices
    
    def test_get_available_voices_edge(self):
        """Test getting Edge TTS voices."""
        voices = get_available_voices("en", "edge-tts")
        assert isinstance(voices, list)
        assert len(voices) > 0
    
    def test_get_available_voices_unknown_lang(self):
        """Test getting voices for unknown language returns English."""
        voices = get_available_voices("xyz", "kokoro")
        assert voices == KOKORO_VOICE_MAP["en"]
    
    def test_get_default_voice(self):
        """Test getting default voice."""
        voice = get_default_voice("en", "kokoro")
        assert voice == KOKORO_VOICE_MAP["en"][0]
    
    def test_get_default_voice_chinese(self):
        """Test getting default Chinese voice."""
        voice = get_default_voice("zh", "kokoro")
        assert voice in KOKORO_VOICE_MAP["zh"]


class TestTranslationModels:
    """Tests for translation model functions."""
    
    def test_get_translation_models_specific(self):
        """Test getting models for specific language pair."""
        models = get_translation_models("en", "zh")
        assert isinstance(models, list)
        assert len(models) > 0
        assert any("Helsinki-NLP" in m for m in models)
    
    def test_get_translation_models_fallback(self):
        """Test getting models falls back to multilingual."""
        models = get_translation_models("ko", "ar")  # Unlikely to have specific model
        assert isinstance(models, list)
        assert len(models) > 0
        # Should fall back to NLLB
        assert any("nllb" in m.lower() for m in models)
    
    def test_default_translation_models_structure(self):
        """Test DEFAULT_TRANSLATION_MODELS structure."""
        assert "multilingual" in DEFAULT_TRANSLATION_MODELS
        assert isinstance(DEFAULT_TRANSLATION_MODELS["multilingual"], list)


class TestAutoDetect:
    """Tests for auto-detect constant."""
    
    def test_auto_detect_code_value(self):
        """Test AUTO_DETECT_CODE value."""
        assert AUTO_DETECT_CODE == "auto"
