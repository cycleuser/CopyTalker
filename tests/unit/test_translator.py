"""
Unit tests for translation module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from copytalker.core.config import TranslationConfig
from copytalker.core.types import TranslationResult
from copytalker.translation.translator import UnifiedTranslator
from copytalker.translation.helsinki import HelsinkiTranslator
from copytalker.translation.nllb import NLLBTranslator


class TestHelsinkiTranslator:
    """Tests for HelsinkiTranslator."""
    
    def test_initialization(self):
        """Test HelsinkiTranslator initialization."""
        config = TranslationConfig()
        translator = HelsinkiTranslator(config)
        
        assert translator.config == config
    
    def test_supports_pair_valid(self):
        """Test supports_pair for valid language pair."""
        translator = HelsinkiTranslator()
        
        # en->zh should be supported
        assert translator.supports_pair("en", "zh") is True
    
    def test_supports_pair_invalid(self):
        """Test supports_pair for invalid language pair."""
        translator = HelsinkiTranslator()
        
        # ko->ar is unlikely to have Helsinki-NLP model
        assert translator.supports_pair("ko", "ar") is False
    
    def test_translate_same_language(self):
        """Test translation when source equals target."""
        translator = HelsinkiTranslator()
        result = translator.translate("Hello", "en", "en")
        
        assert result.original_text == "Hello"
        assert result.translated_text == "Hello"
        assert result.model_used == "none"
    
    @patch('copytalker.translation.helsinki.MarianMTModel')
    @patch('copytalker.translation.helsinki.MarianTokenizer')
    def test_translate(self, mock_tokenizer_class, mock_model_class):
        """Test translation."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": Mock(to=Mock(return_value=Mock())),
            "attention_mask": Mock(to=Mock(return_value=Mock())),
        }
        mock_tokenizer.decode.return_value = "你好"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = [Mock()]
        mock_model.to.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        translator = HelsinkiTranslator(TranslationConfig(device="cpu"))
        result = translator.translate("Hello", "en", "zh")
        
        assert isinstance(result, TranslationResult)
        assert result.original_text == "Hello"
        assert result.translated_text == "你好"


class TestNLLBTranslator:
    """Tests for NLLBTranslator."""
    
    def test_initialization(self):
        """Test NLLBTranslator initialization."""
        translator = NLLBTranslator()
        
        assert translator._model_name == NLLBTranslator.DEFAULT_MODEL
        assert translator.is_loaded is False
    
    def test_supports_pair(self):
        """Test NLLB supports most language pairs."""
        translator = NLLBTranslator()
        
        # NLLB should support most pairs
        assert translator.supports_pair("en", "zh") is True
        assert translator.supports_pair("ko", "ar") is True
    
    def test_translate_same_language(self):
        """Test translation when source equals target."""
        translator = NLLBTranslator()
        result = translator.translate("Hello", "en", "en")
        
        assert result.original_text == "Hello"
        assert result.translated_text == "Hello"
        assert result.model_used == "none"
    
    @patch('copytalker.translation.nllb.AutoModelForSeq2SeqLM')
    @patch('copytalker.translation.nllb.AutoTokenizer')
    def test_translate(self, mock_tokenizer_class, mock_model_class):
        """Test NLLB translation."""
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": Mock(to=Mock(return_value=Mock())),
            "attention_mask": Mock(to=Mock(return_value=Mock())),
        }
        mock_tokenizer.decode.return_value = "翻译结果"
        mock_tokenizer.convert_tokens_to_ids.return_value = 123
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = [Mock()]
        mock_model.to.return_value = mock_model
        mock_model_class.from_pretrained.return_value = mock_model
        
        translator = NLLBTranslator(TranslationConfig(device="cpu"))
        result = translator.translate("Hello", "en", "zh")
        
        assert isinstance(result, TranslationResult)
        assert result.original_text == "Hello"


class TestUnifiedTranslator:
    """Tests for UnifiedTranslator."""
    
    def test_initialization(self):
        """Test UnifiedTranslator initialization."""
        translator = UnifiedTranslator()
        
        assert translator._helsinki is not None
        assert translator._nllb is not None
    
    def test_supports_pair(self):
        """Test supports_pair delegates to backends."""
        translator = UnifiedTranslator()
        
        # Should support common pairs
        assert translator.supports_pair("en", "zh") is True
    
    def test_translate_empty_text(self):
        """Test translation with empty text."""
        translator = UnifiedTranslator()
        result = translator.translate("", "en", "zh")
        
        assert result.translated_text == ""
    
    def test_translate_same_language(self):
        """Test translation when source equals target."""
        translator = UnifiedTranslator()
        result = translator.translate("Hello", "en", "en")
        
        assert result.translated_text == "Hello"
    
    def test_get_available_models(self):
        """Test getting available models for language pair."""
        translator = UnifiedTranslator()
        models = translator.get_available_models("en", "zh")
        
        assert isinstance(models, list)
        assert len(models) > 0


class TestTranslationResult:
    """Tests for TranslationResult."""
    
    def test_is_same_language_true(self):
        """Test is_same_language when equal."""
        result = TranslationResult(
            original_text="Hello",
            translated_text="Hello",
            source_lang="en",
            target_lang="en",
        )
        assert result.is_same_language() is True
    
    def test_is_same_language_false(self):
        """Test is_same_language when different."""
        result = TranslationResult(
            original_text="Hello",
            translated_text="你好",
            source_lang="en",
            target_lang="zh",
        )
        assert result.is_same_language() is False
