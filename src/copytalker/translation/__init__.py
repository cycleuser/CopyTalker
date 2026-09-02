"""
Translation module supporting multiple backends.
"""

from copytalker.translation.translator import UnifiedTranslator
from copytalker.translation.helsinki import HelsinkiTranslator
from copytalker.translation.nllb import NLLBTranslator
from copytalker.translation.ollama import OllamaTranslator

__all__ = [
    "UnifiedTranslator",
    "HelsinkiTranslator",
    "NLLBTranslator",
    "OllamaTranslator",
]
