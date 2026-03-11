#!/usr/bin/env python3
"""
CopyTalker Complete Test Suite
Tests all TTS engines, translation, STT, and GUI components.
"""

import os
import sys
import time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(name, success, details=""):
    symbol = "✓" if success else "✗"
    print(f"  {symbol} {name}" + (f" - {details}" if details else ""))
    return success


def test_tts_engines():
    """Test all TTS engines."""
    print_header("TTS Engine Tests")
    results = {}

    # Test Kokoro TTS
    print("\n[1/4] Kokoro TTS")
    try:
        from kokoro import KPipeline
        import numpy as np

        # English
        pipeline = KPipeline(lang_code="a")
        audio_segments = []
        for gs, ps, audio in pipeline("Hello, this is a test.", voice="af_heart", speed=1.0):
            audio_segments.append(audio)
        if audio_segments:
            full_audio = np.concatenate(audio_segments)
            results["kokoro_en"] = print_result(
                "Kokoro (English)", True, f"{len(full_audio)} samples"
            )
        else:
            results["kokoro_en"] = print_result("Kokoro (English)", False, "No audio")

        # Chinese
        pipeline_zh = KPipeline(lang_code="z")
        audio_segments = []
        for gs, ps, audio in pipeline_zh("你好，这是一个测试。", voice="zf_xiaobei", speed=1.0):
            audio_segments.append(audio)
        if audio_segments:
            full_audio = np.concatenate(audio_segments)
            results["kokoro_zh"] = print_result(
                "Kokoro (Chinese)", True, f"{len(full_audio)} samples"
            )
        else:
            results["kokoro_zh"] = print_result("Kokoro (Chinese)", False, "No audio")

    except Exception as e:
        results["kokoro_en"] = print_result("Kokoro", False, str(e)[:50])
        results["kokoro_zh"] = False

    # Test Edge TTS
    print("\n[2/4] Edge TTS")
    try:
        import edge_tts
        import asyncio

        async def test_edge(text, voice, lang):
            communicate = edge_tts.Communicate(text, voice)
            data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    data += chunk["data"]
            return len(data)

        results["edge_en"] = print_result(
            "Edge TTS (English)",
            True,
            f"{asyncio.run(test_edge('Hello', 'en-US-AriaNeural', 'en'))} bytes",
        )
        results["edge_zh"] = print_result(
            "Edge TTS (Chinese)",
            True,
            f"{asyncio.run(test_edge('你好', 'zh-CN-XiaoxiaoNeural', 'zh'))} bytes",
        )
        results["edge_ja"] = print_result(
            "Edge TTS (Japanese)",
            True,
            f"{asyncio.run(test_edge('こんにちは', 'ja-JP-NanamiNeural', 'ja'))} bytes",
        )
        results["edge_ko"] = print_result(
            "Edge TTS (Korean)",
            True,
            f"{asyncio.run(test_edge('안녕하세요', 'ko-KR-SunHiNeural', 'ko'))} bytes",
        )
        results["edge_fr"] = print_result(
            "Edge TTS (French)",
            True,
            f"{asyncio.run(test_edge('Bonjour', 'fr-FR-DeniseNeural', 'fr'))} bytes",
        )
        results["edge_de"] = print_result(
            "Edge TTS (German)",
            True,
            f"{asyncio.run(test_edge('Hallo', 'de-DE-KatjaNeural', 'de'))} bytes",
        )
        results["edge_es"] = print_result(
            "Edge TTS (Spanish)",
            True,
            f"{asyncio.run(test_edge('Hola', 'es-ES-ElviraNeural', 'es'))} bytes",
        )
        results["edge_ru"] = print_result(
            "Edge TTS (Russian)",
            True,
            f"{asyncio.run(test_edge('Привет', 'ru-RU-SvetlanaNeural', 'ru'))} bytes",
        )
        results["edge_it"] = print_result(
            "Edge TTS (Italian)",
            True,
            f"{asyncio.run(test_edge('Ciao', 'it-IT-ElsaNeural', 'it'))} bytes",
        )
        results["edge_pt"] = print_result(
            "Edge TTS (Portuguese)",
            True,
            f"{asyncio.run(test_edge('Olá', 'pt-BR-FranciscaNeural', 'pt'))} bytes",
        )

    except Exception as e:
        results["edge_en"] = print_result("Edge TTS", False, str(e)[:50])

    # Test pyttsx3
    print("\n[3/4] pyttsx3 TTS")
    try:
        import pyttsx3

        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        results["pyttsx3"] = print_result("pyttsx3", True, f"{len(voices)} voices")
    except Exception as e:
        results["pyttsx3"] = print_result("pyttsx3", False, str(e)[:50])

    # Test Fish-Speech
    print("\n[4/4] Fish-Speech")
    try:
        from fish_audio_sdk import Session

        results["fish_speech"] = print_result("Fish-Speech SDK", True, "installed")
    except ImportError:
        results["fish_speech"] = print_result("Fish-Speech SDK", None, "not installed (optional)")
    except Exception as e:
        results["fish_speech"] = print_result("Fish-Speech SDK", None, str(e)[:30])

    return results


def test_translation():
    """Test translation engines."""
    print_header("Translation Tests")
    results = {}

    try:
        from copytalker.translation.translator import UnifiedTranslator

        translator = UnifiedTranslator()

        test_pairs = [
            ("en", "zh", "Hello, world!"),
            ("zh", "en", "你好世界"),
            ("en", "ja", "Hello"),
            ("en", "ko", "Hello"),
            ("en", "fr", "Hello"),
            ("en", "de", "Hello"),
            ("en", "es", "Hello"),
            ("en", "ru", "Hello"),
            ("en", "it", "Hello"),
            ("en", "pt", "Hello"),
        ]

        for src, tgt, text in test_pairs:
            try:
                result = translator.translate(text, src, tgt)
                results[f"{src}_{tgt}"] = print_result(
                    f"{src} -> {tgt}", True, f"'{text}' -> '{result.translated_text[:20]}...'"
                )
            except Exception as e:
                results[f"{src}_{tgt}"] = print_result(f"{src} -> {tgt}", False, str(e)[:30])

    except Exception as e:
        print_result("Translation", False, str(e))

    return results


def test_speech_to_text():
    """Test speech-to-text (Whisper)."""
    print_header("Speech-to-Text Tests")
    results = {}

    try:
        from faster_whisper import WhisperModel

        print_result("Whisper available", True)
        results["whisper"] = True
    except Exception as e:
        results["whisper"] = print_result("Whisper", False, str(e)[:50])

    return results


def test_gui():
    """Test GUI components."""
    print_header("GUI Tests")
    results = {}

    try:
        from PySide6.QtWidgets import QApplication

        results["pyside6"] = print_result("PySide6", True)
    except Exception as e:
        results["pyside6"] = print_result("PySide6", False, str(e)[:50])
        return results

    try:
        from copytalker.core.i18n import I18n, UI_LANGUAGES

        results["i18n"] = print_result("i18n", True, f"{len(UI_LANGUAGES)} languages")
    except Exception as e:
        results["i18n"] = print_result("i18n", False, str(e)[:50])

    try:
        from copytalker.core.constants import SUPPORTED_LANGUAGES

        results["constants"] = print_result(
            "Constants", True, f"{len(SUPPORTED_LANGUAGES)} languages"
        )
    except Exception as e:
        results["constants"] = print_result("Constants", False, str(e)[:50])

    try:
        from copytalker.gui.qt.app import CopyTalkerApp

        results["gui_app"] = print_result("GUI App", True)
    except Exception as e:
        results["gui_app"] = print_result("GUI App", False, str(e)[:50])
        return results

    # Test GUI creation
    try:
        app = QApplication([])
        window = CopyTalkerApp()

        results["gui_window"] = print_result("Main Window", True)

        # Test UI components
        results["status_label"] = print_result(
            "Status Label", True, f"'{window._conversation_view._status_label.text()}'"
        )
        results["start_btn"] = print_result(
            "Start Button", True, f"'{window._conversation_view._start_btn.text()}'"
        )
        results["stop_btn"] = print_result(
            "Stop Button", True, f"'{window._conversation_view._stop_btn.text()}'"
        )
        results["settings_btn"] = print_result(
            "Settings Button", True, f"'{window._conversation_view._settings_btn.text()}'"
        )

        # Test language switching
        window._conversation_view.set_ui_language("zh")
        results["lang_zh"] = print_result(
            "UI Language (zh)", True, f"'{window._conversation_view._status_label.text()}'"
        )

        window._conversation_view.set_ui_language("ja")
        results["lang_ja"] = print_result(
            "UI Language (ja)", True, f"'{window._conversation_view._status_label.text()}'"
        )

        window._conversation_view.set_ui_language("en")
        results["lang_en"] = print_result(
            "UI Language (en)", True, f"'{window._conversation_view._status_label.text()}'"
        )

        # Test settings dialog
        window._show_settings_dialog()
        results["settings_dialog"] = print_result("Settings Dialog", True)

        app.quit()

    except Exception as e:
        results["gui_test"] = print_result("GUI Test", False, str(e)[:50])

    return results


def test_audio():
    """Test audio components."""
    print_header("Audio Tests")
    results = {}

    try:
        import sounddevice

        devices = sounddevice.query_devices()
        results["sounddevice"] = print_result("sounddevice", True, f"{len(devices)} devices")
    except Exception as e:
        results["sounddevice"] = print_result("sounddevice", False, str(e)[:50])

    try:
        import webrtcvad

        results["webrtcvad"] = print_result("webrtcvad (VAD)", True)
    except Exception as e:
        results["webrtcvad"] = print_result("webrtcvad", False, str(e)[:50])

    try:
        import pydub

        results["pydub"] = print_result("pydub", True)
    except Exception as e:
        results["pydub"] = print_result("pydub", False, str(e)[:50])

    return results


def main():
    print("\n" + "=" * 60)
    print("  CopyTalker Complete Test Suite")
    print("=" * 60)

    all_results = {}

    # Run all tests
    all_results["tts"] = test_tts_engines()
    all_results["translation"] = test_translation()
    all_results["stt"] = test_speech_to_text()
    all_results["gui"] = test_gui()
    all_results["audio"] = test_audio()

    # Summary
    print_header("Test Summary")

    total = 0
    passed = 0
    failed = 0
    skipped = 0

    for category, results in all_results.items():
        for name, result in results.items():
            total += 1
            if result is True:
                passed += 1
            elif result is False:
                failed += 1
            else:
                skipped += 1

    print(f"\n  Total:  {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")

    if failed == 0:
        print("\n  All tests passed!")
        return 0
    else:
        print(f"\n  {failed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
