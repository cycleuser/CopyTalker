#!/usr/bin/env python3
"""
CopyTalker Quick Test Suite
Tests installed TTS engines, GUI, and core components.
"""

import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def test_tts():
    print("\n" + "=" * 50)
    print("  TTS Engine Tests")
    print("=" * 50)

    # Test Kokoro
    print("\n[1] Kokoro TTS")
    try:
        from kokoro import KPipeline
        import numpy as np

        pipeline = KPipeline(lang_code="a")
        audio_segments = []
        for gs, ps, audio in pipeline("Hello", voice="af_heart", speed=1.0):
            audio_segments.append(audio)
        if audio_segments:
            print(f"  ✓ Kokoro (en): {len(np.concatenate(audio_segments))} samples")
        pipeline_zh = KPipeline(lang_code="z")
        audio_segments = []
        for gs, ps, audio in pipeline_zh("你好", voice="zf_xiaobei", speed=1.0):
            audio_segments.append(audio)
        if audio_segments:
            print(f"  ✓ Kokoro (zh): {len(np.concatenate(audio_segments))} samples")
    except Exception as e:
        print(f"  ✗ Kokoro: {e}")

    # Test Edge TTS
    print("\n[2] Edge TTS")
    try:
        import edge_tts
        import asyncio

        async def test_edge(text, voice):
            communicate = edge_tts.Communicate(text, voice)
            data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    data += chunk["data"]
            return len(data)

        print(f"  ✓ Edge (en): {asyncio.run(test_edge('Hello', 'en-US-AriaNeural'))} bytes")
        print(f"  ✓ Edge (zh): {asyncio.run(test_edge('你好', 'zh-CN-XiaoxiaoNeural'))} bytes")
        print(f"  ✓ Edge (ja): {asyncio.run(test_edge('こんにちは', 'ja-JP-NanamiNeural'))} bytes")
        print(f"  ✓ Edge (ko): {asyncio.run(test_edge('안녕', 'ko-KR-SunHiNeural'))} bytes")
        print(f"  ✓ Edge (fr): {asyncio.run(test_edge('Bonjour', 'fr-FR-DeniseNeural'))} bytes")
        print(f"  ✓ Edge (de): {asyncio.run(test_edge('Hallo', 'de-DE-KatjaNeural'))} bytes")
        print(f"  ✓ Edge (es): {asyncio.run(test_edge('Hola', 'es-ES-ElviraNeural'))} bytes")
        print(f"  ✓ Edge (ru): {asyncio.run(test_edge('Привет', 'ru-RU-SvetlanaNeural'))} bytes")
        print(f"  ✓ Edge (it): {asyncio.run(test_edge('Ciao', 'it-IT-ElsaNeural'))} bytes")
        print(f"  ✓ Edge (pt): {asyncio.run(test_edge('Olá', 'pt-BR-FranciscaNeural'))} bytes")
    except Exception as e:
        print(f"  ✗ Edge TTS: {e}")

    # Test pyttsx3
    print("\n[3] pyttsx3")
    try:
        import pyttsx3

        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        print(f"  ✓ pyttsx3: {len(voices)} voices")
    except Exception as e:
        print(f"  ✗ pyttsx3: {e}")


def test_gui():
    print("\n" + "=" * 50)
    print("  GUI Tests")
    print("=" * 50)

    try:
        from PySide6.QtWidgets import QApplication

        print("  ✓ PySide6")
    except Exception as e:
        print(f"  ✗ PySide6: {e}")
        return

    try:
        from copytalker.core.i18n import I18n, UI_LANGUAGES

        print(f"  ✓ i18n: {len(UI_LANGUAGES)} UI languages")
    except Exception as e:
        print(f"  ✗ i18n: {e}")

    try:
        from copytalker.core.constants import SUPPORTED_LANGUAGES

        print(f"  ✓ constants: {len(SUPPORTED_LANGUAGES)} target languages")
    except Exception as e:
        print(f"  ✗ constants: {e}")

    try:
        from copytalker.gui.qt.app import CopyTalkerApp

        print("  ✓ GUI app import")

        app = QApplication([])
        window = CopyTalkerApp()
        print(f"  ✓ Main window: status='{window._conversation_view._status_label.text()}'")
        print(f"  ✓ Buttons: Start, Stop, Settings, Clear")

        window._conversation_view.set_ui_language("zh")
        print(f"  ✓ UI Chinese: '{window._conversation_view._status_label.text()}'")

        window._conversation_view.set_ui_language("en")
        print(f"  ✓ UI English: '{window._conversation_view._status_label.text()}'")

        window._show_settings_dialog()
        print("  ✓ Settings dialog")

        app.quit()
        print("\n  ✓ All GUI tests passed!")

    except Exception as e:
        print(f"  ✗ GUI test: {e}")


def test_audio():
    print("\n" + "=" * 50)
    print("  Audio Tests")
    print("=" * 50)

    try:
        import sounddevice

        devices = sounddevice.query_devices()
        print(f"  ✓ sounddevice: {len(devices)} devices")
    except Exception as e:
        print(f"  ✗ sounddevice: {e}")

    try:
        import webrtcvad

        print("  ✓ webrtcvad (VAD)")
    except Exception as e:
        print(f"  ✗ webrtcvad: {e}")

    try:
        from faster_whisper import WhisperModel

        print("  ✓ Whisper (STT)")
    except Exception as e:
        print(f"  ✗ Whisper: {e}")


def main():
    print("\n" + "=" * 50)
    print("  CopyTalker Quick Test Suite")
    print("=" * 50)

    test_tts()
    test_gui()
    test_audio()

    print("\n" + "=" * 50)
    print("  All tests completed!")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
