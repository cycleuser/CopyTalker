"""
Audio module for capture, playback, and recording.

Uses lazy imports to avoid requiring pyaudio at import time.
"""


def __getattr__(name):
    if name == "AudioCapturer":
        from copytalker.audio.capture import AudioCapturer
        return AudioCapturer
    if name == "AudioPlayer":
        from copytalker.audio.playback import AudioPlayer
        return AudioPlayer
    if name == "VoiceRecorder":
        from copytalker.audio.recorder import VoiceRecorder
        return VoiceRecorder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AudioCapturer", "AudioPlayer", "VoiceRecorder"]
