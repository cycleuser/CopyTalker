"""
Conversation history management for CopyTalker.

Saves original audio, translated audio, original text, and translated text.
Text is saved in a Markdown file with links to audio files.
"""

from __future__ import annotations

import logging
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConversationEntry:
    """A single conversation entry with original and translated content."""
    timestamp: str
    original_text: str = ""
    translated_text: str = ""
    original_lang: str = ""
    target_lang: str = ""
    original_audio_path: Optional[str] = None
    translated_audio_path: Optional[str] = None
    index: int = 0


@dataclass
class ConversationHistory:
    """
    Manages conversation history storage.

    Directory structure:
        history_dir/
            YYYY-MM-DD_HH-MM-SS/
                conversation.md
                audio/
                    001_original.wav
                    001_translated.wav
                    002_original.wav
                    002_translated.wav
                    ...
    """

    history_dir: Path
    session_name: str = ""
    _session_dir: Optional[Path] = None
    _audio_dir: Optional[Path] = None
    _entries: list[ConversationEntry] = field(default_factory=list)
    _entry_index: int = 0
    _markdown_file: Optional[Path] = None

    def __post_init__(self) -> None:
        if isinstance(self.history_dir, str):
            self.history_dir = Path(self.history_dir)

    def start_session(self, session_name: Optional[str] = None) -> Path:
        """
        Start a new conversation session.

        Creates session directory and initializes the markdown file.

        Args:
            session_name: Optional session name. If None, uses timestamp.

        Returns:
            Path to the session directory.
        """
        if session_name is None:
            session_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.session_name = session_name
        self._session_dir = self.history_dir / session_name
        self._audio_dir = self._session_dir / "audio"
        self._markdown_file = self._session_dir / "conversation.md"

        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._audio_dir.mkdir(parents=True, exist_ok=True)

        self._entries = []
        self._entry_index = 0

        self._write_markdown_header()

        logger.info(f"Started conversation session: {self._session_dir}")
        return self._session_dir

    def _write_markdown_header(self) -> None:
        """Write the markdown file header."""
        if self._markdown_file is None:
            return

        header = f"""# CopyTalker Conversation

**Session**: {self.session_name}
**Started**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

"""
        with open(self._markdown_file, "w", encoding="utf-8") as f:
            f.write(header)

    def save_original_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        entry: Optional[ConversationEntry] = None,
    ) -> Optional[str]:
        """
        Save original audio to a WAV file.

        Args:
            audio: Audio data as float32 numpy array.
            sample_rate: Sample rate in Hz.
            entry: Optional entry to update. If None, creates a new entry.

        Returns:
            Path to the saved audio file, or None if failed.
        """
        if self._audio_dir is None:
            logger.warning("Session not started, cannot save audio")
            return None

        if entry is None:
            self._entry_index += 1
            entry = ConversationEntry(
                timestamp=datetime.now().strftime("%H:%M:%S"),
                index=self._entry_index,
            )
            self._entries.append(entry)

        filename = f"{entry.index:03d}_original.wav"
        filepath = self._audio_dir / filename

        try:
            self._save_wav(filepath, audio, sample_rate)
            entry.original_audio_path = f"audio/{filename}"
            logger.debug(f"Saved original audio: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save original audio: {e}")
            return None

    def save_translated_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        entry_index: Optional[int] = None,
    ) -> Optional[str]:
        """
        Save translated audio to a WAV file.

        Args:
            audio: Audio data as float32 numpy array.
            sample_rate: Sample rate in Hz.
            entry_index: Index of the entry to update. If None, uses the latest.

        Returns:
            Path to the saved audio file, or None if failed.
        """
        if self._audio_dir is None:
            logger.warning("Session not started, cannot save audio")
            return None

        if entry_index is None:
            entry_index = self._entry_index

        entry = self._get_entry_by_index(entry_index)
        if entry is None:
            logger.warning(f"No entry found with index {entry_index}")
            return None

        filename = f"{entry_index:03d}_translated.wav"
        filepath = self._audio_dir / filename

        try:
            self._save_wav(filepath, audio, sample_rate)
            entry.translated_audio_path = f"audio/{filename}"
            logger.debug(f"Saved translated audio: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save translated audio: {e}")
            return None

    def add_transcription(
        self,
        text: str,
        language: str = "",
        entry_index: Optional[int] = None,
    ) -> Optional[int]:
        """
        Add transcription text to an entry.

        Args:
            text: Transcribed text.
            language: Source language code.
            entry_index: Index of the entry. If None, uses the latest.

        Returns:
            Entry index, or None if failed.
        """
        if entry_index is None:
            entry_index = self._entry_index

        entry = self._get_entry_by_index(entry_index)
        if entry is None:
            logger.warning(f"No entry found with index {entry_index}")
            return None

        entry.original_text = text
        entry.original_lang = language
        return entry_index

    def add_translation(
        self,
        text: str,
        target_lang: str = "",
        entry_index: Optional[int] = None,
    ) -> Optional[int]:
        """
        Add translation text to an entry.

        Args:
            text: Translated text.
            target_lang: Target language code.
            entry_index: Index of the entry. If None, uses the latest.

        Returns:
            Entry index, or None if failed.
        """
        if entry_index is None:
            entry_index = self._entry_index

        entry = self._get_entry_by_index(entry_index)
        if entry is None:
            logger.warning(f"No entry found with index {entry_index}")
            return None

        entry.translated_text = text
        entry.target_lang = target_lang
        return entry_index

    def create_entry(self) -> int:
        """
        Create a new conversation entry.

        Returns:
            The new entry index.
        """
        self._entry_index += 1
        entry = ConversationEntry(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            index=self._entry_index,
        )
        self._entries.append(entry)
        return self._entry_index

    def get_latest_entry(self) -> Optional[ConversationEntry]:
        """Get the latest conversation entry."""
        if not self._entries:
            return None
        return self._entries[-1]

    def _get_entry_by_index(self, index: int) -> Optional[ConversationEntry]:
        """Get entry by index."""
        for entry in self._entries:
            if entry.index == index:
                return entry
        return None

    def _save_wav(
        self,
        filepath: Path,
        audio: np.ndarray,
        sample_rate: int,
    ) -> None:
        """Save audio data to a WAV file."""
        if audio.dtype != np.int16:
            audio = (audio * 32768).clip(-32768, 32767).astype(np.int16)

        with wave.open(str(filepath), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())

    def flush_to_markdown(self) -> None:
        """Write all entries to the markdown file."""
        if self._markdown_file is None:
            return

        with open(self._markdown_file, "a", encoding="utf-8") as f:
            for entry in self._entries:
                f.write(self._format_entry(entry))

        logger.info(f"Flushed {len(self._entries)} entries to markdown")

    def _format_entry(self, entry: ConversationEntry) -> str:
        """Format a single entry as markdown."""
        lines = [
            f"## Entry {entry.index}",
            f"**Time**: {entry.timestamp}",
            "",
        ]

        if entry.original_text:
            lang_info = f" ({entry.original_lang})" if entry.original_lang else ""
            lines.append(f"### Original{lang_info}")
            lines.append(f"> {entry.original_text}")
            if entry.original_audio_path:
                lines.append(f"")
                lines.append(f"[Original Audio]({entry.original_audio_path})")
            lines.append("")

        if entry.translated_text:
            lang_info = f" ({entry.target_lang})" if entry.target_lang else ""
            lines.append(f"### Translation{lang_info}")
            lines.append(f"{entry.translated_text}")
            if entry.translated_audio_path:
                lines.append(f"")
                lines.append(f"[Translated Audio]({entry.translated_audio_path})")
            lines.append("")

        lines.append("---")
        lines.append("")

        return "\n".join(lines)

    def end_session(self) -> Path:
        """
        End the current session and finalize the markdown file.

        Returns:
            Path to the session directory.
        """
        self.flush_to_markdown()

        if self._markdown_file:
            with open(self._markdown_file, "a", encoding="utf-8") as f:
                f.write(f"\n**Session ended**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        logger.info(f"Session ended: {self._session_dir}")
        return self._session_dir or Path(".")

    @property
    def session_dir(self) -> Optional[Path]:
        """Get the current session directory."""
        return self._session_dir

    @property
    def markdown_path(self) -> Optional[Path]:
        """Get the current markdown file path."""
        return self._markdown_file

    def get_entries(self) -> list[ConversationEntry]:
        """Get all conversation entries."""
        return self._entries.copy()


def get_default_history_dir() -> Path:
    """Get the default history directory."""
    from copytalker.core.config import get_default_cache_dir
    return get_default_cache_dir() / "history"