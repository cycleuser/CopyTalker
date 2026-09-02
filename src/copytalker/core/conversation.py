"""
Conversation turn state machine for continuous dialogue mode.

Models the turn-taking flow of a Xiaoai-style assistant:

    IDLE → LISTENING → PROCESSING → SPEAKING → LISTENING → ...

Thread-safe: all transitions guarded by a lock.  The pipeline and GUI
cooperate via this FSM so that:
- STT is gated while the assistant is speaking (prevents re-capturing TTS).
- The UI can render the current conversational state.
- Barge-in (interrupt) can force SPEAKING → LISTENING.
"""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """States in the conversation turn cycle."""
    IDLE = "idle"                      # not started
    LISTENING = "listening"            # capturing user speech
    PROCESSING = "processing"          # transcribing + translating
    SPEAKING = "speaking"              # playing back translated TTS


# Valid forward transitions.
_TRANSITIONS = {
    ConversationState.IDLE: {ConversationState.LISTENING},
    ConversationState.LISTENING: {ConversationState.PROCESSING, ConversationState.IDLE},
    ConversationState.PROCESSING: {ConversationState.SPEAKING, ConversationState.LISTENING},
    ConversationState.SPEAKING: {ConversationState.LISTENING, ConversationState.IDLE},
}

# Human-readable labels (i18n-friendly).
STATE_LABELS = {
    ConversationState.IDLE: "Idle",
    ConversationState.LISTENING: "Listening…",
    ConversationState.PROCESSING: "Translating…",
    ConversationState.SPEAKING: "Speaking…",
}


class ConversationFSM:
    """Thread-safe conversation turn state machine.

    Listeners are notified on every state change with the new state.
    The pipeline calls ``transition_to`` at each phase; the GUI registers
    a listener to update the UI.
    """

    def __init__(self, initial: ConversationState = ConversationState.IDLE):
        self._state = initial
        self._lock = threading.Lock()
        self._listeners: list[Callable[[ConversationState], None]] = []
        self._barge_in_event = threading.Event()

    @property
    def state(self) -> ConversationState:
        with self._lock:
            return self._state

    def add_listener(self, cb: Callable[[ConversationState], None]) -> None:
        """Register a callback invoked on every state change."""
        self._listeners.append(cb)

    def transition_to(self, new_state: ConversationState) -> bool:
        """Attempt a state transition. Returns True if it happened."""
        with self._lock:
            old = self._state
            if new_state == old:
                return True
            allowed = _TRANSITIONS.get(old, set())
            if new_state not in allowed:
                logger.debug(f"Rejected transition {old.name}→{new_state.name}")
                return False
            self._state = new_state
            listeners = list(self._listeners)
        logger.info(f"Conversation: {old.name} → {new_state.name}")
        for cb in listeners:
            try:
                cb(new_state)
            except Exception as e:
                logger.error(f"Conversation listener error: {e}")
        return True

    def reset(self) -> None:
        """Force back to IDLE (e.g. on stop)."""
        with self._lock:
            self._state = ConversationState.IDLE
            self._barge_in_event.clear()
            listeners = list(self._listeners)
        for cb in listeners:
            try:
                cb(ConversationState.IDLE)
            except Exception:
                pass

    # --- barge-in (interrupt assistant speech) ---
    def request_barge_in(self) -> None:
        """Signal the TTS loop to stop playback early."""
        self._barge_in_event.set()

    def consume_barge_in(self) -> bool:
        """Check & clear a pending barge-in request."""
        return self._barge_in_event.is_set()

    def clear_barge_in(self) -> None:
        self._barge_in_event.clear()

    @property
    def is_speaking(self) -> bool:
        return self.state == ConversationState.SPEAKING

    @property
    def is_listening(self) -> bool:
        return self.state == ConversationState.LISTENING

    @property
    def can_accept_user_speech(self) -> bool:
        """True only when in LISTENING (or IDLE) — used to gate STT."""
        return self.state in (ConversationState.LISTENING, ConversationState.IDLE)