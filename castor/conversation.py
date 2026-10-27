"""The conversation loop: Gemini speaks, the user answers, repeat until the end keyword.

Speech and keyboard input arrive as events on a single queue:
  (PHRASE, text)     - a finished phrase recognised by the microphone
  (KEY, line)        - a line typed on the keyboard (fallback when STT misbehaves)
  (MIC_ERROR, msg)   - the microphone stopped unexpectedly
"""

import queue
from dataclasses import dataclass
from typing import Callable, Protocol

from castor.config import LanguageProfile
from castor.keywords import Signal, match_keyword

PHRASE = "phrase"
KEY = "key"
MIC_ERROR = "mic_error"

QUIT_COMMANDS = frozenset({"q", "quit", "exit"})
LAST_COMMANDS = frozenset({"l", "last"})

Event = tuple[str, str]
Sink = Callable[[Event], None]
Show = Callable[[str], None]
Transcript = tuple[tuple[str, str], ...]


class MicrophoneError(RuntimeError):
    pass


class ChatSession(Protocol):
    def send(self, text: str) -> str: ...


class Speaker(Protocol):
    def say(self, text: str) -> None: ...


class Microphone(Protocol):
    def start(self, sink: Sink) -> None: ...

    def stop(self) -> str:
        """Stop listening and return any words recognised but not yet finalised."""
        ...


@dataclass(frozen=True)
class UserTurn:
    text: str
    ends_conversation: bool
    by_keyboard: bool


def _join(parts: tuple[str, ...]) -> str:
    return " ".join(p for p in parts if p)


def _last_sentence(heard: tuple[str, ...]) -> str:
    sentences = [s for s in heard if s]
    return sentences[-1] if sentences else ""


def _keyboard_turn(heard: tuple[str, ...], line: str) -> UserTurn:
    typed = line.strip()
    if typed.lower() in QUIT_COMMANDS:
        return UserTurn(_join(heard), ends_conversation=True, by_keyboard=True)
    if typed.lower() in LAST_COMMANDS:
        return UserTurn(_last_sentence(heard), ends_conversation=False, by_keyboard=True)
    return UserTurn(_join((*heard, typed)), ends_conversation=False, by_keyboard=True)


def collect_turn(
    events: "queue.Queue[Event]",
    profile: LanguageProfile,
    show: Show,
    flush: Callable[[], str] = lambda: "",
) -> UserTurn:
    """Gather the user's phrases until a keyword, Enter or "q" ends the turn."""
    heard: tuple[str, ...] = ()
    while True:
        kind, payload = events.get()
        if kind == MIC_ERROR:
            raise MicrophoneError(payload)
        if kind == KEY:
            leftover = flush()  # stops the mic, so no more phrases arrive after this
            return _keyboard_turn((*heard, *_pending_phrases(events), leftover), payload)

        match = match_keyword(
            payload, profile.turn_keywords, profile.end_keywords, profile.last_keywords
        )
        if match.text:
            show(f"  … {match.text}")
            heard = (*heard, match.text)
        if match.signal is Signal.END_CONVERSATION:
            return UserTurn(_join(heard), ends_conversation=True, by_keyboard=False)
        if match.signal in (Signal.END_TURN, Signal.END_TURN_LAST):
            if heard:
                send_all = match.signal is Signal.END_TURN
                text = _join(heard) if send_all else _last_sentence(heard)
                return UserTurn(text, ends_conversation=False, by_keyboard=False)
            show("  (nothing heard yet — keep talking)")


def _drain(events: "queue.Queue[Event]") -> tuple[Event, ...]:
    drained: tuple[Event, ...] = ()
    while True:
        try:
            drained = (*drained, events.get_nowait())
        except queue.Empty:
            return drained


def _pending_phrases(events: "queue.Queue[Event]") -> tuple[str, ...]:
    """Phrases the mic finalised while Enter was being handled; other events are dropped."""
    return tuple(payload for kind, payload in _drain(events) if kind == PHRASE)


def _discard_pending(events: "queue.Queue[Event]") -> None:
    """Drop keys pressed while Gemini was speaking, so they don't end the next turn."""
    _drain(events)


def _listen_for_turn(
    microphone: Microphone,
    events: "queue.Queue[Event]",
    profile: LanguageProfile,
    show: Show,
) -> UserTurn:
    hint = (
        f"Listening — say “{profile.turn_keywords[0]}” to send all, "
        f"“{profile.last_keywords[0]}” to send only the last sentence, "
        f"“{profile.end_keywords[0]}” to finish\n"
        "  (keyboard: Enter = send all, l + Enter = send last, q + Enter = quit)"
    )
    while True:
        _discard_pending(events)
        show(hint)
        microphone.start(events.put)
        try:
            turn = collect_turn(events, profile, show, flush=microphone.stop)
        finally:
            microphone.stop()
        if turn.text or turn.ends_conversation:
            return turn
        show("  (nothing heard — try again)")


def run_conversation(
    chat: ChatSession,
    speaker: Speaker,
    microphone: Microphone,
    events: "queue.Queue[Event]",
    profile: LanguageProfile,
    initial_prompt: str,
    show: Show,
    on_message: Callable[[str, str], None] = lambda role, text: None,
) -> Transcript:
    """Run the spoken conversation and return its transcript as (role, text) pairs.

    `on_message` is called for every message as soon as it happens, so callers can
    persist the transcript even if the conversation is interrupted.

    The microphone is only open while the user's turn is being collected, so
    Gemini's own voice from the speakers is never transcribed.
    """
    transcript: Transcript = ()
    message = initial_prompt
    is_last = False
    while True:
        on_message("user", message)
        reply = chat.send(message)
        on_message("model", reply)
        transcript = (*transcript, ("user", message), ("model", reply))
        show(f"\nGemini: {reply}\n")
        speaker.say(reply)
        if is_last:
            return transcript

        turn = _listen_for_turn(microphone, events, profile, show)
        if not turn.text:
            return transcript
        show(f"You: {turn.text}")
        message, is_last = turn.text, turn.ends_conversation
