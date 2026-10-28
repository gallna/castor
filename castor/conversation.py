"""The conversation loop: Gemini speaks, the user answers, repeat until the end keyword.

Speech and keyboard input arrive as events on a single queue:
  (PHRASE, text)     - a finished phrase recognised by the microphone
  (TYPED, line)      - a line typed and confirmed with Enter; merged with the speech
  (SEND_ALL, text)   - keyboard: send everything heard and typed
  (SEND_LAST, text)  - keyboard: send only the last line
  (QUIT, text)       - keyboard: send what was heard and typed, then finish
  (MIC_ERROR, msg)   - the microphone stopped unexpectedly
The keyboard commands carry the text still in the input line, which comes last.
"""

import queue
from dataclasses import dataclass
from typing import Callable, Protocol

from castor.config import LanguageProfile
from castor.keywords import Signal, match_keyword

PHRASE = "phrase"
TYPED = "typed"
SEND_ALL = "send_all"
SEND_LAST = "send_last"
QUIT = "quit"
MIC_ERROR = "mic_error"

COMMANDS = frozenset({SEND_ALL, SEND_LAST, QUIT})

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


def _keyboard_turn(heard: tuple[str, ...], command: str) -> UserTurn:
    if command == SEND_LAST:
        return UserTurn(_last_sentence(heard), ends_conversation=False, by_keyboard=True)
    return UserTurn(_join(heard), ends_conversation=command == QUIT, by_keyboard=True)


def collect_turn(
    events: "queue.Queue[Event]",
    profile: LanguageProfile,
    show: Show,
    flush: Callable[[], str] = lambda: "",
) -> UserTurn:
    """Gather spoken phrases and typed lines until a keyword or keyboard command ends the turn."""
    heard: tuple[str, ...] = ()
    while True:
        kind, payload = events.get()
        if kind == MIC_ERROR:
            raise MicrophoneError(payload)
        if kind in COMMANDS:
            leftover = flush()  # stops the mic, so no more phrases arrive after this
            pending = _pending_lines(events)
            return _keyboard_turn((*heard, *pending, leftover, payload.strip()), kind)
        if kind == TYPED:
            if payload.strip():
                show(f"  ✎ {payload.strip()}")
                heard = (*heard, payload.strip())
            continue

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


def _pending_lines(events: "queue.Queue[Event]") -> tuple[str, ...]:
    """Phrases and typed lines that arrived while a command was handled; others are dropped."""
    return tuple(payload.strip() for kind, payload in _drain(events) if kind in (PHRASE, TYPED))


def _discard_commands(events: "queue.Queue[Event]") -> None:
    """Drop commands given while Gemini was speaking, so they don't end the next turn.

    Lines typed meanwhile, including the text a dropped command carried, are kept
    for the next turn.
    """
    for kind, payload in _drain(events):
        if kind in (TYPED, *COMMANDS) and payload.strip():
            events.put((TYPED, payload))


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
        "  (or type: Enter = add line, Ctrl+Enter = send all, "
        "Ctrl+Alt+Enter = send last, F10 = quit)"
    )
    while True:
        _discard_commands(events)
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
