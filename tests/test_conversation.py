import queue

import pytest

from castor.config import LANGUAGES
from castor.conversation import (
    KEY,
    MIC_ERROR,
    PHRASE,
    MicrophoneError,
    collect_turn,
    run_conversation,
)

EN = LANGUAGES["en"]


def events_of(*items):
    events = queue.Queue()
    for item in items:
        events.put(item)
    return events


def test_collect_turn_joins_phrases_until_turn_keyword():
    events = events_of((PHRASE, "hello gemini"), (PHRASE, "how are you over"))

    turn = collect_turn(events, EN, show=lambda _: None)

    assert turn.text == "hello gemini how are you"
    assert not turn.ends_conversation
    assert not turn.by_keyboard


def test_collect_turn_ignores_bare_keyword_with_nothing_said():
    events = events_of((PHRASE, "over"), (PHRASE, "actually yes over"))

    turn = collect_turn(events, EN, show=lambda _: None)

    assert turn.text == "actually yes"


def test_collect_turn_end_keyword_finishes_conversation():
    events = events_of((PHRASE, "bye now over and out"))

    turn = collect_turn(events, EN, show=lambda _: None)

    assert turn.text == "bye now"
    assert turn.ends_conversation


def test_enter_ends_turn_with_what_was_heard():
    events = events_of((PHRASE, "something misheard"), (KEY, ""))

    turn = collect_turn(events, EN, show=lambda _: None)

    assert turn.text == "something misheard"
    assert turn.by_keyboard
    assert not turn.ends_conversation


def test_typed_text_is_added_to_turn():
    events = events_of((PHRASE, "hello"), (KEY, "typed correction"))

    turn = collect_turn(events, EN, show=lambda _: None)

    assert turn.text == "hello typed correction"


def test_phrases_finalised_while_stopping_after_enter_are_kept():
    events = events_of((PHRASE, "first"), (KEY, ""))

    def flush():
        events.put((PHRASE, "late words"))  # queued by the mic thread during stop()
        return "tail"

    turn = collect_turn(events, EN, show=lambda _: None, flush=flush)

    assert turn.text == "first late words tail"


def test_q_on_keyboard_ends_conversation():
    events = events_of((PHRASE, "bye"), (KEY, " q "))

    turn = collect_turn(events, EN, show=lambda _: None)

    assert turn.text == "bye"
    assert turn.ends_conversation


def test_microphone_failure_is_raised():
    events = events_of((PHRASE, "hello"), (MIC_ERROR, "parec exited"))

    with pytest.raises(MicrophoneError, match="parec exited"):
        collect_turn(events, EN, show=lambda _: None)


class FakeChat:
    def __init__(self):
        self.sent = []

    def send(self, text):
        self.sent.append(text)
        return f"reply {len(self.sent)}"


class FakeSpeaker:
    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)


class FakeMicrophone:
    """Feeds one scripted user turn into the event queue each time it is started."""

    def __init__(self, turns, leftovers=()):
        self._turns = list(turns)
        self._leftovers = list(leftovers)
        self._pending_leftover = ""
        self.events = queue.Queue()
        self.is_listening = False
        self.starts = 0

    def start(self, sink):
        assert not self.is_listening
        self.is_listening = True
        self.starts += 1
        self._pending_leftover = self._leftovers.pop(0) if self._leftovers else ""
        for item in self._turns.pop(0):
            sink(item)

    def stop(self):
        self.is_listening = False
        leftover, self._pending_leftover = self._pending_leftover, ""
        return leftover


def run(mic, chat=None, speaker=None):
    chat = chat or FakeChat()
    speaker = speaker or FakeSpeaker()
    transcript = run_conversation(
        chat=chat,
        speaker=speaker,
        microphone=mic,
        events=mic.events,
        profile=EN,
        initial_prompt="start",
        show=lambda _: None,
    )
    return chat, speaker, transcript


def test_conversation_alternates_speaking_and_listening_until_end_keyword():
    mic = FakeMicrophone(
        [
            [(PHRASE, "first question over")],
            [(PHRASE, "second question over")],
            [(PHRASE, "over and out")],
        ]
    )

    chat, speaker, transcript = run(mic)

    assert chat.sent == ["start", "first question", "second question"]
    assert speaker.said == ["reply 1", "reply 2", "reply 3"]
    assert mic.starts == 3
    assert not mic.is_listening
    assert transcript == (
        ("user", "start"),
        ("model", "reply 1"),
        ("user", "first question"),
        ("model", "reply 2"),
        ("user", "second question"),
        ("model", "reply 3"),
    )


def test_last_words_before_end_keyword_are_sent_and_answered():
    mic = FakeMicrophone([[(PHRASE, "thanks bye over and out")]])

    chat, speaker, _ = run(mic)

    assert chat.sent == ["start", "thanks bye"]
    assert speaker.said == ["reply 1", "reply 2"]


def test_unfinished_words_are_kept_when_turn_ends_by_keyboard():
    mic = FakeMicrophone([[(PHRASE, "hello"), (KEY, "")], [(KEY, "q")]], leftovers=["there"])

    chat, _, _ = run(mic)

    assert chat.sent == ["start", "hello there"]


def test_words_said_before_q_are_sent_as_last_message():
    mic = FakeMicrophone([[(KEY, "q")]], leftovers=["goodbye"])

    chat, speaker, _ = run(mic)

    assert chat.sent == ["start", "goodbye"]
    assert speaker.said == ["reply 1", "reply 2"]


def test_keys_pressed_while_gemini_speaks_are_discarded():
    mic = FakeMicrophone([[(PHRASE, "real question over")], [(KEY, "q")]])

    class StaleKeySpeaker(FakeSpeaker):
        def say(self, text):
            super().say(text)
            mic.events.put((KEY, "q"))

    chat, _, _ = run(mic, speaker=StaleKeySpeaker())

    assert chat.sent == ["start", "real question"]


def test_messages_are_reported_as_they_happen_so_a_crash_keeps_them():
    mic = FakeMicrophone([[(PHRASE, "question over")]])
    recorded = []

    class FailingChat(FakeChat):
        def send(self, text):
            if self.sent:
                raise RuntimeError("api down")
            return super().send(text)

    with pytest.raises(RuntimeError):
        run_conversation(
            chat=FailingChat(),
            speaker=FakeSpeaker(),
            microphone=mic,
            events=mic.events,
            profile=EN,
            initial_prompt="start",
            show=lambda _: None,
            on_message=lambda role, text: recorded.append((role, text)),
        )

    assert recorded == [("user", "start"), ("model", "reply 1"), ("user", "question")]


def test_microphone_is_stopped_when_chat_fails():
    mic = FakeMicrophone([[(PHRASE, "question over")]])

    class FailingChat(FakeChat):
        def send(self, text):
            if self.sent:
                raise RuntimeError("api down")
            return super().send(text)

    with pytest.raises(RuntimeError):
        run(mic, chat=FailingChat())

    assert not mic.is_listening


def test_send_last_keyword_sends_only_the_last_sentence():
    events = events_of((PHRASE, "misheard rubbish"), (PHRASE, "the real question send last"))

    turn = collect_turn(events, EN, show=lambda _: None)

    assert turn.text == "the real question"
    assert not turn.ends_conversation


def test_bare_send_last_keyword_sends_the_previous_sentence():
    events = events_of((PHRASE, "first try"), (PHRASE, "second try"), (PHRASE, "send last"))

    turn = collect_turn(events, EN, show=lambda _: None)

    assert turn.text == "second try"


def test_bare_send_last_with_nothing_said_keeps_listening():
    events = events_of((PHRASE, "send last"), (PHRASE, "question over"))

    turn = collect_turn(events, EN, show=lambda _: None)

    assert turn.text == "question"


def test_l_on_keyboard_sends_only_the_last_sentence():
    events = events_of((PHRASE, "first"), (PHRASE, "second"), (KEY, " l "))

    turn = collect_turn(events, EN, show=lambda _: None)

    assert turn.text == "second"
    assert turn.by_keyboard
    assert not turn.ends_conversation


def test_l_on_keyboard_counts_words_still_being_recognised_as_last_sentence():
    events = events_of((PHRASE, "first"), (KEY, "l"))

    turn = collect_turn(events, EN, show=lambda _: None, flush=lambda: "unfinished")

    assert turn.text == "unfinished"
