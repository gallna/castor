import queue

import pytest

from castor.config import LANGUAGES
from castor.conversation import (
    MIC_ERROR,
    PHRASE,
    QUIT,
    SEND_ALL,
    SEND_LAST,
    TYPED,
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


def test_send_all_ends_turn_with_what_was_heard():
    events = events_of((PHRASE, "something misheard"), (SEND_ALL, ""))

    turn = collect_turn(events, EN, show=lambda _: None)

    assert turn.text == "something misheard"
    assert turn.by_keyboard
    assert not turn.ends_conversation


def test_typed_lines_are_merged_with_speech_in_order():
    events = events_of((PHRASE, "hello"), (TYPED, "typed correction"), (PHRASE, "more over"))

    turn = collect_turn(events, EN, show=lambda _: None)

    assert turn.text == "hello typed correction more"
    assert not turn.by_keyboard


def test_text_still_in_the_input_line_is_sent_last():
    events = events_of((PHRASE, "hello"), (SEND_ALL, "world"))

    turn = collect_turn(events, EN, show=lambda _: None, flush=lambda: "there")

    assert turn.text == "hello there world"


def test_typed_line_counts_as_something_said_for_bare_keyword():
    events = events_of((TYPED, "a typed question"), (PHRASE, "over"))

    turn = collect_turn(events, EN, show=lambda _: None)

    assert turn.text == "a typed question"


def test_typed_text_is_not_checked_for_keywords():
    events = events_of((TYPED, "game over"), (SEND_ALL, ""))

    turn = collect_turn(events, EN, show=lambda _: None)

    assert turn.text == "game over"


def test_phrases_finalised_while_stopping_after_send_are_kept():
    events = events_of((PHRASE, "first"), (SEND_ALL, "typed"))

    def flush():
        events.put((PHRASE, "late words"))  # queued by the mic thread during stop()
        return "tail"

    turn = collect_turn(events, EN, show=lambda _: None, flush=flush)

    assert turn.text == "first late words tail typed"


def test_quit_on_keyboard_ends_conversation():
    events = events_of((PHRASE, "bye"), (QUIT, ""))

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
    mic = FakeMicrophone([[(PHRASE, "hello"), (SEND_ALL, "")], [(QUIT, "")]], leftovers=["there"])

    chat, _, _ = run(mic)

    assert chat.sent == ["start", "hello there"]


def test_words_said_before_quit_are_sent_as_last_message():
    mic = FakeMicrophone([[(QUIT, "")]], leftovers=["goodbye"])

    chat, speaker, _ = run(mic)

    assert chat.sent == ["start", "goodbye"]
    assert speaker.said == ["reply 1", "reply 2"]


def test_commands_pressed_while_gemini_speaks_are_discarded():
    mic = FakeMicrophone([[(PHRASE, "real question over")], [(QUIT, "")]])

    class StaleKeySpeaker(FakeSpeaker):
        def say(self, text):
            super().say(text)
            mic.events.put((QUIT, ""))

    chat, _, _ = run(mic, speaker=StaleKeySpeaker())

    assert chat.sent == ["start", "real question"]


def test_unsent_text_of_a_command_given_while_gemini_speaks_is_kept():
    mic = FakeMicrophone([[(SEND_ALL, "")], [(QUIT, "")]])

    class TypingSpeaker(FakeSpeaker):
        def say(self, text):
            super().say(text)
            if len(self.said) == 1:
                mic.events.put((SEND_ALL, "typed too early"))

    chat, _, _ = run(mic, speaker=TypingSpeaker())

    assert chat.sent[1] == "typed too early"


def test_lines_typed_while_gemini_speaks_go_into_the_next_turn():
    mic = FakeMicrophone([[(PHRASE, "and spoken over")], [(QUIT, "")]])

    class TypingSpeaker(FakeSpeaker):
        def say(self, text):
            super().say(text)
            if len(self.said) == 1:
                mic.events.put((TYPED, "typed early"))

    chat, _, _ = run(mic, speaker=TypingSpeaker())

    assert chat.sent[1] == "typed early and spoken"


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


def test_send_last_on_keyboard_sends_only_the_last_sentence():
    events = events_of((PHRASE, "first"), (PHRASE, "second"), (SEND_LAST, ""))

    turn = collect_turn(events, EN, show=lambda _: None)

    assert turn.text == "second"
    assert turn.by_keyboard
    assert not turn.ends_conversation


def test_send_last_counts_words_still_being_recognised_as_last_sentence():
    events = events_of((PHRASE, "first"), (SEND_LAST, ""))

    turn = collect_turn(events, EN, show=lambda _: None, flush=lambda: "unfinished")

    assert turn.text == "unfinished"


def test_send_last_with_text_in_the_input_line_sends_that_text():
    events = events_of((PHRASE, "first"), (SEND_LAST, "typed instead"))

    turn = collect_turn(events, EN, show=lambda _: None, flush=lambda: "unfinished")

    assert turn.text == "typed instead"
