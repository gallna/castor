import queue

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from castor.conversation import QUIT, SEND_ALL, SEND_LAST, TYPED
from castor.keyboard import TerminalKeyboard, line_event

CTRL_ENTER = ("\x1b[13;5u", "\x1b[27;5;13~")  # kitty protocol, xterm modifyOtherKeys
CTRL_ALT_ENTER = ("\x1b[13;7u", "\x1b[27;7;13~")
ALT_ENTER = "\x1b\r"  # what Ctrl+Alt+Enter sends in terminals without either mode
F10 = "\x1b[21~"
WAIT_S = 5


@pytest.fixture
def keyboard():
    events: queue.Queue = queue.Queue()
    interrupts = []
    with create_pipe_input() as pipe:
        board = TerminalKeyboard(
            events.put, input=pipe, output=DummyOutput(), interrupt=lambda: interrupts.append(1)
        )
        board.start()
        try:
            yield pipe, events, interrupts
        finally:
            board.close()


def next_event(events):
    return events.get(timeout=WAIT_S)


def test_enter_adds_the_typed_line(keyboard):
    pipe, events, _ = keyboard

    pipe.send_text("zażółć gęślą jaźń\r")

    assert next_event(events) == (TYPED, "zażółć gęślą jaźń")


@pytest.mark.parametrize("sequence", CTRL_ENTER)
def test_ctrl_enter_sends_all_with_the_unfinished_line(keyboard, sequence):
    pipe, events, _ = keyboard

    pipe.send_text("and this" + sequence)

    assert next_event(events) == (SEND_ALL, "and this")


@pytest.mark.parametrize("sequence", (*CTRL_ALT_ENTER, ALT_ENTER))
def test_ctrl_alt_enter_sends_only_the_last_line(keyboard, sequence):
    pipe, events, _ = keyboard

    pipe.send_text(sequence)

    assert next_event(events) == (SEND_LAST, "")


@pytest.mark.parametrize("sequence", (F10, "\x04"))
def test_f10_or_ctrl_d_quits(keyboard, sequence):
    pipe, events, _ = keyboard

    pipe.send_text("bye" + sequence)

    assert next_event(events) == (QUIT, "bye")


def test_empty_enter_sends_all_where_ctrl_enter_looks_like_enter(keyboard):
    pipe, events, _ = keyboard

    pipe.send_text("first\r\r")

    assert next_event(events) == (TYPED, "first")
    assert next_event(events) == (SEND_ALL, "")


def test_input_line_is_cleared_after_each_event(keyboard):
    pipe, events, _ = keyboard

    pipe.send_text("one\rtwo" + CTRL_ENTER[0])

    assert next_event(events) == (TYPED, "one")
    assert next_event(events) == (SEND_ALL, "two")


def test_ctrl_c_interrupts(keyboard):
    pipe, events, interrupts = keyboard

    pipe.send_text("\x03x\r")

    assert next_event(events) == (TYPED, "x")
    assert interrupts == [1]


@pytest.mark.parametrize(
    "line, event",
    [
        ("", (SEND_ALL, "")),
        (" l ", (SEND_LAST, "")),
        ("q", (QUIT, "")),
        ("some text", (TYPED, "some text")),
    ],
)
def test_piped_lines_map_to_events(line, event):
    assert line_event(line) == event


@pytest.mark.parametrize("lock", (64, 128, 192))  # Caps Lock, Num Lock, both
def test_ctrl_enter_works_with_lock_keys_on(keyboard, lock):
    pipe, events, _ = keyboard

    pipe.send_text(f"x\x1b[13;{5 + lock}u")

    assert next_event(events) == (SEND_ALL, "x")


@pytest.mark.parametrize("sequence", ("\x1b[98;3u", "\x1b[27;3;98~"))
def test_alt_letter_in_extended_modes_does_not_insert_garbage(keyboard, sequence):
    pipe, events, _ = keyboard

    pipe.send_text("word" + sequence + "\r")

    assert next_event(events) == (TYPED, "word")


@pytest.mark.parametrize("sequence", ("\x1b[23;2~", "\x1b[24;2~"))  # Shift+F11, Shift+F12
def test_other_function_keys_do_not_send(keyboard, sequence):
    pipe, events, _ = keyboard

    pipe.send_text("x" + sequence + "\r")

    assert next_event(events) == (TYPED, "x")


def test_close_right_after_start_stops_the_input_thread():
    with create_pipe_input() as pipe:
        board = TerminalKeyboard(lambda event: None, input=pipe, output=DummyOutput())
        board.start()
        thread = board._thread

        board.close()

        assert not thread.is_alive()


def test_stand_in_keys_are_not_produced_by_any_other_sequence():
    from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES

    from castor.keyboard import _CTRL_ALT_ENTER, _CTRL_ENTER, _extended_key_sequences

    ours = _extended_key_sequences()
    others = [key for sequence, key in ANSI_SEQUENCES.items() if sequence not in ours]
    assert _CTRL_ENTER not in others
    assert _CTRL_ALT_ENTER not in others
