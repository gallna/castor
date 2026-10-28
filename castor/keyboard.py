"""Typed input next to the microphone: an input line whose keys become conversation events.

  Enter               add the typed line to the turn (on an empty line: send all)
  Ctrl+Enter          send everything heard and typed
  Ctrl+Alt+Enter      send only the last line (Alt+Enter works too)
  F10 or Ctrl+D       send what was heard and typed, then finish

Most terminals send the same byte for Enter and Ctrl+Enter, so on start the
terminal is asked for an extended key encoding (kitty keyboard protocol and
xterm modifyOtherKeys) and both are decoded here. Terminals that support
neither still work through the fallbacks: empty Enter and Alt+Enter.
"""

import atexit
import os
import signal
import string
import sys
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.input import Input
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import Output
from prompt_toolkit.patch_stdout import patch_stdout

from castor.conversation import QUIT, SEND_ALL, SEND_LAST, TYPED, Event, Sink

PROMPT = "› "
START_TIMEOUT_S = 5
STOP_TIMEOUT_S = 2

# Stand-ins for the combinations prompt_toolkit can't name: no sequence maps to these two,
# since their legacy encodings clash with cursor position reports.
_CTRL_ENTER = Keys.ControlF3
_CTRL_ALT_ENTER = Keys.ControlF15
_LOCK_BITS = (0, 64, 128, 192)  # Caps Lock / Num Lock may be added to the modifier bits

_ENABLE_EXTENDED_KEYS = "\x1b[>1u" "\x1b[>4;2m"  # kitty: disambiguate; xterm: modifyOtherKeys 2
_RESTORE_KEYS = "\x1b[<u" "\x1b[>4m"

# Piped (non-terminal) input keeps the old line commands.
_LINE_COMMANDS = {"": SEND_ALL, "l": SEND_LAST, "last": SEND_LAST, "q": QUIT, "quit": QUIT, "exit": QUIT}


def _extended_key_sequences() -> dict[str, Keys]:
    """Kitty (CSI code;mods u) and modifyOtherKeys (CSI 27;mods;code ~) forms of the keys we use.

    With these modes on, Ctrl+letter and Esc also change encoding, so they are mapped back.
    """
    def both(code: int, mods: int, key) -> dict:
        return {
            sequence: key
            for lock in _LOCK_BITS
            for sequence in (f"\x1b[{code};{mods + lock}u", f"\x1b[27;{mods + lock};{code}~")
        }

    enter = {
        **both(13, 2, Keys.ControlM),  # Shift+Enter behaves as Enter
        **both(13, 5, _CTRL_ENTER),
        **both(13, 3, _CTRL_ALT_ENTER),  # Alt+Enter
        **both(13, 7, _CTRL_ALT_ENTER),
    }
    ctrl_letters = {
        sequence: key
        for letter in string.ascii_lowercase
        for sequence, key in both(ord(letter), 5, Keys(f"c-{letter}")).items()
    }
    # Alt+key arrives as Esc then the key in legacy mode; reproduce that (word motions etc.).
    alt_keys = {
        sequence: key
        for code in range(ord("!"), ord("~") + 1)
        for sequence, key in both(code, 3, (Keys.Escape, chr(code))).items()
    }
    return {**alt_keys, **enter, **ctrl_letters, "\x1b[27u": Keys.Escape}


ANSI_SEQUENCES.update(_extended_key_sequences())


def line_event(line: str) -> Event:
    """Event for a line read from piped input, where keys can't be told apart."""
    command = _LINE_COMMANDS.get(line.strip().lower())
    return (command, "") if command else (TYPED, line.strip())


def _interrupt_main() -> None:
    os.kill(os.getpid(), signal.SIGINT)  # unlike interrupt_main(), also wakes blocking waits


def _key_bindings(sink: Sink, interrupt: Callable[[], None]) -> KeyBindings:
    bindings = KeyBindings()

    def emit(kind: str, event) -> None:
        text = event.current_buffer.text
        event.current_buffer.reset()
        sink((kind, text))

    @bindings.add(Keys.Enter, eager=True)
    def _enter(event) -> None:
        emit(TYPED if event.current_buffer.text.strip() else SEND_ALL, event)

    @bindings.add(_CTRL_ENTER, eager=True)
    def _send_all(event) -> None:
        emit(SEND_ALL, event)

    @bindings.add(_CTRL_ALT_ENTER, eager=True)
    @bindings.add(Keys.Escape, Keys.Enter, eager=True)
    def _send_last(event) -> None:
        emit(SEND_LAST, event)

    @bindings.add(Keys.F10, eager=True)
    @bindings.add(Keys.ControlD, eager=True)
    def _quit(event) -> None:
        emit(QUIT, event)

    @bindings.add(Keys.ControlC, eager=True)
    def _interrupt(event) -> None:
        interrupt()

    return bindings


class TerminalKeyboard:
    def __init__(
        self,
        sink: Sink,
        input: Optional[Input] = None,
        output: Optional[Output] = None,
        interrupt: Callable[[], None] = _interrupt_main,
    ) -> None:
        self._session = PromptSession(
            PROMPT, key_bindings=_key_bindings(sink, interrupt), input=input, output=output
        )
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        output = self._session.app.output
        output.write_raw(_ENABLE_EXTENDED_KEYS)
        output.flush()
        # Backstop for exits that skip close(): a second Ctrl+C, SIGTERM.
        atexit.register(self._restore_keys)
        self._thread = threading.Thread(target=self._run, daemon=True, name="keyboard")
        self._thread.start()
        if not self._wait_until_running(self._thread):
            self.close()
            raise RuntimeError("keyboard input did not start")

    def _run(self) -> None:
        try:
            self._session.prompt(handle_sigint=False)
        except Exception as exc:  # noqa: BLE001 - typing is optional; speech keeps working
            print(f"Keyboard input stopped: {exc!r}", file=sys.stderr)

    def _wait_until_running(self, thread: threading.Thread) -> bool:
        """Wait until the input loop can be told to exit (raw mode on, event loop set)."""
        app, deadline = self._session.app, time.monotonic() + START_TIMEOUT_S
        while thread.is_alive() and time.monotonic() < deadline:
            if app.is_running and app.loop is not None:
                return True
            time.sleep(0.01)
        return False

    def _restore_keys(self) -> None:
        output = self._session.app.output
        output.write_raw(_RESTORE_KEYS)
        output.flush()

    def close(self) -> None:
        """Stop reading keys and give the terminal back in its normal state; safe to repeat."""
        thread, self._thread = self._thread, None
        if thread is None:
            return
        app = self._session.app
        if self._wait_until_running(thread):
            app.loop.call_soon_threadsafe(lambda: app.exit() if app.is_running else None)
        thread.join(timeout=STOP_TIMEOUT_S)
        self._restore_keys()
        atexit.unregister(self._restore_keys)


def _read_piped_lines(sink: Sink) -> None:
    for line in sys.stdin:
        sink(line_event(line))


@contextmanager
def typed_input(sink: Sink) -> Iterator[None]:
    """Deliver typed input to `sink` while the block runs.

    In a terminal, the input line stays at the bottom and everything printed in
    the block appears above it. Piped input is read line by line instead.
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        threading.Thread(target=_read_piped_lines, args=(sink,), daemon=True, name="keyboard").start()
        yield
        return
    with patch_stdout():
        keyboard = TerminalKeyboard(sink)
        try:
            keyboard.start()
        except RuntimeError as exc:
            print(f"Typing is off ({exc}); speech still works.", file=sys.stderr)
        try:
            yield
        finally:
            keyboard.close()
