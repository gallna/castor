"""Command-line entry point: wires Gemini, Piper and Whisper together."""

import argparse
import os
import queue
import shlex
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

from castor.config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_PLAYER,
    DEFAULT_WHISPER_MODEL,
    LANGUAGES,
    WHISPER_DIR,
)
from castor.conversation import MicrophoneError, run_conversation
from castor.keyboard import typed_input


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="castor", description="Castor: talk with Gemini using your voice.")
    parser.add_argument(
        "--lang",
        choices=sorted(LANGUAGES),
        default=os.environ.get("CASTOR_LANG", "en"),
        help="conversation language (default: en, or $CASTOR_LANG)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        help=f"Gemini model (default: {DEFAULT_GEMINI_MODEL}, or $GEMINI_MODEL)",
    )
    parser.add_argument(
        "--system-prompt",
        type=Path,
        metavar="FILE",
        help="file with the Gemini system instruction (default: built-in, per language)",
    )
    parser.add_argument(
        "--initial-prompt",
        metavar="TEXT",
        help="first message sent to Gemini (default: prompts/initial_prompt_<lang>.txt)",
    )
    parser.add_argument("--transcript", type=Path, help="save the conversation to this text file")
    parser.add_argument("--mic-device", help="PulseAudio/PipeWire source name (see `pactl list short sources`)")
    parser.add_argument("--player", default=DEFAULT_PLAYER, help=f"audio player command (default: {DEFAULT_PLAYER!r})")
    parser.add_argument(
        "--whisper-model",
        default=os.environ.get("WHISPER_MODEL", DEFAULT_WHISPER_MODEL),
        help=f"Whisper model name (tiny, base, small, medium, large-v3…) or local directory "
        f"(default: {DEFAULT_WHISPER_MODEL}, or $WHISPER_MODEL)",
    )
    parser.add_argument(
        "--whisper-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="run Whisper on the CPU or an NVIDIA GPU (needs CUDA 12 + cuDNN 9; default: cpu)",
    )
    return parser.parse_args(argv)


def _fail(message: str) -> int:
    print(f"Error: {message}", file=sys.stderr)
    return 2


def transcript_writer(path: Path) -> Callable[[str, str], None]:
    """Start an empty transcript file and append each message as it happens."""
    names = {"user": "You", "model": "Gemini"}
    path.write_text("", encoding="utf-8")

    def append(role: str, text: str) -> None:
        with path.open("a", encoding="utf-8") as file:
            file.write(f"{names[role]}: {text}\n\n")

    return append


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    profile = LANGUAGES[args.lang]

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return _fail("set GEMINI_API_KEY (get one at https://aistudio.google.com/apikey)")

    default_prompt_file = profile.initial_prompt_file if args.initial_prompt is None else None
    required = (profile.piper_voice, args.system_prompt, default_prompt_file)
    missing = [p for p in required if p is not None and not p.exists()]
    if missing:
        return _fail("missing file(s): " + ", ".join(str(p) for p in missing))

    initial_prompt = (
        default_prompt_file.read_text(encoding="utf-8") if default_prompt_file else args.initial_prompt
    ).strip()
    if not initial_prompt:
        return _fail("initial prompt is empty")
    system_prompt = (
        args.system_prompt.read_text(encoding="utf-8").strip()
        if args.system_prompt
        else profile.system_instruction
    )
    if not system_prompt:
        return _fail(f"system prompt file is empty: {args.system_prompt}")

    # Imported here so --help and config errors don't wait for heavy libraries.
    from castor.gemini_chat import ChatError, GeminiChat
    from castor.stt import WhisperMicrophone
    from castor.tts import PiperSpeaker, SpeakerError

    print(f"Loading speech models (Whisper {args.whisper_model}; downloaded on first use)…")
    speaker = PiperSpeaker(profile.piper_voice, shlex.split(args.player))
    try:
        microphone = WhisperMicrophone(
            args.whisper_model,
            profile.code,
            WHISPER_DIR,
            keywords=(*profile.end_keywords, *profile.last_keywords, *profile.turn_keywords),
            device=args.mic_device,
            compute_device=args.whisper_device,
        )
    except MicrophoneError as exc:
        return _fail(str(exc))
    chat = GeminiChat(api_key, args.model, system_prompt)

    events: queue.Queue = queue.Queue()
    try:
        with typed_input(events.put):
            run_conversation(
                chat=chat,
                speaker=speaker,
                microphone=microphone,
                events=events,
                profile=profile,
                initial_prompt=initial_prompt,
                show=print,
                on_message=transcript_writer(args.transcript) if args.transcript else lambda role, text: None,
            )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except (ChatError, SpeakerError, MicrophoneError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        microphone.stop()
        if args.transcript:
            print(f"Transcript saved to {args.transcript}")

    print("Conversation finished.")
    return 0
