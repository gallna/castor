# Castor

Voice conversation with Gemini: local speech-to-text (faster-whisper) → Gemini → local
text-to-speech (Piper). Turn-taking is walkie-talkie style ("over" / "over and out",
Polish "odbiór" / "bez odbioru"). What Gemini plays is set by `--system-prompt`.
User-facing docs are in README.md.

## Commands

```sh
.venv/bin/pytest -q                      # all tests; no audio, network or models needed
.venv/bin/python -m castor --help
GEMINI_API_KEY=... .venv/bin/python -m castor --lang pl
```

Python 3.10, venv in `.venv/`, pinned deps in `requirements.txt`. `models/` is git-ignored.

## Architecture

Everything meets in one `queue.Queue` of `(kind, payload)` events (`castor/conversation.py`):
`PHRASE` from the microphone; `TYPED`, `SEND_ALL`, `SEND_LAST`, `QUIT` from the keyboard
(commands carry the unsent input-line text); `MIC_ERROR` from mic threads.
`run_conversation` is a plain loop: speak reply → open mic → `collect_turn` → close mic → send.

- `conversation.py` – turn loop; depends only on the `ChatSession`/`Speaker`/`Microphone`
  protocols, so tests use fakes (`tests/test_conversation.py`).
- `stt.py` – `WhisperMicrophone`: `parec` → reader thread → `PhraseSegmenter` → transcriber
  thread → `PHRASE` events. `stop()` is idempotent, delivers already-cut phrases to the sink,
  then returns the transcript of the unfinished phrase.
- `segmenter.py` – energy-based endpointing with an adaptive noise floor; cuts at ~0.75 s pauses.
- `keyboard.py` – prompt_toolkit input line on its own thread, under `patch_stdout` so prints
  appear above it. Ctrl+Enter etc. need kitty/modifyOtherKeys sequences, registered in
  prompt_toolkit's `ANSI_SEQUENCES`; empty Enter and Alt+Enter are the fallbacks. Ctrl+C sends
  SIGINT to the process. Piped stdin falls back to line commands (`line_event`).
- `keywords.py` – keyword only counts at the end of a phrase; diacritics/punctuation ignored.
- `config.py` – per-language `LanguageProfile` (keywords, Piper voice, default system prompt).
- `main.py` – CLI and wiring; heavy imports (Whisper, Piper, Gemini SDK) happen after arg checks.

## Invariants worth keeping

- The mic is closed while Castor speaks, so its own voice is never transcribed. No barge-in.
- Only one thread at a time may call the Whisper model; `stop()` abandons a stuck transcriber
  rather than running the model concurrently.
- Spoken keywords are passed to Whisper as `hotwords`; without them "odbiór" is often misheard.
- Typed text is never checked for keywords; only spoken phrases are.
- Commands given while Gemini speaks are dropped; typed lines are kept for the next turn.
- Gemini replies are spoken, so `speakable.py` strips markdown before TTS.

## Models (`models/`, not code)

- `whisper/` – downloaded on first run (default `small`).
- `piper/` – `.onnx` + `.onnx.json` voices, paths in `config.py`.
- `vosk/` – leftover from the previous STT backend; unused.

## Conventions

- Tests first (pytest, Arrange-Act-Assert, descriptive names); nothing in tests touches audio
  devices or the network.
- Frozen dataclasses and tuples for data passed between parts; module docstring on every file.
- To check STT changes without a microphone, synthesise speech with Piper, resample to 16 kHz,
  and feed it through `PhraseSegmenter` + `stt.transcribe`.
