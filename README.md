# Castor

Talk with Gemini out loud. Castor turns your voice into text on your own machine,
sends it to Gemini, and reads Gemini's answer back to you with a local voice. It is
named after one of the Gemini twins: Castor is the half that runs locally.

What Gemini is depends on the system prompt you give it: a job interviewer, a
language tutor, a sparring partner for an idea, or just someone to chat with.

## How a conversation goes

1. Castor sends an opening message (the *initial prompt*) to Gemini and speaks the reply.
2. It then listens. Talk as long as you like; what it hears is shown on screen.
3. Finish your turn by saying **"over"** (Polish: **"odbiór"**). Your words go to
   Gemini, the answer is spoken, and it is your turn again.
4. End the conversation by saying **"over and out"** (**"bez odbioru"**).

Gemini remembers the whole conversation. Everything except Gemini itself runs
locally: speech recognition with [Whisper](https://github.com/SYSTRAN/faster-whisper)
and speech with [Piper](https://github.com/rhasspy/piper). English and Polish are
supported, and the keyboard always works as a fallback when speech recognition
mishears you.

## Usage

```sh
source .venv/bin/activate          # already created; or: python3 -m venv .venv && pip install -r requirements.txt
export GEMINI_API_KEY=...          # https://aistudio.google.com/apikey

python -m castor               # English
python -m castor --lang pl     # Polish
python -m castor --lang pl --transcript rozmowa.txt
python -m castor --system-prompt interviewer.txt --initial-prompt "Start the interview."
```

| Action             | English          | Polish          | Keyboard fallback      |
|--------------------|------------------|-----------------|------------------------|
| Send your turn (all sentences) | "... over" | "... odbiór" | Enter          |
| Send only the last sentence | "... send last" | "... wyślij ostatnie" | `l` + Enter |
| End conversation   | "... over and out" | "... bez odbioru" | `q` + Enter        |
| Fix misheard words | –                | –               | type text + Enter (appended to what was heard) |

Speech is cut into phrases at pauses (~0.75 s) and each phrase is transcribed by
Whisper, so a phrase appears about 1–2 s after you stop talking (CPU, `small` model).
Keywords only count at the end of a phrase (right before a short pause), so
"move it over there" is not treated as a keyword. Words said before the end
keyword are still sent, and Gemini's answer is spoken before the program exits.

The microphone is only open while it is your turn, so Gemini's voice from the
speakers is never transcribed. Ctrl-C stops the program at any time.

### Options

| Option | Default | |
|---|---|---|
| `--lang en\|pl` | `en` (`$CASTOR_LANG`) | language of speech recognition, voice and Gemini replies |
| `--model` | `gemini-2.5-flash` (`$GEMINI_MODEL`) | Gemini model |
| `--system-prompt FILE` | built-in, per language (`castor/config.py`) | file with the Gemini system instruction; replaces the default, so ask for short plain-spoken replies yourself |
| `--initial-prompt TEXT` | text of `prompts/initial_prompt_<lang>.txt` | first message sent to Gemini |
| `--transcript FILE` | – | save the conversation as text |
| `--mic-device NAME` | system default | source from `pactl list short sources` |
| `--player CMD` | `aplay -q` | e.g. `paplay`, `pw-play` |
| `--whisper-model NAME` | `small` (`$WHISPER_MODEL`) | `tiny`, `base`, `small`, `medium`, `large-v3`, or a local model directory; bigger = more accurate, slower |
| `--whisper-device cpu\|cuda` | `cpu` | `cuda` needs CUDA 12 + cuDNN 9 libraries |

### Models (in `models/`, not part of the code)

- Whisper: downloaded automatically into `whisper/` on first run (`small` ≈ 480 MB);
  one multilingual model serves both languages
- Piper: `piper/en_US-lessac-medium.onnx`, `piper/pl_PL-gosia-medium.onnx` (+ `.onnx.json`)
  ([more voices](https://huggingface.co/rhasspy/piper-voices))

Paths are set in `castor/config.py`.

### Layout

```
castor/
  main.py          command line, wiring, keyboard reader
  conversation.py  turn-taking loop (speak → listen → send)
  keywords.py      "over" / "over and out" detection
  gemini_chat.py   one Gemini chat session = whole conversation context
  tts.py           Piper → aplay
  stt.py           parec → phrases → Whisper
  segmenter.py     cuts the microphone stream into phrases at pauses
  speakable.py     strips markdown before speaking
  config.py        per-language models, keywords, system instructions
tests/             pytest, no audio or network needed
```

Run tests: `.venv/bin/pytest -q`