"""Text-to-speech with Piper; audio is played with an external player (aplay by default)."""

import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Sequence

from piper import PiperVoice

from castor.speakable import to_speakable


class SpeakerError(RuntimeError):
    pass


class PiperSpeaker:
    def __init__(self, voice_path: Path, player: Sequence[str]) -> None:
        self._voice = PiperVoice.load(voice_path)
        self._player = tuple(player)

    def say(self, text: str) -> None:
        """Speak the text and block until playback has finished."""
        speakable = to_speakable(text)
        if not speakable:
            return
        with tempfile.NamedTemporaryFile(suffix=".wav") as wav_file:
            with wave.open(wav_file.name, "wb") as wav:
                self._voice.synthesize_wav(speakable, wav)
            try:
                subprocess.run([*self._player, wav_file.name], check=True)
            except FileNotFoundError as exc:
                raise SpeakerError(f"Audio player not found: {self._player[0]}") from exc
            except subprocess.CalledProcessError as exc:
                raise SpeakerError(f"Audio playback failed ({' '.join(exc.cmd)})") from exc
