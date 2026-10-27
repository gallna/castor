"""Cuts a live 16 kHz mono stream into phrases at pauses, so Whisper can transcribe each one.

Speech is told apart from silence by loudness relative to an adaptive noise floor,
which keeps working in a room with a steady hum (fan, laptop) without tuning.
"""

from collections import deque
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000
CHUNK_BYTES = 4000  # 16-bit mono samples
CHUNK_S = CHUNK_BYTES / 2 / SAMPLE_RATE  # 0.125 s

MIN_SPEECH_RMS = 250  # quieter than this is never speech (16-bit sample scale)
SPEECH_TO_NOISE = 3.0  # speech must be this many times louder than the noise floor
NOISE_FLOOR_RISE = 0.05  # how quickly the floor follows rising background noise
NOISE_FLOOR_CREEP = 0.005  # same while loud, so a new steady hum is learned within seconds
END_SILENCE_S = 0.75  # a pause this long ends the phrase
MIN_VOICED_S = 0.25  # shorter bursts are clicks or breaths, not words
PRE_ROLL_S = 0.25  # audio kept from before speech starts, so first sounds aren't clipped
MAX_PHRASE_S = 25.0  # Whisper reads at most 30 s at once


def _chunks(seconds: float) -> int:
    return max(1, round(seconds / CHUNK_S))


def _rms(chunk: bytes) -> float:
    samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32)
    return float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0


class PhraseSegmenter:
    def __init__(self, max_phrase_s: float = MAX_PHRASE_S) -> None:
        self._max_chunks = _chunks(max_phrase_s)
        self._noise_floor: Optional[float] = None
        self._pre_roll: deque[bytes] = deque(maxlen=_chunks(PRE_ROLL_S))
        self._phrase: list[bytes] = []
        self._voiced = 0
        self._silent_run = 0

    def feed(self, chunk: bytes) -> list[bytes]:
        """Add a CHUNK_BYTES piece of audio; returns the phrases it completed (0 or 1)."""
        is_loud = self._is_loud(chunk)
        if not self._phrase:
            if is_loud:
                self._phrase = [*self._pre_roll, chunk]
                self._voiced, self._silent_run = 1, 0
            else:
                self._pre_roll.append(chunk)
            return []

        self._phrase.append(chunk)
        if is_loud:
            self._voiced, self._silent_run = self._voiced + 1, 0
        else:
            self._silent_run += 1
        if self._silent_run >= _chunks(END_SILENCE_S) or len(self._phrase) >= self._max_chunks:
            phrase = self.flush()
            return [phrase] if phrase else []
        return []

    def flush(self) -> bytes:
        """End the current phrase now; returns its audio, or b"" if it held no speech."""
        phrase = b"".join(self._phrase) if self._voiced >= _chunks(MIN_VOICED_S) else b""
        self._phrase, self._voiced, self._silent_run = [], 0, 0
        self._pre_roll.clear()
        return phrase

    def _is_loud(self, chunk: bytes) -> bool:
        level = _rms(chunk)
        # Capped at start, so talking right away isn't taken for background noise.
        floor = min(level, MIN_SPEECH_RMS) if self._noise_floor is None else self._noise_floor
        is_loud = level > max(MIN_SPEECH_RMS, floor * SPEECH_TO_NOISE)
        # Drop at once to quieter levels, rise slowly so speech doesn't become "noise".
        rise = NOISE_FLOOR_CREEP if is_loud else NOISE_FLOOR_RISE
        self._noise_floor = level if level < floor else floor + (level - floor) * rise
        return is_loud
