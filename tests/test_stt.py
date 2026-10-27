from dataclasses import dataclass

import numpy as np

from castor.stt import transcribe


@dataclass(frozen=True)
class FakeSegment:
    text: str
    no_speech_prob: float = 0.0


class FakeWhisper:
    def __init__(self, segments):
        self._segments = segments
        self.calls = []

    def transcribe(self, audio, **options):
        self.calls.append((audio, options))
        return iter(self._segments), None


def test_segments_are_joined_into_one_phrase():
    model = FakeWhisper([FakeSegment(" Hello there."), FakeSegment(" Over.")])

    assert transcribe(model, b"\x00\x00" * 160, "en") == "Hello there. Over."


def test_segments_that_are_probably_not_speech_are_dropped():
    model = FakeWhisper([FakeSegment(" Thank you.", no_speech_prob=0.9), FakeSegment(" Real words")])

    assert transcribe(model, b"\x00\x00" * 160, "en") == "Real words"


def test_audio_is_passed_as_normalised_floats_in_the_given_language():
    model = FakeWhisper([])
    pcm = np.array([0, 16384, -32768], dtype="<i2").tobytes()

    assert transcribe(model, pcm, "pl") == ""

    audio, options = model.calls[0]
    assert audio.dtype == np.float32
    assert audio.tolist() == [0.0, 0.5, -1.0]
    assert options["language"] == "pl"


def test_keywords_are_passed_as_hotwords():
    model = FakeWhisper([])

    transcribe(model, b"\x00\x00" * 160, "pl", hotwords="bez odbioru, odbiór")

    assert model.calls[0][1]["hotwords"] == "bez odbioru, odbiór"
