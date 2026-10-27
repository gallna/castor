import numpy as np

from castor.segmenter import CHUNK_BYTES, CHUNK_S, PhraseSegmenter

SAMPLES_PER_CHUNK = CHUNK_BYTES // 2


def chunk(amplitude: int) -> bytes:
    """One chunk of a square wave (speech-like energy) or silence when amplitude is 0."""
    wave = np.where(np.arange(SAMPLES_PER_CHUNK) % 40 < 20, amplitude, -amplitude)
    return wave.astype("<i2").tobytes()


def chunks(amplitude: int, seconds: float) -> list[bytes]:
    return [chunk(amplitude)] * round(seconds / CHUNK_S)


def feed_all(segmenter: PhraseSegmenter, audio: list[bytes]) -> list[bytes]:
    return [phrase for c in audio for phrase in segmenter.feed(c)]


def test_phrase_is_cut_after_a_pause():
    segmenter = PhraseSegmenter()

    phrases = feed_all(segmenter, chunks(0, 0.5) + chunks(3000, 1.0) + chunks(0, 1.0))

    assert len(phrases) == 1
    assert len(phrases[0]) >= len(b"".join(chunks(3000, 1.0)))


def test_short_gap_inside_speech_does_not_split_phrase():
    segmenter = PhraseSegmenter()

    audio = chunks(3000, 0.5) + chunks(0, 0.25) + chunks(3000, 0.5) + chunks(0, 1.0)

    assert len(feed_all(segmenter, audio)) == 1


def test_two_phrases_separated_by_a_long_pause():
    segmenter = PhraseSegmenter()

    audio = chunks(3000, 0.5) + chunks(0, 1.0) + chunks(3000, 0.5) + chunks(0, 1.0)

    assert len(feed_all(segmenter, audio)) == 2


def test_silence_and_clicks_are_ignored():
    segmenter = PhraseSegmenter()

    audio = chunks(0, 1.0) + chunks(3000, 0.125) + chunks(0, 1.0)

    assert feed_all(segmenter, audio) == []
    assert segmenter.flush() == b""


def test_flush_returns_unfinished_phrase():
    segmenter = PhraseSegmenter()

    assert feed_all(segmenter, chunks(3000, 1.0)) == []
    assert len(segmenter.flush()) > 0
    assert segmenter.flush() == b""


def test_long_speech_is_cut_at_max_length():
    segmenter = PhraseSegmenter(max_phrase_s=2.0)

    assert len(feed_all(segmenter, chunks(3000, 5.0))) == 2


def test_steady_background_noise_is_not_speech():
    segmenter = PhraseSegmenter()

    noise = chunks(400, 3.0)
    speech = chunks(4000, 1.0) + chunks(400, 1.0)

    assert feed_all(segmenter, noise) == []
    assert len(feed_all(segmenter, speech)) == 1


def test_noise_starting_after_quiet_is_learned_and_not_speech():
    segmenter = PhraseSegmenter()

    quiet, hum = chunks(100, 1.0), chunks(400, 30.0)
    phrases = feed_all(segmenter, quiet + hum)

    # The floor catches up within seconds, so the hum ends up silent, not one endless phrase.
    assert len(phrases) <= 1
    assert feed_all(segmenter, chunks(400, 2.0)) == []
    assert len(feed_all(segmenter, chunks(4000, 1.0) + chunks(400, 1.0))) == 1
