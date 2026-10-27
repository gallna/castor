"""Speech-to-text with Whisper (faster-whisper), recording the microphone through `parec`.

Whisper transcribes whole pieces of audio rather than a live stream, so the
recording is cut into phrases at pauses (see segmenter.py) and each phrase is
transcribed on a separate thread while recording goes on.
"""

import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from castor.conversation import MIC_ERROR, PHRASE, MicrophoneError, Sink
from castor.segmenter import CHUNK_BYTES, SAMPLE_RATE, PhraseSegmenter

STOP_TIMEOUT_S = 2
TRANSCRIBE_TIMEOUT_S = 30  # waiting for phrases still being transcribed when listening stops
NO_SPEECH_PROB = 0.6  # Whisper invents "Thank you." on noise; such segments score high here
_END = None  # tells the transcriber thread to finish


def transcribe(model, audio: bytes, language: str, hotwords: Optional[str] = None) -> str:
    samples = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0
    segments, _info = model.transcribe(
        samples,
        language=language,
        vad_filter=True,  # skips silence inside the phrase, which otherwise invites made-up text
        condition_on_previous_text=False,
        without_timestamps=True,
        hotwords=hotwords,  # makes the spoken keywords much more likely to be spelled right
    )
    texts = (s.text.strip() for s in segments if s.no_speech_prob < NO_SPEECH_PROB)
    return " ".join(t for t in texts if t)


class WhisperMicrophone:
    def __init__(
        self,
        model_name: str,
        language: str,
        download_dir: Path,
        keywords: tuple[str, ...] = (),
        device: Optional[str] = None,
        compute_device: str = "cpu",
    ) -> None:
        from faster_whisper import WhisperModel  # slow import; keeps tests light

        try:
            self._model = WhisperModel(
                model_name,
                device=compute_device,
                compute_type="int8",
                download_root=str(download_dir),
            )
        except Exception as exc:  # noqa: BLE001 - download, CUDA and file errors look alike to the user
            raise MicrophoneError(f"Cannot load Whisper model {model_name!r}: {exc}") from exc
        self._language = language
        self._hotwords = ", ".join(keywords) or None
        self._device = device
        self._stopping = threading.Event()
        self._abandoned = threading.Event()
        self._process: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self._transcriber: Optional[threading.Thread] = None
        self._segmenter: Optional[PhraseSegmenter] = None
        self._phrases: Optional[queue.Queue] = None

    def _record_command(self) -> list[str]:
        command = [
            "parec",
            "--format=s16le",
            f"--rate={SAMPLE_RATE}",
            "--channels=1",
            "--latency-msec=50",
        ]
        return [*command, f"--device={self._device}"] if self._device else command

    def start(self, sink: Sink) -> None:
        if self._process is not None:
            raise MicrophoneError("Microphone is already listening")
        try:
            # stderr is discarded: an undrained pipe could fill up and stall parec.
            process = subprocess.Popen(
                self._record_command(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
        except OSError as exc:
            raise MicrophoneError(f"Cannot start `parec` (install pulseaudio-utils?): {exc}") from exc
        # Fresh objects per session, so a slow-to-exit old thread can't be revived.
        stopping = threading.Event()
        segmenter = PhraseSegmenter()
        phrases: queue.Queue = queue.Queue()
        reader = threading.Thread(
            target=self._record,
            args=(process, segmenter, phrases, sink, stopping),
            daemon=True,
            name="mic",
        )
        abandoned = threading.Event()
        transcriber = threading.Thread(
            target=self._transcribe_phrases,
            args=(phrases, sink, stopping, abandoned),
            daemon=True,
            name="whisper",
        )
        self._process, self._segmenter, self._phrases = process, segmenter, phrases
        self._reader, self._transcriber, self._stopping = reader, transcriber, stopping
        self._abandoned = abandoned
        transcriber.start()
        reader.start()

    @staticmethod
    def _record(
        process: subprocess.Popen,
        segmenter: PhraseSegmenter,
        phrases: queue.Queue,
        sink: Sink,
        stopping: threading.Event,
    ) -> None:
        try:
            while not stopping.is_set():
                data = process.stdout.read(CHUNK_BYTES)
                if not data:
                    break
                for phrase in segmenter.feed(data):
                    phrases.put(phrase)
            problem = f"parec exited with code {process.poll()}"
        except Exception as exc:  # noqa: BLE001 - reported to the main thread below
            problem = f"recording failed: {exc!r}"
        if not stopping.is_set():
            sink((MIC_ERROR, f"Microphone stopped unexpectedly: {problem}"))

    def _transcribe_phrases(
        self, phrases: queue.Queue, sink: Sink, stopping: threading.Event, abandoned: threading.Event
    ) -> None:
        try:
            for audio in iter(phrases.get, _END):
                text = transcribe(self._model, audio, self._language, self._hotwords)
                if abandoned.is_set():
                    return  # stop() gave up waiting; the turn this belongs to is over
                if text:
                    sink((PHRASE, text))
        except Exception as exc:  # noqa: BLE001 - reported to the main thread below
            if not stopping.is_set():
                sink((MIC_ERROR, f"Speech recognition failed: {exc!r}"))
            else:
                print(f"Speech recognition failed while stopping: {exc!r}", file=sys.stderr)

    def stop(self) -> str:
        """Stop recording; safe to call repeatedly. Returns the words of the unfinished phrase.

        Phrases already cut are still transcribed and delivered to the sink first.
        """
        process, reader, transcriber = self._process, self._reader, self._transcriber
        segmenter, phrases = self._segmenter, self._phrases
        if process is None:
            return ""
        self._stopping.set()
        process.terminate()
        try:
            process.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        reader.join(timeout=STOP_TIMEOUT_S)
        process.stdout.close()
        self._process = self._reader = self._transcriber = self._segmenter = self._phrases = None
        # The segmenter is only safe to touch once the reader thread has finished with it.
        leftover = b"" if reader.is_alive() else segmenter.flush()
        phrases.put(_END)
        transcriber.join(timeout=TRANSCRIBE_TIMEOUT_S)
        if transcriber.is_alive():
            # Never run the model on two threads at once; drop its late results instead.
            self._abandoned.set()
            return ""
        if not leftover:
            return ""
        try:
            return transcribe(self._model, leftover, self._language, self._hotwords)
        except Exception as exc:  # noqa: BLE001 - surfaced as the error callers already handle
            raise MicrophoneError(f"Speech recognition failed: {exc!r}") from exc
