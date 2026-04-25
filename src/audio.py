import sys

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000


class AudioRecorder:
    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self._sample_rate = sample_rate
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None

    def start(self) -> None:
        if self._stream is not None:
            return
        self._frames = []
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        audio = np.concatenate(self._frames) if self._frames else np.zeros(0, dtype="float32")
        self._frames = []  # release memory immediately
        return audio

    def _callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            print(f"[whisper-dictation] audio overrun: {status}", file=sys.stderr)
        self._frames.append(indata[:, 0].copy())
