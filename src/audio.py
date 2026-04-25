from __future__ import annotations

import io
import wave
from typing import Optional

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
WARN_RMS_THRESHOLD = 0.001  # below this on a test recording → likely no mic access


class AudioRecorder:
    def __init__(self, sample_rate: int = SAMPLE_RATE, device: Optional[int] = None) -> None:
        self._sample_rate = sample_rate
        self._device = device
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None

    def check(self, verbose: bool = False) -> None:
        """Record a silent test clip at startup and warn if the mic appears dead."""
        try:
            test = sd.rec(
                int(0.1 * self._sample_rate),
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                device=self._device,
            )
            sd.wait()
            rms = float(np.sqrt(np.mean(test**2)))
            if rms < WARN_RMS_THRESHOLD:
                device_name = sd.query_devices(self._device or sd.default.device[0])["name"]
                print(
                    f"[whisper-dictation] WARNING: mic '{device_name}' appears silent "
                    f"(RMS={rms:.6f}). Check Privacy & Security → Microphone, "
                    f"or set 'device:' in config.yaml to a different input device."
                )
                if verbose:
                    _print_input_devices()
        except Exception as exc:
            print(f"[whisper-dictation] WARNING: mic check failed: {exc}")

    def start(self) -> None:
        self._frames = []
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            device=self._device,
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
        self._frames.append(indata[:, 0].copy())


def _print_input_devices() -> None:
    print("[whisper-dictation] Available input devices:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  [{i}] {d['name']}")
    print("[whisper-dictation] Set 'device: <index>' in config.yaml to choose one.")
