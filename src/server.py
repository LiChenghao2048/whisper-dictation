from __future__ import annotations

import io
import subprocess
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import zhconv

SAMPLE_RATE = 16_000
MIN_AUDIO_SECONDS = 0.1


class WhisperServer:
    def __init__(
        self,
        binary: Path,
        model: str,
        language: Optional[str],
        task: str,
        host: str,
        port: int,
        temperature: float = 0.0,
        prompt: Optional[str] = None,
        simplified: bool = False,
    ) -> None:
        self._cmd = [
            str(binary), "serve",
            "--model", model,
            "--task", task,
            "--host", host,
            "--port", str(port),
        ]
        if language:
            self._cmd += ["--language", language]
        self._temperature = temperature
        self._prompt = prompt
        self._simplified = simplified
        self._endpoint = (
            "/v1/audio/translations"
            if task == "translate"
            else "/v1/audio/transcriptions"
        )
        self._base_url = f"http://{host}:{port}"
        self._process: subprocess.Popen | None = None

    def start(self, timeout: float = 150.0) -> None:
        self._process = subprocess.Popen(
            self._cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            proc = self._process  # local snapshot — stop() may clear _process concurrently
            if proc is None:
                return  # stop() was called while we were starting
            if proc.poll() is not None:
                exit_code = proc.returncode
                proc.wait()  # reap so the OS entry is fully cleaned up
                self._process = None
                raise RuntimeError(
                    f"WhisperKit server exited during startup (exit code {exit_code})"
                )
            try:
                r = requests.get(f"{self._base_url}/health", timeout=2)
                if r.status_code == 200:
                    return
            except requests.RequestException:
                # Covers ConnectionError, ReadTimeout, and any other transport error
                pass
            time.sleep(1)
        self.stop()
        raise TimeoutError(
            f"WhisperKit server did not become healthy within {timeout:.0f}s"
        )

    def stop(self) -> None:
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None

    def is_alive(self) -> bool:
        proc = self._process  # snapshot to avoid TOCTOU if stop() clears _process concurrently
        return proc is not None and proc.poll() is None

    def restart(self, timeout: float = 150.0) -> None:
        self.stop()
        self.start(timeout=timeout)

    def transcribe(self, audio: np.ndarray, debug_path: Optional[Path] = None) -> str:
        if len(audio) < SAMPLE_RATE * MIN_AUDIO_SECONDS:
            return ""
        wav_bytes = _to_wav_bytes(audio)
        if debug_path is not None:
            debug_path.write_bytes(wav_bytes)
        form: dict = {"model": "whisper-1", "temperature": str(self._temperature)}
        if self._prompt:
            form["prompt"] = self._prompt
        r = requests.post(
            f"{self._base_url}{self._endpoint}",
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data=form,
            timeout=30,
        )
        r.raise_for_status()
        text = r.json().get("text", "").strip()
        if self._simplified and text:
            text = zhconv.convert(text, "zh-hans")
        return text


def _to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    # Scale by 32768 so -1.0 maps to -32768 and clip ensures +1.0 stays at 32767
    pcm = np.clip(audio * 32768, -32768, 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()
