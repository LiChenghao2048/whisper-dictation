#!/usr/bin/env python3
from __future__ import annotations

import queue
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from audio import AudioRecorder
from config import Config
from hotkey import HotkeyListener
from server import WhisperServer
from typer import TextTyper

_QUEUE_MAX = 5  # bounded — prevents unbounded memory growth if server is slow


def main() -> None:
    project_root = Path(__file__).parent.parent
    config = Config.load(project_root / "config.yaml")
    binary = project_root / "WhisperKit" / ".build" / "release" / "argmax-cli"

    if not binary.exists():
        sys.exit(f"[whisper-dictation] binary not found at {binary} — run ./setup.sh first")

    recorder = AudioRecorder()
    typer = TextTyper()
    server = WhisperServer(
        binary=binary,
        model=config.model,
        language=config.language,
        task=config.task,
        host=config.server.host,
        port=config.server.port,
    )

    work_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=_QUEUE_MAX)
    stop_event = threading.Event()

    def worker() -> None:
        while not stop_event.is_set():
            try:
                audio = work_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                text = server.transcribe(audio)
                if text:
                    typer.type_text(text)
            except Exception as exc:
                print(f"[whisper-dictation] transcription error: {exc}", file=sys.stderr)
            finally:
                work_queue.task_done()

    def on_start() -> None:
        recorder.start()

    def on_stop() -> None:
        audio = recorder.stop()
        try:
            work_queue.put_nowait(audio)
        except queue.Full:
            print("[whisper-dictation] queue full — audio dropped", file=sys.stderr)

    listener = HotkeyListener(
        key=config.hotkey,
        mode=config.mode,
        on_start=on_start,
        on_stop=on_stop,
    )

    def shutdown(sig=None, frame=None) -> None:
        print("\n[whisper-dictation] shutting down…")
        stop_event.set()
        listener.stop()
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("[whisper-dictation] starting WhisperKit server…")
    server.start()
    print(f"[whisper-dictation] ready — {config.mode} {config.hotkey} to dictate")

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    listener.start()

    while not stop_event.is_set():
        time.sleep(0.5)


if __name__ == "__main__":
    main()
