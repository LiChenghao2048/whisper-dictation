#!/usr/bin/env python3
from __future__ import annotations

import queue
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent))

from audio import AudioRecorder
from cleanup import TextCleaner
from config import Config
from hotkey import HotkeyListener
from server import WhisperServer
from typer import TextTyper

_QUEUE_MAX = 5  # bounded — prevents unbounded memory growth if server is slow
_DEBUG_DIR = Path("/tmp")


def make_worker(
    server: WhisperServer,
    typer: TextTyper,
    work_queue: queue.Queue,
    stop_event: threading.Event,
    debug_audio: bool = False,
    cleaner: TextCleaner | None = None,
):
    """Return the worker callable. Extracted so it can be unit-tested independently."""
    _debug_counter = [0]

    def worker() -> None:
        while not stop_event.is_set():
            try:
                audio = work_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                debug_path = None
                if debug_audio:
                    _debug_counter[0] += 1
                    debug_path = _DEBUG_DIR / f"wd-debug-{_debug_counter[0]:03d}.wav"
                text = server.transcribe(audio, debug_path=debug_path)
                if text and cleaner is not None:
                    try:
                        text = cleaner.clean(text)
                    except Exception as exc:
                        print(f"[whisper-dictation] cleanup error (using raw): {exc}", file=sys.stderr)
                if text:
                    print(f"[whisper-dictation] transcribed: {text!r}", file=sys.stderr)
                    typer.type_text(text)
            except requests.RequestException as exc:
                print(f"[whisper-dictation] network error: {exc}", file=sys.stderr)
                if not server.is_alive():
                    print("[whisper-dictation] server crashed — restarting...", file=sys.stderr)
                    try:
                        server.restart()
                        print("[whisper-dictation] server restarted", file=sys.stderr)
                    except Exception as restart_exc:
                        print(f"[whisper-dictation] restart failed: {restart_exc}", file=sys.stderr)
            except Exception as exc:
                print(f"[whisper-dictation] transcription error: {exc}", file=sys.stderr)
            finally:
                if debug_path is not None and debug_path.exists():
                    print(f"[whisper-dictation] debug audio saved — run: afplay {debug_path}", file=sys.stderr)
                work_queue.task_done()

    return worker


def make_callbacks(
    recorder: AudioRecorder,
    work_queue: queue.Queue,
):
    """Return (on_start, on_stop) callables. Extracted so they can be unit-tested independently."""
    def on_start() -> None:
        try:
            recorder.start()
        except Exception as exc:
            print(f"[whisper-dictation] mic error on start: {exc}", file=sys.stderr)

    def on_stop() -> None:
        try:
            audio = recorder.stop()
        except Exception as exc:
            print(f"[whisper-dictation] mic error on stop: {exc}", file=sys.stderr)
            return
        try:
            work_queue.put_nowait(audio)
        except queue.Full:
            print("[whisper-dictation] queue full — audio dropped", file=sys.stderr)

    return on_start, on_stop


def main() -> None:
    project_root = Path(__file__).parent.parent
    config = Config.load(project_root / "config.yaml")
    binary = project_root / "WhisperKit" / ".build" / "release" / "argmax-cli"

    if not binary.exists():
        sys.exit(f"[whisper-dictation] binary not found at {binary} — run ./setup.sh first")

    recorder = AudioRecorder(device=config.device)
    typer = TextTyper()
    server = WhisperServer(
        binary=binary,
        model=config.model,
        language=config.language,
        task=config.task,
        host=config.server.host,
        port=config.server.port,
        temperature=config.temperature,
        prompt=config.prompt,
        simplified=config.simplified,
    )
    cleaner: TextCleaner | None = None
    if config.cleanup.enabled:
        cleaner = TextCleaner(
            model=config.cleanup.model,
            host=config.cleanup.host,
            port=config.cleanup.port,
        )

    work_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=_QUEUE_MAX)
    stop_event = threading.Event()

    on_start, on_stop = make_callbacks(recorder, work_queue)

    listener = HotkeyListener(
        keys=config.hotkey,
        mode=config.mode,
        on_start=on_start,
        on_stop=on_stop,
    )

    # Signal handler only sets the event — all teardown happens in the main thread
    # after the loop exits, avoiding sys.exit() bypassing cleanup.
    def shutdown(sig=None, frame=None) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("[whisper-dictation] starting WhisperKit server...")
    server.start()

    recorder.check(verbose=config.debug_audio)  # warn early if mic is silent; list devices only in debug mode

    if cleaner is not None:
        if cleaner.is_available():
            print(f"[whisper-dictation] Ollama cleanup enabled ({config.cleanup.model})")
        else:
            print(
                f"[whisper-dictation] WARNING: Ollama not reachable at "
                f"{config.cleanup.host}:{config.cleanup.port} — cleanup disabled for this session",
                file=sys.stderr,
            )
            cleaner = None

    print(f"[whisper-dictation] ready — {config.mode} {'+'.join(config.hotkey)} to dictate")
    if config.debug_audio:
        print(f"[whisper-dictation] debug_audio on — recordings saved to {_DEBUG_DIR}/wd-debug-NNN.wav")

    worker_thread = threading.Thread(
        target=make_worker(server, typer, work_queue, stop_event, config.debug_audio, cleaner),
        daemon=True,
    )
    worker_thread.start()

    listener.start()

    while not stop_event.is_set():
        if not listener.is_alive():
            print("[whisper-dictation] ERROR: hotkey listener died — check Accessibility permissions", file=sys.stderr)
            stop_event.set()
            break
        time.sleep(0.5)

    print("\n[whisper-dictation] shutting down...")
    listener.stop()
    recorder.stop()  # close mic stream if hotkey was held at shutdown
    worker_thread.join(timeout=2)
    if worker_thread.is_alive():
        print("[whisper-dictation] WARNING: worker thread did not exit cleanly", file=sys.stderr)
    server.stop()


if __name__ == "__main__":
    main()
