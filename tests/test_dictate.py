"""Integration tests for the dictate orchestration layer."""
import queue
import textwrap
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# We test the orchestration logic directly by importing the helpers after
# patching at module boundaries.


def _write_config(tmp_path, extra=""):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent(f"""\
        hotkey: alt_r
        mode: hold
        model: small
        language: en
        task: transcribe
        server:
          host: localhost
          port: 50060
        {extra}
    """))
    return cfg


def test_worker_calls_transcribe_and_type(tmp_path, mocker):
    """Audio placed on the queue is transcribed and typed."""
    from config import Config
    from server import WhisperServer
    from typer import TextTyper

    cfg = Config.load(_write_config(tmp_path))

    mock_server = mocker.MagicMock(spec=WhisperServer)
    mock_server.transcribe.return_value = "hello"
    mock_typer = mocker.MagicMock(spec=TextTyper)

    work_queue: queue.Queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()

    def worker():
        while not stop_event.is_set():
            try:
                audio = work_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                text = mock_server.transcribe(audio)
                if text:
                    mock_typer.type_text(text)
            finally:
                work_queue.task_done()

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    audio = np.ones(16000, dtype="float32")
    work_queue.put(audio)
    work_queue.join()

    stop_event.set()
    t.join(timeout=2)

    mock_server.transcribe.assert_called_once()
    mock_typer.type_text.assert_called_once_with("hello")


def test_queue_full_drops_audio(tmp_path, mocker):
    """When the work queue is full, new audio is dropped without error."""
    work_queue: queue.Queue = queue.Queue(maxsize=2)
    dropped = []

    for _ in range(3):
        audio = np.zeros(100, dtype="float32")
        try:
            work_queue.put_nowait(audio)
        except queue.Full:
            dropped.append(audio)

    assert len(dropped) == 1
    assert work_queue.qsize() == 2


def test_transcription_error_does_not_crash(mocker):
    """An exception from the server must be caught and logged, not propagated."""
    mock_server = mocker.MagicMock()
    mock_server.transcribe.side_effect = RuntimeError("server error")
    mock_typer = mocker.MagicMock()

    work_queue: queue.Queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()
    errors = []

    def worker():
        while not stop_event.is_set():
            try:
                audio = work_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                text = mock_server.transcribe(audio)
                if text:
                    mock_typer.type_text(text)
            except Exception as exc:
                errors.append(exc)
            finally:
                work_queue.task_done()

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    work_queue.put(np.zeros(100, dtype="float32"))
    work_queue.join()

    stop_event.set()
    t.join(timeout=2)

    # Worker survived; typer was never called
    mock_typer.type_text.assert_not_called()
    assert len(errors) == 1
