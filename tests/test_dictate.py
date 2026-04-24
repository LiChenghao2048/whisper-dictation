"""Tests for the dictate orchestration layer, using the real make_worker factory."""
import queue
import textwrap
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest
import requests

from dictate import make_worker
from server import WhisperServer
from typer import TextTyper


def _run_worker(mock_server, mock_typer, audios: list, *, queue_max: int = 5):
    """Drive make_worker with a list of audio clips; return after all are processed."""
    work_queue: queue.Queue = queue.Queue(maxsize=queue_max)
    stop_event = threading.Event()

    t = threading.Thread(
        target=make_worker(mock_server, mock_typer, work_queue, stop_event),
        daemon=True,
    )
    t.start()

    for audio in audios:
        work_queue.put(audio)
    work_queue.join()

    stop_event.set()
    t.join(timeout=2)
    return work_queue


# --- happy path ---

def test_worker_transcribes_and_types(mocker):
    mock_server = mocker.MagicMock(spec=WhisperServer)
    mock_server.transcribe.return_value = "hello"
    mock_typer = mocker.MagicMock(spec=TextTyper)

    _run_worker(mock_server, mock_typer, [np.ones(16000, dtype="float32")])

    mock_server.transcribe.assert_called_once()
    mock_typer.type_text.assert_called_once_with("hello")


def test_worker_skips_empty_transcription(mocker):
    mock_server = mocker.MagicMock(spec=WhisperServer)
    mock_server.transcribe.return_value = ""
    mock_typer = mocker.MagicMock(spec=TextTyper)

    _run_worker(mock_server, mock_typer, [np.ones(16000, dtype="float32")])

    mock_typer.type_text.assert_not_called()


# --- queue bounds ---

def test_queue_full_drops_audio():
    work_queue: queue.Queue = queue.Queue(maxsize=2)
    dropped = 0
    for _ in range(3):
        try:
            work_queue.put_nowait(np.zeros(100, dtype="float32"))
        except queue.Full:
            dropped += 1
    assert dropped == 1
    assert work_queue.qsize() == 2


# --- error isolation ---

def test_generic_error_does_not_crash_worker(mocker):
    mock_server = mocker.MagicMock(spec=WhisperServer)
    mock_server.transcribe.side_effect = RuntimeError("boom")
    mock_typer = mocker.MagicMock(spec=TextTyper)

    _run_worker(mock_server, mock_typer, [np.zeros(100, dtype="float32")])

    mock_typer.type_text.assert_not_called()


# --- crash recovery ---

def test_worker_restarts_server_on_network_error_when_process_dead(mocker):
    """When transcribe raises RequestException and is_alive() is False, restart() is called."""
    mock_server = mocker.MagicMock(spec=WhisperServer)
    mock_server.transcribe.side_effect = requests.ConnectionError("refused")
    mock_server.is_alive.return_value = False  # process has died
    mock_typer = mocker.MagicMock(spec=TextTyper)

    _run_worker(mock_server, mock_typer, [np.ones(16000, dtype="float32")])

    mock_server.restart.assert_called_once()
    mock_typer.type_text.assert_not_called()


def test_worker_does_not_restart_on_network_error_when_process_alive(mocker):
    """Transient network error while server is alive must not trigger restart."""
    mock_server = mocker.MagicMock(spec=WhisperServer)
    mock_server.transcribe.side_effect = requests.ConnectionError("timeout")
    mock_server.is_alive.return_value = True  # still running
    mock_typer = mocker.MagicMock(spec=TextTyper)

    _run_worker(mock_server, mock_typer, [np.ones(16000, dtype="float32")])

    mock_server.restart.assert_not_called()


def test_worker_continues_after_failed_restart(mocker):
    """A restart failure must not crash the worker — it should keep processing."""
    mock_server = mocker.MagicMock(spec=WhisperServer)
    mock_server.transcribe.side_effect = [
        requests.ConnectionError("refused"),
        "recovered",
    ]
    mock_server.is_alive.return_value = False
    mock_server.restart.side_effect = RuntimeError("restart failed")
    mock_typer = mocker.MagicMock(spec=TextTyper)

    audio = np.ones(16000, dtype="float32")
    _run_worker(mock_server, mock_typer, [audio, audio])

    assert mock_server.restart.call_count == 1
    mock_typer.type_text.assert_called_once_with("recovered")
