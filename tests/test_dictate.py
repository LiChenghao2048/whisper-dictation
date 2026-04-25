"""Tests for the dictate orchestration layer, using the real make_worker factory."""
import queue
import textwrap
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest
import requests

from dictate import make_callbacks, make_worker
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

def test_on_stop_logs_warning_and_does_not_block_when_queue_full(mocker, capsys):
    """on_stop must print a warning and not block when the work queue is full."""
    from audio import AudioRecorder
    mock_recorder = mocker.MagicMock(spec=AudioRecorder)
    mock_recorder.stop.return_value = np.zeros(16000, dtype="float32")

    work_queue: queue.Queue = queue.Queue(maxsize=1)
    work_queue.put_nowait(np.zeros(16000, dtype="float32"))  # fill queue

    _, on_stop = make_callbacks(mock_recorder, work_queue)
    on_stop()  # must not block or raise

    assert "queue full" in capsys.readouterr().err


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


def test_recorder_stopped_on_shutdown_while_recording(mocker):
    """If the mic is active when shutdown fires, recorder.stop() must be called
    so the PortAudio stream is closed and the microphone is released."""
    mock_stream = mocker.MagicMock()
    mocker.patch("audio.sd.InputStream", return_value=mock_stream)

    from audio import AudioRecorder
    recorder = AudioRecorder()
    recorder.start()  # simulate: user is mid-recording when shutdown fires

    assert recorder._stream is not None

    recorder.stop()  # this is what main() calls in its shutdown sequence

    mock_stream.stop.assert_called_once()
    mock_stream.close.assert_called_once()
    assert recorder._stream is None
    assert recorder._frames == []


# --- mic error isolation ---

def test_portaudio_error_on_start_does_not_propagate(mocker, capsys):
    """PortAudio errors must not propagate through on_start — that would kill pynput's listener."""
    from audio import AudioRecorder
    broken_recorder = mocker.MagicMock(spec=AudioRecorder)
    broken_recorder.start.side_effect = Exception("PortAudioError -9986")

    on_start, _ = make_callbacks(broken_recorder, queue.Queue())
    on_start()  # must not raise

    assert "mic error" in capsys.readouterr().err


def test_portaudio_error_on_stop_does_not_propagate_and_skips_queue(mocker, capsys):
    """PortAudio errors in on_stop must be caught; no audio must be queued."""
    from audio import AudioRecorder
    broken_recorder = mocker.MagicMock(spec=AudioRecorder)
    broken_recorder.stop.side_effect = Exception("PortAudioError -9986")

    work_queue: queue.Queue = queue.Queue()
    _, on_stop = make_callbacks(broken_recorder, work_queue)
    on_stop()  # must not raise

    assert "mic error" in capsys.readouterr().err
    assert work_queue.empty()


def test_worker_debug_audio_prints_afplay_command(mocker, capsys, tmp_path):
    """afplay message appears only after the WAV file is actually written."""
    mocker.patch("dictate._DEBUG_DIR", tmp_path)

    def _transcribe(audio, debug_path=None):
        if debug_path is not None:
            debug_path.write_bytes(b"fake wav")
        return "hello"

    mock_server = mocker.MagicMock(spec=WhisperServer)
    mock_server.transcribe.side_effect = _transcribe
    mock_typer = mocker.MagicMock(spec=TextTyper)

    work_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    t = threading.Thread(
        target=make_worker(mock_server, mock_typer, work_queue, stop_event, debug_audio=True),
        daemon=True,
    )
    t.start()
    work_queue.put(np.ones(16000, dtype="float32"))
    work_queue.join()
    stop_event.set()
    t.join(timeout=2)

    captured = capsys.readouterr()
    assert "afplay" in captured.err
    assert "wd-debug-" in captured.err


def test_worker_debug_audio_prints_afplay_even_on_network_error(mocker, capsys, tmp_path):
    """afplay message must fire even when RequestException interrupts after the file was written."""
    mocker.patch("dictate._DEBUG_DIR", tmp_path)

    def _transcribe(audio, debug_path=None):
        if debug_path is not None:
            debug_path.write_bytes(b"fake wav")
        raise requests.ConnectionError("refused")

    mock_server = mocker.MagicMock(spec=WhisperServer)
    mock_server.transcribe.side_effect = _transcribe
    mock_server.is_alive.return_value = True
    mock_typer = mocker.MagicMock(spec=TextTyper)

    work_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    t = threading.Thread(
        target=make_worker(mock_server, mock_typer, work_queue, stop_event, debug_audio=True),
        daemon=True,
    )
    t.start()
    work_queue.put(np.ones(16000, dtype="float32"))
    work_queue.join()
    stop_event.set()
    t.join(timeout=2)

    assert "afplay" in capsys.readouterr().err


def test_worker_debug_audio_no_afplay_on_too_short_clip(mocker, capsys, tmp_path):
    """If audio is too short, transcribe() skips the write — no afplay message must appear."""
    mocker.patch("dictate._DEBUG_DIR", tmp_path)

    mock_server = mocker.MagicMock(spec=WhisperServer)
    mock_server.transcribe.return_value = ""  # short-audio early return — file never written
    mock_typer = mocker.MagicMock(spec=TextTyper)

    work_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    t = threading.Thread(
        target=make_worker(mock_server, mock_typer, work_queue, stop_event, debug_audio=True),
        daemon=True,
    )
    t.start()
    work_queue.put(np.ones(100, dtype="float32"))
    work_queue.join()
    stop_event.set()
    t.join(timeout=2)

    assert "afplay" not in capsys.readouterr().err


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


def test_worker_keeps_processing_when_every_restart_fails(mocker):
    """Worker must survive indefinitely when server is dead and every restart also fails."""
    mock_server = mocker.MagicMock(spec=WhisperServer)
    mock_server.transcribe.side_effect = requests.ConnectionError("refused")
    mock_server.is_alive.return_value = False
    mock_server.restart.side_effect = RuntimeError("restart failed")
    mock_typer = mocker.MagicMock(spec=TextTyper)

    audio = np.ones(16000, dtype="float32")
    _run_worker(mock_server, mock_typer, [audio, audio, audio])

    assert mock_server.restart.call_count == 3
    mock_typer.type_text.assert_not_called()
