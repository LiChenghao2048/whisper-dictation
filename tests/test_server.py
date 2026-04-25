import io
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
import requests

from server import WhisperServer, _to_wav_bytes, SAMPLE_RATE, MIN_AUDIO_SECONDS


BINARY = Path("/fake/argmax-cli")


def _make_server(**kwargs):
    defaults = dict(
        binary=BINARY,
        model="small",
        language="en",
        task="transcribe",
        host="localhost",
        port=50060,
    )
    defaults.update(kwargs)
    return WhisperServer(**defaults)


# --- endpoint selection ---

def test_transcribe_task_uses_transcriptions_endpoint():
    server = _make_server(task="transcribe")
    assert server._endpoint == "/v1/audio/transcriptions"


def test_translate_task_uses_translations_endpoint():
    server = _make_server(task="translate")
    assert server._endpoint == "/v1/audio/translations"


# --- start / stop ---

def test_start_polls_health_until_ok(mocker):
    mock_popen = mocker.patch("server.subprocess.Popen")
    mock_get = mocker.patch("server.requests.get")
    mock_get.side_effect = [
        requests.ConnectionError(),
        requests.ConnectionError(),
        MagicMock(status_code=200),
    ]
    mocker.patch("server.time.sleep")

    server = _make_server()
    server.start(timeout=10)

    assert mock_get.call_count == 3


def test_start_retries_on_health_check_timeout(mocker):
    mocker.patch("server.subprocess.Popen")
    mock_get = mocker.patch("server.requests.get")
    mock_get.side_effect = [
        requests.Timeout(),
        requests.Timeout(),
        MagicMock(status_code=200),
    ]
    mocker.patch("server.time.sleep")

    server = _make_server()
    server.start(timeout=10)

    assert mock_get.call_count == 3


def test_start_raises_timeout_when_server_never_healthy(mocker):
    mocker.patch("server.subprocess.Popen")
    mocker.patch("server.requests.get", side_effect=requests.ConnectionError())
    mocker.patch("server.time.sleep")
    mocker.patch("server.time.monotonic", side_effect=[0, 0, 200])  # immediate timeout

    server = _make_server()
    with pytest.raises(TimeoutError):
        server.start(timeout=1)


def test_stop_terminates_process(mocker):
    mock_process = mocker.MagicMock()
    server = _make_server()
    server._process = mock_process

    server.stop()

    mock_process.terminate.assert_called_once()
    mock_process.wait.assert_called_once()
    assert server._process is None


def test_stop_kills_if_terminate_hangs(mocker):
    import subprocess
    mock_process = mocker.MagicMock()
    mock_process.wait.side_effect = [subprocess.TimeoutExpired(cmd="x", timeout=5), None]
    server = _make_server()
    server._process = mock_process

    server.stop()

    mock_process.kill.assert_called_once()


def test_stop_is_idempotent():
    server = _make_server()
    server.stop()  # called with no process — must not raise
    server.stop()


# --- transcribe ---

def test_transcribe_returns_text(mocker):
    mock_post = mocker.patch("server.requests.post")
    mock_post.return_value.json.return_value = {"text": " hello world "}
    mock_post.return_value.raise_for_status = MagicMock()

    server = _make_server()
    audio = np.ones(SAMPLE_RATE * 2, dtype="float32") * 0.1
    result = server.transcribe(audio)

    assert result == "hello world"


def test_transcribe_skips_too_short_audio():
    server = _make_server()
    short_audio = np.zeros(int(SAMPLE_RATE * MIN_AUDIO_SECONDS) - 1, dtype="float32")
    result = server.transcribe(short_audio)
    assert result == ""


# --- _to_wav_bytes ---

def test_to_wav_bytes_produces_valid_wav():
    audio = np.sin(np.linspace(0, 2 * np.pi, SAMPLE_RATE)).astype("float32")
    wav = _to_wav_bytes(audio)

    with wave.open(io.BytesIO(wav)) as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == SAMPLE_RATE
        assert wf.getnframes() == SAMPLE_RATE


def test_to_wav_bytes_clips_values():
    audio = np.array([2.0, -2.0, 0.5], dtype="float32")
    wav = _to_wav_bytes(audio)

    with wave.open(io.BytesIO(wav)) as wf:
        raw = wf.readframes(3)
    samples = np.frombuffer(raw, dtype="<i2")
    assert samples[0] == 32767   # clipped to max
    assert samples[1] == -32767  # clipped to min
