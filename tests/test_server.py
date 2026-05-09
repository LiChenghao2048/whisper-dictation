import io
import itertools
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

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
        temperature=0.0,
        prompt=None,
    )
    defaults.update(kwargs)
    return WhisperServer(**defaults)


# --- endpoint selection ---

def test_transcribe_task_uses_transcriptions_endpoint():
    assert _make_server(task="transcribe")._endpoint == "/v1/audio/transcriptions"


def test_translate_task_uses_translations_endpoint():
    assert _make_server(task="translate")._endpoint == "/v1/audio/translations"


def test_language_included_in_cmd_when_set():
    server = _make_server(language="zh")
    assert "--language" in server._cmd
    idx = server._cmd.index("--language")
    assert server._cmd[idx + 1] == "zh"


def test_language_omitted_from_cmd_when_none():
    server = _make_server(language=None)
    assert "--language" not in server._cmd


# --- start / stop ---

def _running_proc(mocker):
    """Return a mock Popen whose poll() signals the process is still alive."""
    mock_proc = mocker.MagicMock()
    mock_proc.poll.return_value = None
    return mock_proc


def test_start_polls_health_until_ok(mocker):
    mock_proc = _running_proc(mocker)
    mocker.patch("server.subprocess.Popen", return_value=mock_proc)
    mock_get = mocker.patch("server.requests.get")
    mock_get.side_effect = [
        requests.ConnectionError(),
        requests.ConnectionError(),
        MagicMock(status_code=200),
    ]
    mocker.patch("server.time.sleep")

    _make_server().start(timeout=10)

    assert mock_get.call_count == 3


def test_start_handles_read_timeout_not_just_connection_error(mocker):
    """ReadTimeout must not leak the subprocess — it must be caught like ConnectionError."""
    mock_proc = _running_proc(mocker)
    mocker.patch("server.subprocess.Popen", return_value=mock_proc)
    mock_get = mocker.patch("server.requests.get")
    mock_get.side_effect = [
        requests.ReadTimeout(),
        MagicMock(status_code=200),
    ]
    mocker.patch("server.time.sleep")

    _make_server().start(timeout=10)  # must not raise

    assert mock_get.call_count == 2


def test_start_raises_timeout_when_server_never_healthy(mocker):
    mock_proc = _running_proc(mocker)
    mocker.patch("server.subprocess.Popen", return_value=mock_proc)
    mocker.patch("server.requests.get", side_effect=requests.ConnectionError())
    mocker.patch("server.time.sleep")
    # First call returns 0 (deadline calc), subsequent calls return a value past deadline
    mocker.patch(
        "server.time.monotonic",
        side_effect=itertools.chain([0.0], itertools.repeat(200.0)),
    )

    with pytest.raises(TimeoutError):
        _make_server().start(timeout=1)


def test_start_raises_immediately_when_process_exits(mocker):
    """If WhisperKit exits during startup (bad model, missing weights), raise at once."""
    mock_proc = mocker.MagicMock()
    mock_proc.poll.return_value = 1
    mock_proc.returncode = 1
    mocker.patch("server.subprocess.Popen", return_value=mock_proc)
    mocker.patch("server.time.sleep")
    mocker.patch(
        "server.time.monotonic",
        side_effect=itertools.chain([0.0], itertools.repeat(0.5)),
    )

    with pytest.raises(RuntimeError, match="exited during startup"):
        _make_server().start(timeout=100)

    mock_proc.wait.assert_called_once()  # zombie must be reaped


def test_start_returns_cleanly_when_stop_called_concurrently(mocker):
    """If stop() clears _process while start() is health-polling, start() must not crash."""
    mock_proc = _running_proc(mocker)
    server = _make_server()

    original_get = requests.get

    call_count = [0]

    def _get(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 2:
            server._process = None  # simulate stop() clearing _process mid-poll
        raise requests.ConnectionError()

    mocker.patch("server.subprocess.Popen", return_value=mock_proc)
    mocker.patch("server.requests.get", side_effect=_get)
    mocker.patch("server.time.sleep")
    mocker.patch(
        "server.time.monotonic",
        side_effect=itertools.chain([0.0], itertools.repeat(0.5)),
    )

    server.start(timeout=100)  # must return cleanly, not raise AttributeError


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
    _make_server().stop()
    _make_server().stop()


# --- is_alive / restart ---

def test_is_alive_false_when_no_process():
    assert _make_server().is_alive() is False


def test_is_alive_true_when_process_running(mocker):
    mock_process = mocker.MagicMock()
    mock_process.poll.return_value = None  # still running
    server = _make_server()
    server._process = mock_process
    assert server.is_alive() is True


def test_is_alive_false_when_process_exited(mocker):
    mock_process = mocker.MagicMock()
    mock_process.poll.return_value = 1  # exited
    server = _make_server()
    server._process = mock_process
    assert server.is_alive() is False


def test_restart_stops_then_starts(mocker):
    mock_proc = _running_proc(mocker)
    mocker.patch("server.subprocess.Popen", return_value=mock_proc)
    mock_get = mocker.patch("server.requests.get", return_value=MagicMock(status_code=200))
    mocker.patch("server.time.sleep")

    server = _make_server()
    mock_process = mocker.MagicMock()
    server._process = mock_process

    server.restart()

    mock_process.terminate.assert_called_once()  # stop() was called
    assert mock_get.called                        # start() was called


# --- transcribe ---

def test_transcribe_returns_text(mocker):
    mock_post = mocker.patch("server.requests.post")
    mock_post.return_value.json.return_value = {"text": " hello world "}
    mock_post.return_value.raise_for_status = MagicMock()

    audio = np.ones(SAMPLE_RATE * 2, dtype="float32") * 0.1
    assert _make_server().transcribe(audio) == "hello world"

    _, kwargs = mock_post.call_args
    assert kwargs["data"]["model"] == "whisper-1"


def test_transcribe_sends_temperature(mocker):
    mock_post = mocker.patch("server.requests.post")
    mock_post.return_value.json.return_value = {"text": "hello"}
    mock_post.return_value.raise_for_status = MagicMock()

    audio = np.ones(SAMPLE_RATE * 2, dtype="float32") * 0.1
    _make_server(temperature=0.4).transcribe(audio)

    _, kwargs = mock_post.call_args
    assert kwargs["data"]["temperature"] == "0.4"


def test_transcribe_sends_prompt_when_set(mocker):
    mock_post = mocker.patch("server.requests.post")
    mock_post.return_value.json.return_value = {"text": "hello"}
    mock_post.return_value.raise_for_status = MagicMock()

    audio = np.ones(SAMPLE_RATE * 2, dtype="float32") * 0.1
    _make_server(prompt="WhisperKit CoreML").transcribe(audio)

    _, kwargs = mock_post.call_args
    assert kwargs["data"]["prompt"] == "WhisperKit CoreML"


def test_transcribe_omits_prompt_when_none(mocker):
    mock_post = mocker.patch("server.requests.post")
    mock_post.return_value.json.return_value = {"text": "hello"}
    mock_post.return_value.raise_for_status = MagicMock()

    audio = np.ones(SAMPLE_RATE * 2, dtype="float32") * 0.1
    _make_server(prompt=None).transcribe(audio)

    _, kwargs = mock_post.call_args
    assert "prompt" not in kwargs["data"]


def test_transcribe_skips_too_short_audio():
    short = np.zeros(int(SAMPLE_RATE * MIN_AUDIO_SECONDS) - 1, dtype="float32")
    assert _make_server().transcribe(short) == ""


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
    assert samples[0] == 32767   # +overflow clipped to int16 max
    assert samples[1] == -32768  # -overflow clipped to int16 min


# --- simplified Chinese conversion ---

def test_transcribe_converts_traditional_to_simplified(mocker):
    mock_post = mocker.patch("server.requests.post")
    mock_post.return_value.json.return_value = {"text": "漢字"}  # Traditional
    mock_post.return_value.raise_for_status = MagicMock()

    audio = np.ones(SAMPLE_RATE * 2, dtype="float32") * 0.1
    result = _make_server(simplified=True).transcribe(audio)

    assert result == "汉字"  # Simplified


def test_transcribe_no_conversion_when_simplified_false(mocker):
    mock_post = mocker.patch("server.requests.post")
    mock_post.return_value.json.return_value = {"text": "漢字"}
    mock_post.return_value.raise_for_status = MagicMock()

    audio = np.ones(SAMPLE_RATE * 2, dtype="float32") * 0.1
    result = _make_server(simplified=False).transcribe(audio)

    assert result == "漢字"  # unchanged


def test_transcribe_simplified_skipped_for_empty_text(mocker):
    mock_post = mocker.patch("server.requests.post")
    mock_post.return_value.json.return_value = {"text": ""}
    mock_post.return_value.raise_for_status = MagicMock()

    audio = np.ones(SAMPLE_RATE * 2, dtype="float32") * 0.1
    result = _make_server(simplified=True).transcribe(audio)

    assert result == ""
