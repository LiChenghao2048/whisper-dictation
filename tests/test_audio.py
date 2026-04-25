import numpy as np
import pytest

from audio import AudioRecorder, SAMPLE_RATE


def test_stop_before_start_returns_empty():
    recorder = AudioRecorder()
    audio = recorder.stop()
    assert isinstance(audio, np.ndarray)
    assert audio.shape == (0,)
    assert audio.dtype == np.float32


def test_frames_cleared_after_stop(mocker):
    mock_stream = mocker.MagicMock()
    mocker.patch("audio.sd.InputStream", return_value=mock_stream)

    recorder = AudioRecorder()
    recorder.start()

    chunk = np.ones((512, 1), dtype="float32") * 0.5
    recorder._callback(chunk, 512, None, None)
    recorder._callback(chunk, 512, None, None)

    assert len(recorder._frames) == 2

    audio = recorder.stop()

    # Frames must be cleared to avoid memory growth across sessions
    assert recorder._frames == []
    assert len(audio) == 1024
    np.testing.assert_allclose(audio, 0.5)


def test_stream_closed_on_stop(mocker):
    mock_stream = mocker.MagicMock()
    mocker.patch("audio.sd.InputStream", return_value=mock_stream)

    recorder = AudioRecorder()
    recorder.start()
    recorder.stop()

    mock_stream.stop.assert_called_once()
    mock_stream.close.assert_called_once()
    assert recorder._stream is None


def test_multiple_start_stop_cycles_no_accumulation(mocker):
    mock_stream = mocker.MagicMock()
    mocker.patch("audio.sd.InputStream", return_value=mock_stream)

    recorder = AudioRecorder()
    for _ in range(5):
        recorder.start()
        chunk = np.ones((100, 1), dtype="float32")
        recorder._callback(chunk, 100, None, None)
        recorder.stop()

    assert recorder._frames == []
    assert recorder._stream is None


def test_device_passed_to_inputstream(mocker):
    mock_stream = mocker.MagicMock()
    mock_sd = mocker.patch("audio.sd.InputStream", return_value=mock_stream)

    AudioRecorder(device=3).start()

    _, kwargs = mock_sd.call_args
    assert kwargs["device"] == 3


def test_check_warns_on_silent_mic(mocker, capsys):
    silent = np.zeros((int(0.1 * SAMPLE_RATE), 1), dtype="float32")
    mocker.patch("audio.sd.rec", return_value=silent)
    mocker.patch("audio.sd.wait")
    mocker.patch("audio.sd.query_devices", return_value={"name": "FakeMic"})
    mocker.patch("audio.sd.default", **{"device": [0, 1]})

    AudioRecorder().check()

    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "silent" in captured.out
    assert "Available input devices" not in captured.out  # device list suppressed without verbose


def test_check_verbose_prints_device_list_on_silent_mic(mocker, capsys):
    silent = np.zeros((int(0.1 * SAMPLE_RATE), 1), dtype="float32")
    mocker.patch("audio.sd.rec", return_value=silent)
    mocker.patch("audio.sd.wait")
    mocker.patch("audio.sd.default", **{"device": [0, 1]})

    def _query(device=None):
        if device is None:
            return [{"name": "FakeMic", "max_input_channels": 1}]
        return {"name": "FakeMic"}

    mocker.patch("audio.sd.query_devices", side_effect=_query)

    AudioRecorder().check(verbose=True)

    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "Available input devices" in captured.out
    assert "FakeMic" in captured.out


def test_check_device_zero_named_correctly_in_warning(mocker, capsys):
    silent = np.zeros((int(0.1 * SAMPLE_RATE), 1), dtype="float32")
    mocker.patch("audio.sd.rec", return_value=silent)
    mocker.patch("audio.sd.wait")
    mocker.patch("audio.sd.query_devices", return_value={"name": "BuiltInMic"})

    AudioRecorder(device=0).check()

    captured = capsys.readouterr()
    assert "BuiltInMic" in captured.out  # device=0 must not fall through to system default


def test_check_passes_silently_on_active_mic(mocker, capsys):
    loud = np.ones((int(0.1 * SAMPLE_RATE), 1), dtype="float32") * 0.05
    mocker.patch("audio.sd.rec", return_value=loud)
    mocker.patch("audio.sd.wait")

    AudioRecorder().check()

    captured = capsys.readouterr()
    assert "WARNING" not in captured.out
