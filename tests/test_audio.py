import numpy as np
import pytest

from audio import AudioRecorder, SAMPLE_RATE


def _make_recorder(mock_sd):
    """Return a recorder whose sounddevice is mocked out."""
    recorder = AudioRecorder()
    return recorder


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

    # Simulate two callback chunks
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

    # After 5 cycles, no frames should remain
    assert recorder._frames == []
    assert recorder._stream is None
