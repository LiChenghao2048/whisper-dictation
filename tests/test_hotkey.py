import threading
from unittest.mock import MagicMock, patch
from pynput import keyboard

from hotkey import HotkeyListener


def _make_listener(mode="hold"):
    on_start = MagicMock()
    on_stop = MagicMock()
    listener = HotkeyListener(key="alt_r", mode=mode, on_start=on_start, on_stop=on_stop)
    return listener, on_start, on_stop


# --- hold mode ---

def test_hold_press_calls_on_start():
    listener, on_start, on_stop = _make_listener("hold")
    listener._on_press(keyboard.Key.alt_r)
    on_start.assert_called_once()
    on_stop.assert_not_called()


def test_hold_release_calls_on_stop():
    listener, on_start, on_stop = _make_listener("hold")
    listener._on_press(keyboard.Key.alt_r)
    listener._on_release(keyboard.Key.alt_r)
    on_stop.assert_called_once()


def test_hold_press_twice_not_double_started():
    listener, on_start, on_stop = _make_listener("hold")
    listener._on_press(keyboard.Key.alt_r)
    listener._on_press(keyboard.Key.alt_r)  # key repeat event
    on_start.assert_called_once()


def test_hold_release_without_press_ignored():
    listener, on_start, on_stop = _make_listener("hold")
    listener._on_release(keyboard.Key.alt_r)
    on_stop.assert_not_called()


# --- toggle mode ---

def test_toggle_first_press_starts():
    listener, on_start, on_stop = _make_listener("toggle")
    listener._on_press(keyboard.Key.alt_r)
    on_start.assert_called_once()
    on_stop.assert_not_called()


def test_toggle_second_press_stops():
    listener, on_start, on_stop = _make_listener("toggle")
    listener._on_press(keyboard.Key.alt_r)
    listener._on_press(keyboard.Key.alt_r)
    on_start.assert_called_once()
    on_stop.assert_called_once()


def test_toggle_release_does_nothing():
    listener, on_start, on_stop = _make_listener("toggle")
    listener._on_press(keyboard.Key.alt_r)
    listener._on_release(keyboard.Key.alt_r)
    on_stop.assert_not_called()


# --- unrelated key ignored ---

def test_other_key_ignored():
    listener, on_start, on_stop = _make_listener("hold")
    listener._on_press(keyboard.Key.space)
    listener._on_release(keyboard.Key.space)
    on_start.assert_not_called()
    on_stop.assert_not_called()


# --- stop clears state ---

def test_stop_resets_recording_flag(mocker):
    mock_kb_listener = mocker.MagicMock()
    mocker.patch("hotkey.keyboard.Listener", return_value=mock_kb_listener)

    listener, on_start, on_stop = _make_listener("hold")
    listener.start()
    listener._on_press(keyboard.Key.alt_r)
    assert listener._recording is True

    listener.stop()
    assert listener._recording is False
    assert listener._listener is None


# --- is_alive ---

def test_is_alive_false_before_start():
    listener, _, _ = _make_listener()
    assert listener.is_alive() is False


def test_is_alive_true_when_running(mocker):
    mock_kb_listener = mocker.MagicMock()
    mock_kb_listener.is_alive.return_value = True
    mocker.patch("hotkey.keyboard.Listener", return_value=mock_kb_listener)

    listener, _, _ = _make_listener()
    listener.start()
    assert listener.is_alive() is True


def test_is_alive_false_after_stop(mocker):
    mock_kb_listener = mocker.MagicMock()
    mocker.patch("hotkey.keyboard.Listener", return_value=mock_kb_listener)

    listener, _, _ = _make_listener()
    listener.start()
    listener.stop()
    assert listener.is_alive() is False


# --- thread safety ---

def test_recording_flag_protected_by_lock():
    """stop() and _on_press() share the same lock — stop() must wait for any in-progress press."""
    listener, on_start, on_stop = _make_listener("hold")
    results = []

    def slow_press():
        with listener._lock:
            listener._recording = True
            results.append("press_started")

    t = threading.Thread(target=slow_press)
    t.start()
    t.join()

    listener.stop()  # must acquire lock cleanly and reset _recording
    assert listener._recording is False
    assert results == ["press_started"]
