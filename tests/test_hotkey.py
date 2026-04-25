import pytest
import threading
from unittest.mock import MagicMock
from pynput import keyboard

from hotkey import HotkeyListener


def _make_listener(mode="hold", keys=None):
    if keys is None:
        keys = ["alt_r"]
    on_start = MagicMock()
    on_stop = MagicMock()
    listener = HotkeyListener(keys=keys, mode=mode, on_start=on_start, on_stop=on_stop)
    return listener, on_start, on_stop


# --- hold mode (single key) ---

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


# --- toggle mode (single key) ---

def test_toggle_first_press_starts():
    listener, on_start, on_stop = _make_listener("toggle")
    listener._on_press(keyboard.Key.alt_r)
    on_start.assert_called_once()
    on_stop.assert_not_called()


def test_toggle_second_press_stops():
    listener, on_start, on_stop = _make_listener("toggle")
    listener._on_press(keyboard.Key.alt_r)    # start
    listener._on_release(keyboard.Key.alt_r)  # release — required before re-trigger
    listener._on_press(keyboard.Key.alt_r)    # stop
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


# --- chord (multi-key) ---

def test_chord_hold_starts_only_when_all_keys_pressed():
    listener, on_start, on_stop = _make_listener("hold", keys=["cmd_r", "alt_r"])
    listener._on_press(keyboard.Key.cmd_r)
    on_start.assert_not_called()
    listener._on_press(keyboard.Key.alt_r)
    on_start.assert_called_once()


def test_chord_hold_stops_on_first_key_release():
    listener, on_start, on_stop = _make_listener("hold", keys=["cmd_r", "alt_r"])
    listener._on_press(keyboard.Key.cmd_r)
    listener._on_press(keyboard.Key.alt_r)
    listener._on_release(keyboard.Key.alt_r)
    on_stop.assert_called_once()


def test_chord_partial_press_does_not_start():
    listener, on_start, on_stop = _make_listener("hold", keys=["cmd_r", "alt_r"])
    listener._on_press(keyboard.Key.cmd_r)
    on_start.assert_not_called()
    listener._on_release(keyboard.Key.cmd_r)
    on_start.assert_not_called()


def test_chord_toggle_starts_on_full_chord():
    listener, on_start, on_stop = _make_listener("toggle", keys=["cmd_r", "alt_r"])
    listener._on_press(keyboard.Key.cmd_r)
    on_start.assert_not_called()
    listener._on_press(keyboard.Key.alt_r)
    on_start.assert_called_once()


def test_chord_toggle_stops_on_second_full_chord():
    listener, on_start, on_stop = _make_listener("toggle", keys=["cmd_r", "alt_r"])
    listener._on_press(keyboard.Key.cmd_r)
    listener._on_press(keyboard.Key.alt_r)  # start
    listener._on_release(keyboard.Key.cmd_r)
    listener._on_release(keyboard.Key.alt_r)
    listener._on_press(keyboard.Key.cmd_r)
    listener._on_press(keyboard.Key.alt_r)  # stop
    on_start.assert_called_once()
    on_stop.assert_called_once()


# --- stop clears state ---

def test_stop_resets_recording_flag(mocker):
    mock_kb_listener = mocker.MagicMock()
    mocker.patch("hotkey.keyboard.Listener", return_value=mock_kb_listener)

    listener, on_start, on_stop = _make_listener("hold")
    listener.start()
    listener._on_press(keyboard.Key.alt_r)
    assert listener._recording is True
    assert listener._chord_active is True

    listener.stop()
    assert listener._recording is False
    assert listener._chord_active is False
    assert listener._listener is None


def test_stop_clears_held_keys(mocker):
    mock_kb_listener = mocker.MagicMock()
    mocker.patch("hotkey.keyboard.Listener", return_value=mock_kb_listener)

    listener, _, _ = _make_listener("hold", keys=["cmd_r", "alt_r"])
    listener.start()
    listener._on_press(keyboard.Key.cmd_r)  # partially held
    assert keyboard.Key.cmd_r in listener._held

    listener.stop()
    assert listener._held == set()


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


# --- key-repeat suppression ---

def test_hold_key_repeat_does_not_double_start():
    listener, on_start, on_stop = _make_listener("hold")
    listener._on_press(keyboard.Key.alt_r)
    listener._on_press(keyboard.Key.alt_r)  # OS key-repeat while held
    on_start.assert_called_once()


def test_chord_toggle_key_repeat_does_not_re_toggle():
    """OS key-repeat must not re-fire the toggle while the chord is still physically held."""
    listener, on_start, on_stop = _make_listener("toggle", keys=["cmd_r", "alt_r"])
    listener._on_press(keyboard.Key.cmd_r)
    listener._on_press(keyboard.Key.alt_r)   # chord complete → start
    listener._on_press(keyboard.Key.alt_r)   # key-repeat — must be suppressed
    listener._on_press(keyboard.Key.cmd_r)   # key-repeat — must be suppressed
    on_start.assert_called_once()
    on_stop.assert_not_called()


def test_chord_toggle_fires_again_after_full_release():
    """Chord must be re-triggerable after all keys are released."""
    listener, on_start, on_stop = _make_listener("toggle", keys=["cmd_r", "alt_r"])
    listener._on_press(keyboard.Key.cmd_r)
    listener._on_press(keyboard.Key.alt_r)   # start
    listener._on_release(keyboard.Key.alt_r)
    listener._on_release(keyboard.Key.cmd_r)
    listener._on_press(keyboard.Key.cmd_r)
    listener._on_press(keyboard.Key.alt_r)   # stop
    on_start.assert_called_once()
    on_stop.assert_called_once()


# --- validation ---

def test_invalid_key_name_raises_value_error():
    with pytest.raises(ValueError, match="invalid_key_xyz"):
        HotkeyListener(keys=["invalid_key_xyz"], mode="hold", on_start=MagicMock(), on_stop=MagicMock())


def test_empty_key_list_raises_value_error():
    with pytest.raises(ValueError, match="at least one key"):
        HotkeyListener(keys=[], mode="hold", on_start=MagicMock(), on_stop=MagicMock())


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
