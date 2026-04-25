from __future__ import annotations

import threading
from typing import Callable

from pynput import keyboard


class HotkeyListener:
    def __init__(
        self,
        keys: list[str],
        mode: str,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
    ) -> None:
        chord: set[keyboard.Key] = set()
        for k in keys:
            try:
                chord.add(getattr(keyboard.Key, k))
            except AttributeError:
                valid = sorted(key.name for key in keyboard.Key)
                raise ValueError(f"Unknown hotkey {k!r}. Valid key names: {valid}")
        self._chord = chord
        self._mode = mode
        self._on_start = on_start
        self._on_stop = on_stop
        self._recording = False
        self._held: set[keyboard.Key] = set()
        self._lock = threading.Lock()
        self._listener: keyboard.Listener | None = None

    def start(self) -> None:
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None
        with self._lock:
            self._recording = False
            self._held.clear()

    def is_alive(self) -> bool:
        return self._listener is not None and self._listener.is_alive()

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key not in self._chord:
            return
        with self._lock:
            self._held.add(key)
            if self._held != self._chord:
                return
            if self._mode == "hold" and not self._recording:
                self._recording = True
                self._on_start()
            elif self._mode == "toggle":
                if not self._recording:
                    self._recording = True
                    self._on_start()
                else:
                    self._recording = False
                    self._on_stop()

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key not in self._chord:
            return
        with self._lock:
            self._held.discard(key)
            if self._mode == "hold" and self._recording:
                self._recording = False
                self._on_stop()
