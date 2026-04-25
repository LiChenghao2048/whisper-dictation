from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

_VALID_MODES = {"hold", "toggle"}
_VALID_TASKS = {"transcribe", "translate"}


@dataclass
class ServerConfig:
    host: str
    port: int


@dataclass
class Config:
    hotkey: list[str]
    mode: str           # "hold" or "toggle"
    model: str
    language: str
    task: str           # "transcribe" or "translate"
    device: Optional[int]   # sounddevice input device index; None = system default
    debug_audio: bool   # save each recording to /tmp for inspection
    server: ServerConfig

    @classmethod
    def load(cls, path: Path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        mode = data["mode"]
        task = data.get("task", "transcribe")
        if mode not in _VALID_MODES:
            raise ValueError(f"config: mode must be one of {sorted(_VALID_MODES)}, got {mode!r}")
        if task not in _VALID_TASKS:
            raise ValueError(f"config: task must be one of {sorted(_VALID_TASKS)}, got {task!r}")
        raw_hotkey = data["hotkey"]
        hotkey = raw_hotkey if isinstance(raw_hotkey, list) else [raw_hotkey]
        srv = data.get("server", {})
        return cls(
            hotkey=hotkey,
            mode=mode,
            model=data["model"],
            language=data["language"],
            task=task,
            device=data.get("device", None),
            debug_audio=data.get("debug_audio", False),
            server=ServerConfig(
                host=srv.get("host", "localhost"),
                port=int(srv.get("port", 50060)),
            ),
        )
