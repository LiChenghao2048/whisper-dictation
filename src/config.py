from dataclasses import dataclass
from pathlib import Path

import yaml

_VALID_MODES = {"hold", "toggle"}
_VALID_TASKS = {"transcribe", "translate"}


@dataclass
class ServerConfig:
    host: str
    port: int


@dataclass
class Config:
    hotkey: str
    mode: str    # "hold" or "toggle"
    model: str
    language: str
    task: str    # "transcribe" or "translate"
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
        srv = data.get("server", {})
        return cls(
            hotkey=data["hotkey"],
            mode=mode,
            model=data["model"],
            language=data["language"],
            task=task,
            server=ServerConfig(
                host=srv.get("host", "localhost"),
                port=int(srv.get("port", 50060)),
            ),
        )
