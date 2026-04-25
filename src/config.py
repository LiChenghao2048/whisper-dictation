from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ServerConfig:
    host: str
    port: int


@dataclass
class Config:
    hotkey: str
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
        srv = data.get("server", {})
        return cls(
            hotkey=data["hotkey"],
            mode=data["mode"],
            model=data["model"],
            language=data["language"],
            task=data.get("task", "transcribe"),
            device=data.get("device", None),
            debug_audio=data.get("debug_audio", False),
            server=ServerConfig(
                host=srv.get("host", "localhost"),
                port=int(srv.get("port", 50060)),
            ),
        )
