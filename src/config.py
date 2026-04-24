from dataclasses import dataclass
from pathlib import Path

import yaml


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
        srv = data.get("server", {})
        return cls(
            hotkey=data["hotkey"],
            mode=data["mode"],
            model=data["model"],
            language=data["language"],
            task=data.get("task", "transcribe"),
            server=ServerConfig(
                host=srv.get("host", "localhost"),
                port=int(srv.get("port", 50060)),
            ),
        )
