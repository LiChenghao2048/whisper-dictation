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
class CleanupConfig:
    enabled: bool
    model: str
    host: str
    port: int


@dataclass
class Config:
    hotkey: list[str]
    mode: str           # "hold" or "toggle"
    model: str
    language: Optional[str]  # None = auto-detect; "zh"/"en"/etc to force a language
    simplified: bool         # convert Traditional Chinese to Simplified after transcription
    task: str           # "transcribe" or "translate"
    temperature: float  # whisper sampling temperature 0.0–1.0; 0.0 = deterministic
    prompt: Optional[str]   # seed text to guide transcription accuracy
    device: Optional[int]   # sounddevice input device index; None = system default
    debug_audio: bool   # save each recording to /tmp for inspection
    server: ServerConfig
    cleanup: CleanupConfig

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
        temperature = float(data.get("temperature", 0.0))
        if not 0.0 <= temperature <= 1.0:
            raise ValueError(f"config: temperature must be between 0.0 and 1.0, got {temperature}")
        raw_hotkey = data["hotkey"]
        hotkey = raw_hotkey if isinstance(raw_hotkey, list) else [raw_hotkey]
        srv = data.get("server", {})
        cln = data.get("cleanup", {})
        return cls(
            hotkey=hotkey,
            mode=mode,
            model=data["model"],
            language=data.get("language") or None,
            simplified=bool(data.get("simplified", False)),
            task=task,
            temperature=temperature,
            prompt=data.get("prompt") or None,
            device=data.get("device", None),
            debug_audio=data.get("debug_audio", False),
            server=ServerConfig(
                host=srv.get("host", "localhost"),
                port=int(srv.get("port", 50060)),
            ),
            cleanup=CleanupConfig(
                enabled=bool(cln.get("enabled", False)),
                model=cln.get("model", "llama3.2"),
                host=cln.get("host", "localhost"),
                port=int(cln.get("port", 11434)),
            ),
        )
