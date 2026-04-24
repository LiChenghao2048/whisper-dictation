import textwrap
from pathlib import Path

import pytest

from config import Config


def test_load_full(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(textwrap.dedent("""\
        hotkey: alt_r
        mode: hold
        model: small
        language: en
        task: transcribe
        server:
          host: localhost
          port: 50060
    """))
    cfg = Config.load(cfg_file)
    assert cfg.hotkey == "alt_r"
    assert cfg.mode == "hold"
    assert cfg.model == "small"
    assert cfg.language == "en"
    assert cfg.task == "transcribe"
    assert cfg.server.host == "localhost"
    assert cfg.server.port == 50060


def test_load_defaults_task_and_server(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(textwrap.dedent("""\
        hotkey: alt_r
        mode: toggle
        model: base
        language: zh
        server:
          host: 127.0.0.1
          port: 9999
    """))
    cfg = Config.load(cfg_file)
    assert cfg.task == "transcribe"  # default when omitted
    assert cfg.mode == "toggle"
    assert cfg.server.port == 9999


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        Config.load(Path("/nonexistent/config.yaml"))
