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
        temperature: 0.3
        prompt: "hello world"
        device: 2
        debug_audio: true
        server:
          host: localhost
          port: 50060
        cleanup:
          enabled: true
          model: llama3.2
          host: localhost
          port: 11434
    """))
    cfg = Config.load(cfg_file)
    assert cfg.hotkey == ["alt_r"]
    assert cfg.mode == "hold"
    assert cfg.model == "small"
    assert cfg.language == "en"
    assert cfg.task == "transcribe"
    assert cfg.temperature == 0.3
    assert cfg.prompt == "hello world"
    assert cfg.device == 2
    assert cfg.debug_audio is True
    assert cfg.server.host == "localhost"
    assert cfg.server.port == 50060
    assert cfg.cleanup.enabled is True
    assert cfg.cleanup.model == "llama3.2"
    assert cfg.cleanup.host == "localhost"
    assert cfg.cleanup.port == 11434


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
    assert cfg.task == "transcribe"    # default when omitted
    assert cfg.temperature == 0.0      # default when omitted
    assert cfg.prompt is None          # default when omitted
    assert cfg.device is None          # default when omitted
    assert cfg.debug_audio is False    # default when omitted
    assert cfg.mode == "toggle"
    assert cfg.server.port == 9999
    assert cfg.cleanup.enabled is False
    assert cfg.cleanup.model == "llama3.2"
    assert cfg.cleanup.port == 11434


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        Config.load(Path("/nonexistent/config.yaml"))


def test_invalid_mode_raises(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(textwrap.dedent("""\
        hotkey: alt_r
        mode: push
        model: small
        language: en
    """))
    with pytest.raises(ValueError, match="mode"):
        Config.load(cfg_file)


def test_invalid_task_raises(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(textwrap.dedent("""\
        hotkey: alt_r
        mode: hold
        model: small
        language: en
        task: summarize
    """))
    with pytest.raises(ValueError, match="task"):
        Config.load(cfg_file)


def test_invalid_temperature_raises(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(textwrap.dedent("""\
        hotkey: alt_r
        mode: hold
        model: small
        language: en
        temperature: 1.5
    """))
    with pytest.raises(ValueError, match="temperature"):
        Config.load(cfg_file)


def test_temperature_below_zero_raises(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(textwrap.dedent("""\
        hotkey: alt_r
        mode: hold
        model: small
        language: en
        temperature: -0.1
    """))
    with pytest.raises(ValueError, match="temperature"):
        Config.load(cfg_file)


def test_empty_prompt_treated_as_none(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(textwrap.dedent("""\
        hotkey: alt_r
        mode: hold
        model: small
        language: en
        prompt:
    """))
    cfg = Config.load(cfg_file)
    assert cfg.prompt is None


def test_cleanup_custom_values(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(textwrap.dedent("""\
        hotkey: alt_r
        mode: hold
        model: small
        language: en
        cleanup:
          enabled: true
          model: mistral
          host: 10.0.0.1
          port: 9999
    """))
    cfg = Config.load(cfg_file)
    assert cfg.cleanup.enabled is True
    assert cfg.cleanup.model == "mistral"
    assert cfg.cleanup.host == "10.0.0.1"
    assert cfg.cleanup.port == 9999


def test_chord_hotkey_loaded_as_list(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(textwrap.dedent("""\
        hotkey: [cmd_r, alt_r]
        mode: hold
        model: small
        language: en
    """))
    cfg = Config.load(cfg_file)
    assert cfg.hotkey == ["cmd_r", "alt_r"]


def test_single_string_hotkey_normalized_to_list(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(textwrap.dedent("""\
        hotkey: alt_r
        mode: hold
        model: small
        language: en
    """))
    cfg = Config.load(cfg_file)
    assert cfg.hotkey == ["alt_r"]
