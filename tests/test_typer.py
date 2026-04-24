import pytest
from unittest.mock import patch, MagicMock, call

from typer import TextTyper


def test_type_text_copies_and_pastes(mocker):
    mocker.patch("typer.pyperclip.paste", return_value="old clipboard")
    mock_copy = mocker.patch("typer.pyperclip.copy")
    mock_run = mocker.patch("typer.subprocess.run")
    mocker.patch("typer.time.sleep")

    TextTyper().type_text("hello world")

    mock_copy.assert_any_call("hello world")
    mock_run.assert_called_once()
    assert "osascript" in mock_run.call_args[0][0]
    assert 'keystroke "v"' in mock_run.call_args[0][0][-1]


def test_type_text_restores_clipboard(mocker):
    mocker.patch("typer.pyperclip.paste", return_value="old clipboard")
    mock_copy = mocker.patch("typer.pyperclip.copy")
    mocker.patch("typer.subprocess.run")
    mocker.patch("typer.time.sleep")

    TextTyper().type_text("hello")

    calls = [c.args[0] for c in mock_copy.call_args_list]
    assert calls[0] == "hello"           # first: set transcription text
    assert calls[-1] == "old clipboard"  # last: restore original


def test_type_text_empty_string_does_nothing(mocker):
    mock_copy = mocker.patch("typer.pyperclip.copy")
    mock_run = mocker.patch("typer.subprocess.run")

    TextTyper().type_text("")

    mock_copy.assert_not_called()
    mock_run.assert_not_called()


def test_type_text_unicode(mocker):
    mocker.patch("typer.pyperclip.paste", return_value="")
    mock_copy = mocker.patch("typer.pyperclip.copy")
    mocker.patch("typer.subprocess.run")
    mocker.patch("typer.time.sleep")

    TextTyper().type_text("你好世界")

    mock_copy.assert_any_call("你好世界")
