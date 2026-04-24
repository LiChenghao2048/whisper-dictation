import pytest
from unittest.mock import patch, MagicMock, call

from typer import TextTyper


def test_type_text_copies_and_pastes(mocker):
    mock_copy = mocker.patch("typer.pyperclip.copy")
    mock_run = mocker.patch("typer.subprocess.run")

    typer = TextTyper()
    typer.type_text("hello world")

    mock_copy.assert_called_once_with("hello world")
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "osascript" in cmd
    assert 'keystroke "v"' in cmd[-1]


def test_type_text_empty_string_does_nothing(mocker):
    mock_copy = mocker.patch("typer.pyperclip.copy")
    mock_run = mocker.patch("typer.subprocess.run")

    typer = TextTyper()
    typer.type_text("")

    mock_copy.assert_not_called()
    mock_run.assert_not_called()


def test_type_text_unicode(mocker):
    mock_copy = mocker.patch("typer.pyperclip.copy")
    mocker.patch("typer.subprocess.run")

    typer = TextTyper()
    typer.type_text("你好世界")

    mock_copy.assert_called_once_with("你好世界")
