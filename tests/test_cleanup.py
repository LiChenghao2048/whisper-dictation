import pytest
import requests

from cleanup import TextCleaner


def _make_cleaner(**kwargs):
    defaults = dict(model="llama3.2", host="localhost", port=11434)
    defaults.update(kwargs)
    return TextCleaner(**defaults)


# --- clean ---

def test_clean_returns_model_response(mocker):
    mock_post = mocker.patch("cleanup.requests.post")
    mock_post.return_value.json.return_value = {
        "message": {"role": "assistant", "content": "  cleaned text  "}
    }
    mock_post.return_value.raise_for_status = mocker.MagicMock()

    result = _make_cleaner().clean("uh hello um world")

    assert result == "cleaned text"
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args[1]
    payload = call_kwargs["json"]
    assert payload["model"] == "llama3.2"
    assert payload["stream"] is False
    assert any(m["role"] == "user" for m in payload["messages"])


def test_clean_sends_original_text_in_user_message(mocker):
    mock_post = mocker.patch("cleanup.requests.post")
    mock_post.return_value.json.return_value = {"message": {"role": "assistant", "content": "ok"}}
    mock_post.return_value.raise_for_status = mocker.MagicMock()

    _make_cleaner().clean("uh I meant to say hello")

    payload = mock_post.call_args[1]["json"]
    user_msg = next(m for m in payload["messages"] if m["role"] == "user")
    assert "uh I meant to say hello" in user_msg["content"]


def test_clean_raises_on_http_error(mocker):
    mock_post = mocker.patch("cleanup.requests.post")
    mock_post.return_value.raise_for_status.side_effect = requests.HTTPError("500")

    with pytest.raises(requests.HTTPError):
        _make_cleaner().clean("hello")


def test_clean_raises_on_connection_error(mocker):
    mocker.patch("cleanup.requests.post", side_effect=requests.ConnectionError("refused"))

    with pytest.raises(requests.ConnectionError):
        _make_cleaner().clean("hello")


# --- is_available ---

def test_is_available_true_when_ollama_responds(mocker):
    mock_get = mocker.patch("cleanup.requests.get")
    mock_get.return_value.status_code = 200

    assert _make_cleaner().is_available() is True
    mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=2)


def test_is_available_false_on_non_200(mocker):
    mock_get = mocker.patch("cleanup.requests.get")
    mock_get.return_value.status_code = 503

    assert _make_cleaner().is_available() is False


def test_is_available_false_on_connection_error(mocker):
    mocker.patch("cleanup.requests.get", side_effect=requests.ConnectionError())

    assert _make_cleaner().is_available() is False


def test_is_available_false_on_timeout(mocker):
    mocker.patch("cleanup.requests.get", side_effect=requests.Timeout())

    assert _make_cleaner().is_available() is False


# --- URL construction ---

def test_clean_posts_to_configured_host_and_port(mocker):
    mock_post = mocker.patch("cleanup.requests.post")
    mock_post.return_value.json.return_value = {"message": {"role": "assistant", "content": "ok"}}
    mock_post.return_value.raise_for_status = mocker.MagicMock()

    _make_cleaner(host="192.168.1.5", port=9999).clean("hello")

    url = mock_post.call_args[0][0]
    assert "192.168.1.5:9999" in url
