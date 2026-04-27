from __future__ import annotations

import requests

_SYSTEM = (
    "You are a dictation editor. "
    "Remove filler words (uh, um, ah, eh, hmm). "
    "If the speaker changed their mind mid-sentence, keep only the final intended meaning. "
    "Output only the cleaned text, nothing else. No explanations, no quotes."
)

_USER_TEMPLATE = "Clean up this dictation:\n\n{text}"


class TextCleaner:
    def __init__(self, model: str, host: str, port: int) -> None:
        self._model = model
        self._base_url = f"http://{host}:{port}"

    def clean(self, text: str) -> str:
        r = requests.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": _USER_TEMPLATE.format(text=text)},
                ],
            },
            timeout=15,
        )
        r.raise_for_status()
        body = r.json()
        try:
            return body["message"]["content"].strip()
        except (KeyError, TypeError) as exc:
            raise ValueError(f"unexpected Ollama response shape: {body}") from exc

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self._base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False
