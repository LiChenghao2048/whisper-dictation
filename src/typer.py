import subprocess

import pyperclip


class TextTyper:
    def type_text(self, text: str) -> None:
        if not text:
            return
        pyperclip.copy(text)
        subprocess.run(
            [
                "osascript", "-e",
                'tell application "System Events" to keystroke "v" using command down',
            ],
            check=True,
            capture_output=True,
        )
