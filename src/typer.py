import subprocess
import time

import pyperclip


class TextTyper:
    def type_text(self, text: str) -> None:
        if not text:
            return
        saved = pyperclip.paste()
        pyperclip.copy(text)
        subprocess.run(
            [
                "osascript", "-e",
                'tell application "System Events" to keystroke "v" using command down',
            ],
            check=True,
            capture_output=True,
        )
        # Brief pause so the paste event completes before we restore the clipboard
        time.sleep(0.15)
        pyperclip.copy(saved)
