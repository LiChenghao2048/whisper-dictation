import subprocess
import time

import pyperclip


class TextTyper:
    def type_text(self, text: str) -> None:
        if not text:
            return
        saved = pyperclip.paste()
        pyperclip.copy(text)
        try:
            subprocess.run(
                [
                    "osascript", "-e",
                    'tell application "System Events" to keystroke "v" using command down',
                ],
                check=True,
                capture_output=True,
            )
            # Brief pause so the paste event completes before restoring the clipboard
            time.sleep(0.15)
        finally:
            pyperclip.copy(saved)
