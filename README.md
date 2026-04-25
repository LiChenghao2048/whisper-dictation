# whisper-dictation

Hold **Right Command + Right Option** to dictate — speech is transcribed on-device via WhisperKit and typed into whatever window is focused. No cloud, no API key.

---

## How it works

```
Right Command + Right Option held → mic recording starts
Right Command + Right Option released → audio sent to local WhisperKit server → text typed into focused window
```

The WhisperKit server loads the model once at startup (~90 s) and stays warm for the entire session. All inference runs on Apple Silicon (Neural Engine).

---

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- macOS 14.0 (Sonoma) or later
- Xcode 15+ (for building WhisperKit)
- Python 3.11+
- Accessibility permission granted to Terminal (System Settings → Privacy & Security → Accessibility)

---

## Setup

```bash
cd whisper-dictation
./setup.sh
```

`setup.sh` clones WhisperKit at commit `80d9676` (the first version with the local server feature), builds the server binary, and installs Python dependencies.

---

## Running

```bash
python3 src/dictate.py
```

Wait for `ready — hold cmd_r+alt_r to dictate`, then hold **Right Command + Right Option** and speak. Release to transcribe.

Stop with **Ctrl+C**.

---

## Configuration

Edit `config.yaml`:

| Key | Default | Options |
|---|---|---|
| `hotkey` | `[cmd_r, alt_r]` | one key name or a YAML list for a chord |
| `mode` | `hold` | `hold`, `toggle` |
| `model` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3` |
| `language` | `en` | any Whisper language code (e.g. `zh`) |
| `task` | `transcribe` | `transcribe`, `translate` |
| `server.port` | `50060` | any free port |

To switch to Chinese → English translation:
```yaml
language: zh
task: translate
```

---

## Running tests

No device or running server needed — all I/O is mocked.

```bash
pip3 install -r requirements-dev.txt
pytest
```

---

## Stack

```
src/dictate.py      — orchestrator: hotkey → record → queue → transcribe → type
src/hotkey.py       — global Right Option listener (pynput)
src/audio.py        — microphone recording (sounddevice, 16 kHz mono)
src/server.py       — WhisperKit subprocess lifecycle + HTTP client
src/typer.py        — clipboard → Cmd+V into focused window (osascript)
src/config.py       — config.yaml loader
WhisperKit/         — local server binary (gitignored, built by setup.sh)
```
