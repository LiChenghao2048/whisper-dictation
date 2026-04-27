# whisper-dictation

Hold **Right Command + Right Option** to dictate — speech is transcribed on-device via WhisperKit and typed into whatever window is focused. No cloud, no API key.

---

## How it works

```
Right Command + Right Option held → mic recording starts
Right Command + Right Option released → audio sent to local WhisperKit server → (optional Ollama cleanup) → text typed into focused window
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

| Key | Default | Description |
|---|---|---|
| `hotkey` | `[cmd_r, alt_r]` | one key name or a YAML list for a chord |
| `mode` | `hold` | `hold` or `toggle` |
| `model` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3` |
| `language` | `en` | any Whisper language code (e.g. `zh`) |
| `task` | `transcribe` | `transcribe` or `translate` |
| `temperature` | `0.0` | sampling temperature 0.0–1.0; 0.0 = deterministic |
| `prompt` | _(none)_ | seed text to improve accuracy for names or domain vocabulary |
| `server.port` | `50060` | port the WhisperKit server listens on |

### Prompt — improving accuracy for names and jargon

Set `prompt` to words the model frequently gets wrong:

```yaml
prompt: "Chenghao, WhisperKit, CoreML, argmax-cli"
```

Whisper uses this as prior context when decoding. Particularly useful for proper nouns.

### Temperature — tuning transcription style

`temperature: 0.0` is fully deterministic. Try `0.2`–`0.4` if you find the output too rigid for natural speech; higher values introduce more variation.

### Translation example

```yaml
language: zh
task: translate
```

---

## Ollama LLM cleanup (optional)

Enable local post-processing to automatically remove filler words (`uh`, `um`, `ah`, `eh`, `hmm`) and resolve mid-sentence corrections to their final intended meaning.

**1. Install Ollama:** https://ollama.com

**2. Pull a model:**
```bash
ollama pull llama3.2
```

**3. Enable in `config.yaml`:**
```yaml
cleanup:
  enabled: true
  model: llama3.2   # any model you have pulled
  host: localhost
  port: 11434
```

The Ollama server must be running before you start `dictate.py`. If it is unreachable at startup, cleanup is disabled for that session and a warning is printed — dictation still works normally with the raw Whisper transcript.

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
src/dictate.py      — orchestrator: hotkey → record → queue → transcribe → (cleanup) → type
src/hotkey.py       — global Right Option listener (pynput)
src/audio.py        — microphone recording (sounddevice, 16 kHz mono)
src/server.py       — WhisperKit subprocess lifecycle + HTTP client
src/typer.py        — clipboard → Cmd+V into focused window (osascript)
src/cleanup.py      — Ollama LLM post-processing (filler removal, correction resolution)
src/config.py       — config.yaml loader
WhisperKit/         — local server binary (gitignored, built by setup.sh)
```
