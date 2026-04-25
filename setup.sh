#!/usr/bin/env bash
set -euo pipefail

# Pinned WhisperKit commit -- server feature (BUILD_ALL=1) exists from this point onward.
# v0.9.4 tag predates the server; no newer stable tag exists at time of writing.
WHISPERKIT_COMMIT="80d9676"
WHISPERKIT_REPO="https://github.com/argmaxinc/WhisperKit.git"

echo "=== whisper-dictation setup ==="

# --- WhisperKit ---
if [ ! -d "WhisperKit" ]; then
    echo "Cloning WhisperKit..."
    git clone "$WHISPERKIT_REPO" WhisperKit
fi

echo "Pinning WhisperKit to ${WHISPERKIT_COMMIT}..."
git -C WhisperKit checkout "${WHISPERKIT_COMMIT}"

echo "Building WhisperKit server binary (this takes ~4 minutes)..."
# Clear any stale build cache -- paths are absolute and break if the directory moves
rm -rf WhisperKit/.build
( cd WhisperKit && BUILD_ALL=1 swift build -c release 2>&1 )

echo ""
echo "WhisperKit binary: WhisperKit/.build/release/argmax-cli"

# --- Python deps ---
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

echo ""
echo "=== Setup complete ==="
echo "Run:  python3 src/dictate.py"
