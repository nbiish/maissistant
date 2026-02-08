#!/bin/bash
set -e

echo "Building Python Sidecar..."
cd python-backend

# Ensure pyinstaller is installed
uv sync

# Build the binary
uv run pyinstaller --clean --noconfirm --name python-backend --onefile src/main.py

# Move to Tauri binaries folder
mkdir -p ../src-tauri/binaries
mv dist/python-backend ../src-tauri/binaries/python-backend-aarch64-apple-darwin

echo "Sidecar built successfully!"

