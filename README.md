# MAIssistant

A local-first, voice-enabled AI agent desktop application built with Tauri v2, React, and Agno.

## Prerequisites

- **Node.js** (v18+) & **pnpm**
- **Rust** (latest stable)
- **Python** (v3.11+) & **uv** (package manager)
- **FFmpeg** (recommended for audio handling)

## Setup

1. **Install Dependencies**
   ```bash
   pnpm install
   ```

2. **Setup Python Backend**
   ```bash
   cd python-backend
   uv sync
   cd ..
   ```

3. **Build Python Sidecar**
   This compiles the Python API into a standalone binary for Tauri.
   ```bash
   chmod +x scripts/build_sidecar.sh
   ./scripts/build_sidecar.sh
   ```

## Development

Run the Tauri development server (this will launch the app and sidecar):
```bash
pnpm tauri dev
```

## Architecture

- **Frontend**: React + TypeScript + Vite (Multi-page: Main, Settings)
- **Backend**: Python FastAPI (packaged as a Sidecar via PyInstaller)
- **Desktop Shell**: Tauri v2 (Rust)
- **Agent Logic**: Agno (in Python)
- **Storage**: `tauri-plugin-store` (local JSON)

## Features

- **System Tray**: Manage agent instances from the Mac menu bar.
- **Multi-Agent**: Spawn independent agent windows.
- **Voice Integration**: Push-to-talk and TTS (via local models or API).
- **Settings**: Configurable OpenAI API Key, Models, and MCP Servers.
