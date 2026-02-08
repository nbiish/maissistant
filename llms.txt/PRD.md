# Product Requirements Document (PRD)

## Project Overview
- **Name:** MAIssistant
- **Version:** 0.3.0
- **Description:** A desktop AI assistant that integrates multiple LLM providers (OpenRouter, Zenmux, Z.ai) with robust fallback mechanisms, local TTS, and visual context awareness.
- **Purpose:** To provide a resilient, multi-modal AI assistant that "sees" your screen and "speaks" with low latency.

## Core Features

### 1. Multi-Model Architecture & Resilience
- **Providers:**
  - **OpenRouter & Zenmux:** Primary providers using **Kimi 2.5** (`moonshotai/kimi-k2.5`) for cost-effective multi-modal reasoning.
  - **Z.ai:** Secondary/Fallback provider using **GLM 4.7** (`glm-4.7`).
- **Fallback Logic:**
  - If the primary provider (OpenRouter/Zenmux) fails (network, rate limit, routing), the system automatically retries with Z.ai GLM 4.7.
  - Frontend automatically handles credential passing for both primary and fallback providers.

### 2. Local Text-to-Speech (TTS)
- **Engine:** **PocketTTS** (Hugging Face).
- **Voice:** Default `alba-mackenna/casual` (configurable in code).
- **Performance:** Local CPU/GPU inference for privacy and low latency.

### 3. Visual Context & Screen Capture
- **Source Selection:** List and select specific windows or monitors.
- **Capture:** On-demand screenshot capture sent as Base64 to multi-modal models.

### 4. Memory & Session Management
- **Multi-Chat:** Support for multiple concurrent chat sessions.
- **Isolation:** Each chat session has its own dedicated SQLite database (`data/sessions/{id}.db`) for strict memory separation using Agno.

## Technical Architecture

### Frontend (React + TypeScript)
- **State Management:** Manages chat history, capturing state, and settings.
- **Settings:** Configures API keys for OpenRouter, Zenmux, and Z.ai.
- **Audio:** Plays Base64 audio returned from backend.

### Backend (Python - FastAPI)
- **Agent Brain:** 
  - Manages Agno agents.
  - Handles dynamic DB creation per session.
  - Executes fallback logic (Primary -> Fallback).
- **TTS Manager:** Wraps PocketTTS for speech generation.
- **API:** Exposes `/chat` (text/image) and `/speak` (TTS) endpoints.

### System Layer (Rust - Tauri)
- **Window/Screen Management:** Lists sources and captures screenshots.
- **Persistence:** Securely stores settings and API keys.

## Data Flow
1. **Chat:** User sends message -> Frontend includes Primary Key + Fallback Key -> Backend tries Primary Model -> If Fail, Backend uses Fallback Key & Model -> Response returned.
2. **TTS:** User clicks "Speak" -> Backend generates Audio (PocketTTS) -> Base64 Audio returned -> Frontend plays Blob.
