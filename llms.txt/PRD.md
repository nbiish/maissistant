# Product Requirements Document (PRD)

## Project Overview
- **Name:** MAIssistant
- **Version:** 0.2.0
- **Description:** A desktop AI assistant that integrates multiple LLM providers (Gemini Vertex, OpenRouter, Zenmux) and provides visual context awareness through screen capture capabilities.
- **Purpose:** To allow users to get AI insights about their current screen content (specific windows or monitors) using their preferred AI models.

## Core Features

### 1. Multi-Model Architecture
- **Providers:**
  - **Gemini Vertex API:** Direct integration with Google's Vertex AI models.
  - **OpenRouter:** Access to a wide range of models via OpenRouter API.
  - **Zenmux:** Integration with Zenmux platform for enterprise/aggregated models.
- **User Configuration:**
  - Users must be able to input and save API keys for each provider.
  - Users can select the active provider and specific model from a dropdown.

### 2. Visual Context & Screen Capture
- **Source Selection:**
  - Users can list available monitors and open application windows.
  - Users can select a specific source (Monitor 1, Chrome Window, etc.) to "watch" or capture on demand.
- **Capture Mechanism:**
  - On-demand capture when a query is sent.
  - Images are processed and sent to the multimodal LLM.

### 3. User Interface
- **Chat Interface:** Standard chat UI for Q&A.
- **Settings Panel:**
  - API Key management.
  - Model selection.
- **Capture Controls:**
  - Source selector (Window/Monitor).
  - Preview of captured content (optional/thumbnail).

## Technical Architecture

### Frontend (React + TypeScript)
- Manages UI state, user inputs, and settings.
- Communicates with Rust backend for system operations (window listing, capturing).
- Communicates with Python backend for AI inference.

### Backend (Python - FastAPI)
- Acts as the AI Orchestrator.
- **Modules:**
  - `model_manager`: Handles initialization and calls to Gemini, OpenRouter, Zenmux.
  - `processing`: Handles image encoding/decoding.
  - `api`: FastAPI endpoints for chat and configuration.
- **Storage:** API keys should be stored securely (or passed per request from the secure frontend/Rust storage). *Decision: Pass keys from frontend for now to keep backend stateless regarding secrets, or store in `.env` managed by frontend.* -> *Refined: Frontend stores keys in Tauri Store (local disk encrypted/secure) and sends them with requests.*

### System Layer (Rust - Tauri)
- **Capabilities:**
  - `list_windows()`: Returns list of open windows.
  - `list_monitors()`: Returns list of monitors.
  - `capture_screen(source_id)`: Captures screenshot and returns Base64 string.

## Data Flow
1. User configures API Keys in Settings.
2. User selects "Chrome" as capture source.
3. User asks "What is on this page?".
4. Frontend calls Rust -> `capture_screen("Chrome_ID")`.
5. Rust returns Base64 Image.
6. Frontend calls Python -> `/chat` with `{ message: "...", image: "base64...", provider: "...", api_key: "..." }`.
7. Python `model_manager` selects provider, formats payload, calls external API.
8. Response is returned to Frontend and displayed.
