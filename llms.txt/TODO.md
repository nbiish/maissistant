# TODO

> Keep tasks atomic and testable.

## In Progress

- [ ] Verify Kimi 2.5 default and GLM 4.7 fallback logic in live environment.
- [ ] Test PocketTTS integration for latency and quality.

## Completed

- [x] Replace VoxCPM with PocketTTS (Hugging Face).
- [x] Implement multi-chat support with unique Agno memory databases (`data/sessions/{session_id}.db`).
- [x] Set Kimi 2.5 (`moonshotai/kimi-k2.5`) as the default model for OpenRouter and Zenmux.
- [x] Implement fallback logic to use Z.ai GLM 4.7 (`glm-4.7`) if primary model fails.
- [x] Update Frontend (Settings & Chat) to support new defaults and fallback keys.
- [x] Integrate `pocket-tts` into Python backend.
