A good `TODO.md` for this project should organize work into phases for infra, voice I/O, agent architecture, and self-improving “agentic programmers,” all wired through an OpenAI-compatible client that defaults to OpenRouter while remaining provider-agnostic.  It should also explicitly separate STT (Parakeet v3 on Ollama) and TTS (VoxCPM CLI voice cloning) tasks so agents can independently improve each layer of the voice-to-voice loop.[1][2][3][4][5][6]

Below is a suggested `TODO.md` you can drop straight into your repo and then refine.[7]

***

### Project overview  

- Build a local-first, voice-to-voice “super smart buddy” that speaks in the user’s cloned voice, using Agno agents orchestrated over an OpenAI-compatible API that defaults to OpenRouter.[4][6]
- Use NVIDIA Parakeet v3 (running via Ollama) for multilingual, high-throughput speech-to-text, and VoxCPM via CLI for tokenizer-free, zero-shot voice cloning and TTS.[2][3][5][8][9][1]
- Keep the LLM layer fully provider-agnostic by using the OpenAI SDK with a configurable `base_url` and `model` so agents can swap between OpenRouter and any other OpenAI-compatible endpoint.[6][10][11][4]

***

### Phase 0 – Repo bootstrap & configuration  

- [ ] Initialize `super-smart-buddy` repo, add `.gitignore`, and create top-level dirs: `src/`, `config/`, `scripts/`, `agents/`, `voice/`, `tests/`.[12][7]
- [ ] Add `pyproject.toml` or `requirements.txt` with Agno, OpenAI SDK, and any CLI helpers you prefer for subprocess management and async I/O.[13][4][6]
- [ ] Define `.env.example` with `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENROUTER_SITE_URL`, `OPENROUTER_APP_NAME`, `OLLAMA_BASE_URL`, `VOXCPM_BINARY`, and `AGNO_DB_PATH`.[11][4][6]
- [ ] Implement a small `src/config.py` that loads env vars, validates required keys, and exposes a typed settings object for all agents to share.[12]

***

### Phase 1 – Provider-agnostic LLM client (OpenAI SDK + OpenRouter)  

- [ ] Implement `src/llm/client.py` that wraps the OpenAI SDK and accepts `base_url`, `api_key`, and `model` from config, defaulting `base_url` to OpenRouter’s OpenAI-compatible endpoint.[10][4][6][11]
- [ ] Add simple streaming and non-streaming helpers like `chat(messages, **kwargs)` and `stream_chat(messages, **kwargs)` that work identically across providers.[6][11]
- [ ] Implement a model registry or enum (e.g., `SMART_BUDDY_MODEL`, `CODER_MODEL`) that can be re-pointed to different OpenRouter or other OpenAI-compatible models without code changes.[4][11][6]
- [ ] Add a smoke test script `scripts/test_llm_client.py` that sends a trivial prompt and prints the response and latency, verifying OpenRouter compatibility.[10][11][4]

***

### Phase 2 – STT pipeline (Parakeet v3 on Ollama)  

- [ ] Create `voice/stt_parakeet.py` with a `transcribe(audio_path: str) -> str` function that POSTs to the local Ollama server running Parakeet v3.[8][1][2]
- [ ] Define configuration for model name (e.g., `parakeet-tdt-0.6b-v3`) and max audio length, aligned with the model’s high-throughput, long-context capabilities.[1][2][8]
- [ ] Implement a CLI helper `scripts/record_and_transcribe.py` that records from the default mic, saves `wav`, calls `transcribe`, and prints the transcript.[14][2][1]
- [ ] Add basic error handling for timeouts, empty audio, and STT failures, returning structured errors that agents can reason about.[7]

***

### Phase 3 – TTS + voice cloning pipeline (VoxCPM CLI)  

- [ ] Create `voice/tts_voxcpm.py` with a `synthesize(text: str, output_path: str) -> None` function that shells out to the VoxCPM CLI binary.[3][5][9]
- [ ] Add configuration for reference voice audio path and optional reference transcript, matching VoxCPM’s zero-shot voice cloning interface.[5][9][15][3]
- [ ] Support both “reference-bootstrapped voice” mode and “generic TTS” mode, so the buddy can fall back gracefully if reference audio is missing.[3][5]
- [ ] Implement a `scripts/say.py` test script that takes text, calls VoxCPM, and plays the resulting audio, validating end-to-end CLI integration.[5][3]

***

### Phase 4 – Core Agno agents (listener, thinker, speaker)  

- [ ] Define an Agno `listener_agent` that continuously or intermittently captures mic input, calls Parakeet v3 STT, and emits structured messages like `{ text, timestamp, confidence }`.[2][13][1]
- [ ] Define a `thinker_agent` that takes listener messages and calls the provider-agnostic LLM client, conditioned with a system prompt that encodes “super smart buddy that thinks like me.”[13][4][6]
- [ ] Define a `speaker_agent` that accepts the LLM’s response and routes it to VoxCPM TTS, returning an audio file path and any playback metadata.[3][5][13]
- [ ] Implement an Agno “orchestrator” workflow that wires `listener -> thinker -> speaker` into a single voice-to-voice loop, with hooks for logging and memory.[12][13]

***

### Phase 5 – Agentic programmers (self-updating code agents)  

- [ ] Create a `agents/programmer_agent.py` that uses the same OpenAI-compatible client but is specialized for reading the repo, editing files, and proposing diffs to improve performance, UX, or robustness.[4][6][13][12]
- [ ] Give the programmer agent tools for reading and writing files, running tests, and profiling latency of STT, LLM, and TTS segments, similar to HGM-style self-improving agents.[13][12]
- [ ] Add a `TODO.md` parsing convention where programmer agents claim tasks by appending their name or status tags like `[agent:programmer][status:in-progress]`.[12]
- [ ] Implement a review gate so that programmer changes are either human-reviewed or passed through a separate “critic” agent before merging.[12]

***

### Phase 6 – Memory, personality, and context  

- [ ] Hook the Agno memory DB (e.g., `agno.db` or `mai-memories.db`) into the `thinker_agent` so the buddy can recall user facts, preferences, and past conversations.[12]
- [ ] Make the memory database path swap-able via an env variable such as `MAI_AGENT_FACTORY_MEMORY` to allow different personas or contexts.[12]
- [ ] Iterate on the system prompt and few-shot examples to align the buddy’s personality with the user’s voice, values, and style, using the cloned voice as an additional grounding signal.[5][7][3]

***

### Phase 7 – DX, monitoring, and tests  

- [ ] Add a rich CLI status UI (e.g., text-based dashboard) showing STT transcript, LLM thoughts, and TTS status in real time for debugging and demos.[7]
- [ ] Write unit and integration tests for `llm/client.py`, `stt_parakeet.py`, and `tts_voxcpm.py`, including provider switching tests for the OpenAI-compatible layer.[6][4]
- [ ] Add basic telemetry and logging (timestamps, latency per stage, error counts) so programmer agents can optimize the slowest parts of the pipeline over time.[12]

If you want, the next step can be turning this into a more “agent-readable” format with explicit tool schemas and function signatures for each agent role.[13][12]

[1](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
[2](https://blogs.nvidia.com/blog/speech-ai-dataset-models/)
[3](https://github.com/OpenBMB/VoxCPM)
[4](https://langfuse.com/integrations/gateways/openrouter)
[5](https://voxcpm.com)
[6](https://openrouter.ai/docs/api-reference/overview)
[7](https://www.perplexity.ai/search/0f3513a0-a016-4758-8614-b9310470003f)
[8](https://huggingface.co/models?other=base_model%3Aquantized%3Anvidia%2Fparakeet-tdt-0.6b-v3)
[9](https://voxcpm.net)
[10](https://www.reddit.com/r/OpenWebUI/comments/1gfr7hp/openrouter/)
[11](https://fal.ai/models/openrouter/router/openai/v1/responses)
[12](https://www.perplexity.ai/search/1c18bdf5-79da-4b94-8ffc-a5aa5e1dc53a)
[13](https://www.perplexity.ai/search/e966c4fb-b21b-43bc-83ee-5f52946a264a)
[14](https://www.perplexity.ai/search/0dac748a-c861-488c-8699-e24990f272ff)
[15](https://www.reddit.com/r/LocalLLaMA/comments/1njzxmx/voxcpm_05b_tokenizerfree_tts_and_voice_cloning/)
[16](https://www.reddit.com/r/LocalLLaMA/comments/1mv6wwe/nvidiaparakeettdt06bv3_now_multilingual/)
[17](https://www.facebook.com/groups/DeepNetGroup/posts/2478770149182519/)
[18](https://www.youtube.com/watch?v=5xaojV7rZvw)
[19](https://github.com/wildminder/ComfyUI-VoxCPM)
[20](https://dev.to/nodeshiftcloud/how-to-install-nvidia-parakeet-tdt-06b-v2-locally-36ck)
[21](https://www.youtube.com/watch?v=tr7SDfmInIY)
[22](https://mastra.ai/models/providers/nvidia)
[23](https://community.n8n.io/t/how-to-connect-an-openai-compatible-model-to-ai-agent/162897)
[24](https://x.com/nithinraok_)