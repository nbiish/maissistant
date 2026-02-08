Here is an updated `TODO.md` for your **agentic programmer** team, accurately reflecting the correct separation of ASR (NVIDIA Parakeet v3 for STT), TTS (VoxCPM CLI for your voice), and agentic brain (Agno with OpenRouter-agnostic LLMs). This structure is tailored for rapid, robust, and provider-agnostic AI voice buddy development using your preferred stack.

***

# TODO.md — Super Smart Buddy Project (Agno + NVIDIA Parakeet + VoxCPM + OpenRouter)

### **I. Project Foundation**

- [ ] **Repository Setup**
    - [ ] Create project directory, initialize Git.
    - [ ] Create standard project subfolders:
        - `/agent` (Agno logic and prompts)
        - `/stt` (Speech-to-text server, configs)
        - `/tts` (Voice cloning, VoxCPM CLI tools)
        - `/scripts` (Helper/control scripts)
        - `/resources` (Reference audio/transcripts)
    - [ ] `.gitignore`: venv, models, audio, cache, outputs.

- [ ] **Virtual Environment**
    - [ ] Python 3.11+ only, `venv` setup.

- [ ] **Dependency Installation**
    - [ ] `pip install agno openai fastapi uvicorn sounddevice numpy scipy pydub requests python-dotenv`
    - [ ] Additional dependencies for Neptune/Transformers/NeMo if running ASR via Python.
    - [ ] Download/install VoxCPM and NVIDIA Parakeet TDT model (doc steps in `README.md`).

***

### **II. Environment Config**

- [ ] **.env Template**
    - [ ] `OPENAI_API_KEY=`
    - [ ] `OPENAI_BASE_URL=https://openrouter.ai/api/v1`
    - [ ] `PARAKEET_ENDPOINT=http://localhost:8001/transcribe`
    - [ ] `VOXCPM_EXEC=./tts/voxCPM_infer.py`

- [ ] **Audio I/O Configuration**
    - [ ] Identify default system mic and output device.
    - [ ] Document fallback device codes for CLI gstreamer/PulseAudio.

***

### **III. Speech-to-Text (STT) — NVIDIA Parakeet Sidecar**

- [ ] **Model Download & Setup**
    - [ ] Download Parakeet TDT v3 model locally.
    - [ ] Provision access to CUDA if available.

- [ ] **API Service**
    - [ ] `stt_server.py`: FastAPI endpoint for `POST /transcribe` ([input: wav/mp3, output: text])
    - [ ] Use NeMo/Transformers for in-server inference, keep model warm/persistent.
    - [ ] Handle batch and streaming input (VAD chunking if practical).
    - [ ] Document sample inference command:
      ```bash
      curl -X POST -F audio=@sample.wav http://localhost:8001/transcribe
      ```

- [ ] **Testing**
    - [ ] Validate with local voice samples, compare against Whisper output for accuracy.

***

### **IV. Text-to-Speech (TTS) — VoxCPM Voice Cloning**

- [ ] **VoxCPM Installation**
    - [ ] Install VoxCPM via repo; download/checkpoint (0.5B+ recommended).

- [ ] **Your Voice Enrollment**
    - [ ] Produce a "golden sample": clean 10–20 sec reference audio ([your_voice_reference.wav]), plus transcript ([your_voice_ref.txt]).
    - [ ] Store in `/resources`.

- [ ] **CLI Wrapper Module**
    - [ ] `tts_wrapper.py`: Python API for
      ```bash
      python inference.py --text "..." --prompt-audio "/resources/..." --output "tts_out.wav"
      ```
    - [ ] Include error codes, latency logging, auto-cleanup.
    - [ ] Option: Daemonize VoxCPM for lower latency on repeated calls.

***

### **V. Agno Agent Brain (Provider-Agnostic LLM via OpenRouter)**

- [ ] **Agno Setup**
    - [ ] Configure agent persona, memory, and prompt style (your voice/personality) in `/agent`.
    - [ ] Plug in OpenAI client with OpenRouter endpoint.
    - [ ] Expose LLM swap via config var or CLI arg.

- [ ] **Persona & Memory**
    - [ ] Define persona prompt: "You are a super smart buddy—technical, concise, and mimic my speaking style."
    - [ ] Enable session persistence (local DB or file-system memory).
    - [ ] Option: add file tool/search tool for agent to reference docs.

***

### **VI. Integration Loop (main.py)**

1. **Record Audio:** Capture from mic (push-to-talk or always-on w/ VAD)
2. **Transcribe:** Send buffer to `/stt/transcribe`, receive recognized text.
3. **LLM Reasoning:** Feed recognized text to Agno, receive agent response.
4. **Synthesize (TTS):** Send agent response to `tts_wrapper.py`, play back synthesized wav in your voice.

- [ ] Implement quick error handling at each stage for robustness.
- [ ] Optionally: support "streaming" TTS for lower perceived latency (sentence-by-sentence).

***

### **VII. Testing and Optimization**

- [ ] Full round-trip test (audio → text → agent → TTS → audio playback)
- [ ] Test LLM provider swap (try different models via OpenRouter)
- [ ] Tune voice sample, transcript alignment, and LLM system prompts as needed.

***

### **VIII. Documentation & Continuous Improvement**

- [ ] Keep all install/config/setup/testing steps updated in `README.md`
- [ ] Separate fast-path troubleshooting and advanced configuration guides for streamlined onboarding.

***

This clearly defines each build phase and guarantees modularity—letting your team optimize or swap out each major component (ASR, Agent, TTS, LLM provider) independently.  
Let me know if you want a sample skeleton code layout, CLI command set, or planned API schemas next.