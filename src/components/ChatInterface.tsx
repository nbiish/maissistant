import { useState, useRef, useEffect } from 'react';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { invoke } from '@tauri-apps/api/core';
import { Store } from "@tauri-apps/plugin-store";

const store = new Store("settings.json");

// Types
interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  image?: string;
}

interface CaptureSource {
    id: string;
    name: string;
    kind: string;
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState('');
  
  // Audio
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  
  // Capture
  const [sources, setSources] = useState<CaptureSource[]>([]);
  const [selectedSourceId, setSelectedSourceId] = useState<string>("");
  const [isCapturing, setIsCapturing] = useState(false);

  // Settings
  const [provider, setProvider] = useState("openrouter");
  const [apiKey, setApiKey] = useState("");
  const [fallbackKey, setFallbackKey] = useState("");
  const [model, setModel] = useState("moonshotai/kimi-k2.5");

  // Pause & Toggles
  const [isPaused, setIsPaused] = useState(false);
  const [isScreenEnabled, setIsScreenEnabled] = useState(true);
  const [isVoiceEnabled, setIsVoiceEnabled] = useState(true);

  useEffect(() => {
    const win = getCurrentWindow();
    // Initialize session ID with window label + timestamp for uniqueness or just random
    setSessionId(`${win.label}-${Date.now()}`);
    
    // Initial load
    refreshSources();
    loadSettings();

    // Reload settings on focus (simple way to catch updates from Settings window)
    const unlisten = win.onFocusChanged((event) => {
        if (event.payload) loadSettings();
    });

    return () => {
        unlisten.then(f => f());
    }
  }, []);

  // Pause Effect: Stop recording if paused
  useEffect(() => {
    if (isPaused) {
        if (isRecording) {
            mediaRecorderRef.current?.stop();
            setIsRecording(false);
        }
    }
  }, [isPaused, isRecording]);

  const loadSettings = async () => {
      try {
        const p = await store.get<string>("active_provider");
        if (p) {
            setProvider(p);
            let keyKey = "gemini_api_key";
            if (p === "openrouter") keyKey = "openrouter_api_key";
            if (p === "zenmux" || p === "z.ai-code") keyKey = "zenmux_api_key";
            
            const key = await store.get<string>(keyKey);
            if (key) setApiKey(key);
        } else {
            // Default
             const key = await store.get<string>("openrouter_api_key");
             if (key) setApiKey(key);
        }
        
        // Always load fallback key (Z.ai/Zenmux key) unless we are already using it
        const zKey = await store.get<string>("zenmux_api_key");
        if (zKey) setFallbackKey(zKey);

        const m = await store.get<string>("active_model");
        if (m) setModel(m);
      } catch (e) {
          console.error("Error loading settings:", e);
      }
  };

  const refreshSources = async () => {
      try {
          const s = await invoke<CaptureSource[]>("list_sources");
          setSources(s);
      } catch (e) {
          console.error("Failed to list sources", e);
      }
  };

  const startNewChat = () => {
      setMessages([]);
      const newId = `session-${Date.now()}`;
      setSessionId(newId);
      console.log("Started new chat:", newId);
  };

  const playAudio = (base64Audio: string) => {
      try {
          const audioStr = `data:audio/wav;base64,${base64Audio}`;
          const audio = new Audio(audioStr);
          audio.play().catch(e => console.error("Playback error:", e));
      } catch (e) {
          console.error("Error playing audio:", e);
      }
  };

  const sendMessage = async (text: string) => {
    if (isPaused) return;

    let imageBase64: string | undefined = undefined;
    
    // Capture logic
    if (selectedSourceId && isScreenEnabled) {
        const source = sources.find(s => s.id === selectedSourceId);
        if (source) {
            setIsCapturing(true);
            try {
                imageBase64 = await invoke<string>("capture_source", { id: source.id, kind: source.kind });
            } catch (e) {
                setMessages(prev => [...prev, { role: 'system', content: `Capture failed: ${e}` }]);
                setIsCapturing(false);
                return; // Stop if capture failed but was requested? Or continue? Let's stop.
            }
            setIsCapturing(false);
        }
    }

    const newMsg: Message = { role: 'user', content: text, image: imageBase64 };
    setMessages(prev => [...prev, newMsg]);
    
    try {
      // Ensure we have settings
      await loadSettings();

      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            message: text,
            provider: provider,
            api_key: apiKey,
            fallback_key: fallbackKey,
            model: model,
            image: imageBase64,
            session_id: sessionId 
        })
      });
      
      if (!response.ok) {
          const err = await response.json();
          throw new Error(err.detail || "Request failed");
      }

      const data = await response.json();
      setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
      
    } catch (e) {
      console.error(e);
      setMessages(prev => [...prev, { role: 'system', content: `Error: ${e}` }]);
    }
  };

  const toggleRecording = async () => {
    // Stop if currently recording
    if (isRecording) {
        mediaRecorderRef.current?.stop();
        setIsRecording(false);
        return;
    }
    
    // Prevent start if paused or voice disabled
    if (isPaused || !isVoiceEnabled) return;
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const recorder = new MediaRecorder(stream);
        mediaRecorderRef.current = recorder;
        chunksRef.current = [];
        
        recorder.ondataavailable = (e) => {
            if (e.data.size > 0) chunksRef.current.push(e.data);
        };

        recorder.onstop = async () => {
            const blob = new Blob(chunksRef.current, { type: 'audio/wav' });
            
            // STT Logic
            try {
                const formData = new FormData();
                formData.append('file', blob, 'recording.wav');
                const response = await fetch('http://localhost:8000/transcribe', { 
                    method: 'POST', 
                    body: formData 
                });
                const data = await response.json();
                if (data.text) {
                    setInput(data.text); // Set transcribed text to input
                }
            } catch (e) {
                console.error("Transcription failed", e);
            }
            
            stream.getTracks().forEach(t => t.stop());
        };

        recorder.start();
        setIsRecording(true);
    } catch (e) {
        console.error("Mic error", e);
    }
  };

  const handleSend = () => {
    if (!input.trim()) return;
    sendMessage(input);
    setInput('');
  };

  return (
    <div className="chat-container" style={{ padding: '20px', display: 'flex', flexDirection: 'column', height: '100vh', boxSizing: 'border-box', color: '#fff', background: '#1e1e1e' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
          <div>
            <h2 style={{margin: 0}}>{provider.toUpperCase()} Agent</h2>
            <div style={{ fontSize: '0.7em', color: '#777' }}>Session: {sessionId}</div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            {/* Global Pause Button */}
            <button 
                onClick={() => setIsPaused(!isPaused)} 
                style={{ 
                    padding: '5px 15px', 
                    background: isPaused ? '#2ecc71' : '#f39c12', 
                    color: '#fff', 
                    border: 'none', 
                    borderRadius: '4px', 
                    cursor: 'pointer', 
                    fontWeight: 'bold',
                    minWidth: '80px'
                }}
            >
                {isPaused ? "▶ Resume" : "⏸ Pause"}
            </button>

            <div style={{ fontSize: '0.8em', color: '#aaa' }}>{model}</div>
            <button onClick={startNewChat} style={{ padding: '5px 10px', background: '#2ecc71', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer', fontSize: '0.8em' }}>
                New Chat
            </button>
          </div>
      </div>
      
      {/* Capture Controls */}
      <div style={{ display: 'flex', gap: '10px', marginBottom: '10px', background: '#333', padding: '8px', borderRadius: '4px', alignItems: 'center' }}>
          <select 
            value={selectedSourceId} 
            onChange={(e) => setSelectedSourceId(e.target.value)}
            disabled={isPaused}
            style={{ flex: 1, padding: '5px', background: isPaused ? '#444' : '#222', color: isPaused ? '#aaa' : '#fff', border: '1px solid #555' }}
          >
              <option value="">No Screen Capture</option>
              {sources.map(s => (
                  <option key={s.id} value={s.id}>[{s.kind.toUpperCase()}] {s.name}</option>
              ))}
          </select>
          
          <label style={{ display: 'flex', alignItems: 'center', gap: '5px', color: '#ccc', fontSize: '0.9em', cursor: 'pointer' }}>
            <input 
                type="checkbox" 
                checked={isScreenEnabled} 
                onChange={e => setIsScreenEnabled(e.target.checked)}
                disabled={isPaused}
            />
            Screen
          </label>

          <button onClick={refreshSources} disabled={isPaused} style={{ padding: '5px 10px', cursor: 'pointer', background: '#555', color: '#fff', border: 'none', borderRadius: '3px', opacity: isPaused ? 0.5 : 1 }}>
              ↻
          </button>
      </div>

      <div className="messages" style={{ flex: 1, overflowY: 'auto', border: '1px solid #333', borderRadius: '8px', padding: '10px', marginBottom: '10px', background: '#252526' }}>
        {messages.length === 0 && <p style={{color: '#888', textAlign: 'center'}}>Select a source and ask a question...</p>}
        {messages.map((m, i) => (
          <div key={i} className={`message ${m.role}`} style={{ marginBottom: '15px', padding: '10px', borderRadius: '6px', background: m.role === 'user' ? 'rgba(0, 120, 255, 0.2)' : 'rgba(255, 255, 255, 0.1)' }}>
            <div style={{ fontWeight: 'bold', marginBottom: '5px', color: m.role === 'user' ? '#61dafb' : '#eee' }}>{m.role.toUpperCase()}</div>
            {m.image && (
                <div style={{ marginBottom: '8px' }}>
                    <img src={`data:image/png;base64,${m.image}`} alt="capture" style={{ maxWidth: '100%', maxHeight: '200px', borderRadius: '4px', border: '1px solid #555' }} />
                </div>
            )}
            <div style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>
            {m.role === 'assistant' && (
                <button 
                    onClick={async () => {
                        try {
                            const res = await fetch('http://localhost:8000/speak', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ text: m.content })
                            });
                            const data = await res.json();
                            if (data.status === 'success' && data.audio) {
                                playAudio(data.audio);
                            } else {
                                console.error("TTS Error:", data);
                            }
                        } catch (e) {
                            console.error("TTS Request Failed:", e);
                        }
                    }}
                    style={{marginLeft: '10px', fontSize: '0.8em', cursor: 'pointer', background: 'none', border: 'none'}}
                >
                    🔊
                </button>
            )}
          </div>
        ))}
        {isCapturing && <div style={{ color: '#aaa', fontStyle: 'italic' }}>Capturing screen...</div>}
      </div>

      <div className="input-area" style={{ display: 'flex', gap: '10px' }}>
        <input 
          style={{ flex: 1, padding: '10px', borderRadius: '4px', border: '1px solid #555', background: isPaused ? '#444' : '#2d2d2d', color: isPaused ? '#aaa' : '#fff' }}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && !isPaused && handleSend()}
          placeholder={isPaused ? "Application Paused" : "Ask about the selected window..."}
          disabled={isCapturing || isPaused}
        />
        
        <label style={{ display: 'flex', alignItems: 'center', gap: '5px', color: '#ccc', fontSize: '0.9em', cursor: 'pointer', marginRight: '5px' }}>
            <input 
                type="checkbox" 
                checked={isVoiceEnabled} 
                onChange={e => setIsVoiceEnabled(e.target.checked)}
                disabled={isPaused}
            />
            Voice
        </label>

        <button 
            onClick={toggleRecording} 
            disabled={isPaused || !isVoiceEnabled}
            style={{ 
                padding: '10px 20px', 
                borderRadius: '4px', 
                border: 'none', 
                background: isRecording ? '#e74c3c' : '#555', 
                color: '#fff', 
                cursor: (isPaused || !isVoiceEnabled) ? 'not-allowed' : 'pointer', 
                marginRight: '10px',
                opacity: (isPaused || !isVoiceEnabled) ? 0.5 : 1
            }}
        >
          {isRecording ? 'Stop' : 'Mic'}
        </button>
        <button 
            onClick={handleSend} 
            disabled={isCapturing || isPaused} 
            style={{ 
                padding: '10px 20px', 
                borderRadius: '4px', 
                border: 'none', 
                background: '#0078ff', 
                color: '#fff', 
                cursor: (isCapturing || isPaused) ? 'not-allowed' : 'pointer', 
                opacity: (isCapturing || isPaused) ? 0.5 : 1 
            }}
        >
          Send
        </button>
      </div>
    </div>
  );
}
