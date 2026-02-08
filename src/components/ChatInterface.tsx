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
  const [windowLabel, setWindowLabel] = useState('');
  
  // Capture
  const [sources, setSources] = useState<CaptureSource[]>([]);
  const [selectedSourceId, setSelectedSourceId] = useState<string>("");
  const [isCapturing, setIsCapturing] = useState(false);

  // Settings
  const [provider, setProvider] = useState("gemini");
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("gemini-1.5-flash");

  useEffect(() => {
    const win = getCurrentWindow();
    setWindowLabel(win.label);
    
    // Initial load
    refreshSources();
    loadSettings();

    // Reload settings on focus (simple way to catch updates from Settings window)
    const unlisten = win.onFocusChanged(({ focused }) => {
        if (focused) loadSettings();
    });

    return () => {
        unlisten.then(f => f());
    }
  }, []);

  const loadSettings = async () => {
      try {
        const p = await store.get<string>("active_provider");
        if (p) {
            setProvider(p);
            let keyKey = "gemini_api_key";
            if (p === "openrouter") keyKey = "openrouter_api_key";
            if (p === "zenmux") keyKey = "zenmux_api_key";
            
            const key = await store.get<string>(keyKey);
            if (key) setApiKey(key);
        } else {
            // Default
             const key = await store.get<string>("gemini_api_key");
             if (key) setApiKey(key);
        }
        
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

  const sendMessage = async (text: string) => {
    let imageBase64: string | undefined = undefined;
    
    // Capture logic
    if (selectedSourceId) {
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
            model: model,
            image: imageBase64,
            session_id: windowLabel 
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

  const handleSend = () => {
    if (!input.trim()) return;
    sendMessage(input);
    setInput('');
  };

  return (
    <div className="chat-container" style={{ padding: '20px', display: 'flex', flexDirection: 'column', height: '100vh', boxSizing: 'border-box', color: '#fff', background: '#1e1e1e' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
          <h2>{provider.toUpperCase()} Agent</h2>
          <div style={{ fontSize: '0.8em', color: '#aaa' }}>{model}</div>
      </div>
      
      {/* Capture Controls */}
      <div style={{ display: 'flex', gap: '10px', marginBottom: '10px', background: '#333', padding: '8px', borderRadius: '4px' }}>
          <select 
            value={selectedSourceId} 
            onChange={(e) => setSelectedSourceId(e.target.value)}
            style={{ flex: 1, padding: '5px', background: '#222', color: '#fff', border: '1px solid #555' }}
          >
              <option value="">No Screen Capture</option>
              {sources.map(s => (
                  <option key={s.id} value={s.id}>[{s.kind.toUpperCase()}] {s.name}</option>
              ))}
          </select>
          <button onClick={refreshSources} style={{ padding: '5px 10px', cursor: 'pointer', background: '#555', color: '#fff', border: 'none', borderRadius: '3px' }}>
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
          </div>
        ))}
        {isCapturing && <div style={{ color: '#aaa', fontStyle: 'italic' }}>Capturing screen...</div>}
      </div>

      <div className="input-area" style={{ display: 'flex', gap: '10px' }}>
        <input 
          style={{ flex: 1, padding: '10px', borderRadius: '4px', border: '1px solid #555', background: '#2d2d2d', color: '#fff' }}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Ask about the selected window..."
          disabled={isCapturing}
        />
        <button onClick={handleSend} disabled={isCapturing} style={{ padding: '10px 20px', borderRadius: '4px', border: 'none', background: '#0078ff', color: '#fff', cursor: 'pointer', opacity: isCapturing ? 0.5 : 1 }}>
          Send
        </button>
      </div>
    </div>
  );
}
