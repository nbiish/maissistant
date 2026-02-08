import { useState, useEffect } from "react";
import { Store } from "@tauri-apps/plugin-store";

const store = new Store("settings.json");

export default function SettingsApp() {
  const [activeProvider, setActiveProvider] = useState("gemini");
  const [geminiApiKey, setGeminiApiKey] = useState("");
  const [openRouterApiKey, setOpenRouterApiKey] = useState("");
  const [zenmuxApiKey, setZenmuxApiKey] = useState("");
  const [model, setModel] = useState("gemini-1.5-flash");

  useEffect(() => {
    const loadSettings = async () => {
      const provider = await store.get<string>("active_provider");
      const gKey = await store.get<string>("gemini_api_key");
      const oKey = await store.get<string>("openrouter_api_key");
      const zKey = await store.get<string>("zenmux_api_key");
      const mod = await store.get<string>("active_model");

      if (provider) setActiveProvider(provider);
      if (gKey) setGeminiApiKey(gKey);
      if (oKey) setOpenRouterApiKey(oKey);
      if (zKey) setZenmuxApiKey(zKey);
      if (mod) setModel(mod);
    };
    loadSettings();
  }, []);

  const saveSettings = async () => {
    await store.set("active_provider", activeProvider);
    await store.set("gemini_api_key", geminiApiKey);
    await store.set("openrouter_api_key", openRouterApiKey);
    await store.set("zenmux_api_key", zenmuxApiKey);
    await store.set("active_model", model);
    await store.save();
    alert("Settings saved!");
  };

  return (
    <div className="container" style={{ padding: "20px", color: "#fff" }}>
      <h1>Settings</h1>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          saveSettings();
        }}
        style={{ display: "flex", flexDirection: "column", gap: "1rem" }}
      >
        <div style={{ display: "flex", flexDirection: "column" }}>
          <label>Active Provider:</label>
          <select 
            value={activeProvider} 
            onChange={(e) => setActiveProvider(e.target.value)}
            style={{ padding: "8px", borderRadius: "4px", background: "#333", color: "#fff", border: "1px solid #555" }}
          >
            <option value="gemini">Gemini Vertex/AI Studio</option>
            <option value="openrouter">OpenRouter</option>
            <option value="zenmux">Zenmux</option>
          </select>
        </div>

        <div style={{ display: "flex", flexDirection: "column" }}>
          <label>Gemini API Key:</label>
          <input
            type="password"
            value={geminiApiKey}
            onChange={(e) => setGeminiApiKey(e.target.value)}
            placeholder="AI Studio Key"
            style={{ padding: "8px", borderRadius: "4px", background: "#333", color: "#fff", border: "1px solid #555" }}
          />
        </div>

        <div style={{ display: "flex", flexDirection: "column" }}>
          <label>OpenRouter API Key:</label>
          <input
            type="password"
            value={openRouterApiKey}
            onChange={(e) => setOpenRouterApiKey(e.target.value)}
            placeholder="sk-or-..."
            style={{ padding: "8px", borderRadius: "4px", background: "#333", color: "#fff", border: "1px solid #555" }}
          />
        </div>

        <div style={{ display: "flex", flexDirection: "column" }}>
          <label>Zenmux API Key:</label>
          <input
            type="password"
            value={zenmuxApiKey}
            onChange={(e) => setZenmuxApiKey(e.target.value)}
            placeholder="Zenmux Key"
            style={{ padding: "8px", borderRadius: "4px", background: "#333", color: "#fff", border: "1px solid #555" }}
          />
        </div>

        <div style={{ display: "flex", flexDirection: "column" }}>
          <label>Model Name:</label>
          <input
            type="text"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder="e.g. gemini-1.5-flash, gpt-4o, zenmux/z-ai/glm-4.6v"
            style={{ padding: "8px", borderRadius: "4px", background: "#333", color: "#fff", border: "1px solid #555" }}
          />
          <small style={{ color: "#aaa", marginTop: "4px" }}>
            Enter the model ID supported by your selected provider.
          </small>
        </div>

        <button 
            type="submit"
            style={{ padding: "10px", borderRadius: "4px", background: "#0078ff", color: "#fff", border: "none", cursor: "pointer", fontWeight: "bold" }}
        >
            Save Settings
        </button>
      </form>
    </div>
  );
}
