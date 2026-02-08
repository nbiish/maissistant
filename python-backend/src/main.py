from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from typing import Optional
from src.agent_brain import AgentBrain
from src.tts_manager import TTSManager
import shutil
import os

app = FastAPI(title="MAIssistant Backend")

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_brain = AgentBrain()
tts_manager = TTSManager()

class ChatRequest(BaseModel):
    message: str
    provider: str
    api_key: str
    fallback_key: Optional[str] = None
    model: Optional[str] = None
    image: Optional[str] = None # Base64 encoded image
    session_id: Optional[str] = None

class SpeakRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"status": "running", "service": "MAIssistant Backend"}

@app.post("/chat")
async def chat(request: ChatRequest):
    print(f"Received chat request for provider: {request.provider}, model: {request.model}, session: {request.session_id}")
    
    if not request.api_key:
        raise HTTPException(status_code=400, detail="API Key is required")

    session_id = request.session_id or "default_session"

    try:
        response = await agent_brain.chat(
            message=request.message,
            session_id=session_id,
            provider=request.provider,
            api_key=request.api_key,
            model_name=request.model,
            image_base64=request.image,
            fallback_key=request.fallback_key
        )
        return {"response": response}
    except Exception as e:
        print(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Placeholder for STT
    print(f"Received audio file for transcription: {file.filename}")
    
    # Save temp file
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Mock transcription
    transcription = "This is a simulated transcription of the audio."
    
    # Cleanup
    os.remove(temp_filename)
    
    return {"text": transcription}

@app.post("/speak")
async def speak(request: SpeakRequest):
    print(f"Received text for speech: {request.text}")
    
    try:
        audio_base64 = tts_manager.generate_speech(request.text)
        return {"status": "success", "audio": audio_base64}
    except Exception as e:
        print(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
