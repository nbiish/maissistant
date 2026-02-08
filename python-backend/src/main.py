from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from src.model_manager import ModelManager

app = FastAPI(title="MAIssistant Backend")

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = ModelManager()

class ChatRequest(BaseModel):
    message: str
    provider: str
    api_key: str
    model: Optional[str] = None
    image: Optional[str] = None # Base64 encoded image
    session_id: Optional[str] = None

@app.get("/")
async def root():
    return {"status": "running", "service": "MAIssistant Backend"}

@app.post("/chat")
async def chat(request: ChatRequest):
    print(f"Received chat request for provider: {request.provider}, model: {request.model}")
    
    if not request.api_key:
        raise HTTPException(status_code=400, detail="API Key is required")

    response = await model_manager.generate_response(
        message=request.message,
        provider=request.provider,
        model_name=request.model,
        api_key=request.api_key,
        image_base64=request.image
    )
    
    return {"response": response}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
