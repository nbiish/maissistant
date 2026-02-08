import os
import base64
from io import BytesIO
from typing import Optional
from pocket_tts import TTSModel
import torch
import soundfile as sf

class TTSManager:
    def __init__(self):
        self.model: Optional[TTSModel] = None
        self.voice_state = None
        self.model_loaded = False

    def load_model(self):
        if not self.model_loaded:
            print("Loading PocketTTS model...")
            self.model = TTSModel.load_model()
            # Default voice
            self.voice_state = self.model.get_state_for_audio_prompt(
                "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
            )
            self.model_loaded = True
            print("PocketTTS model loaded.")

    def generate_speech(self, text: str) -> str:
        if not self.model_loaded:
            self.load_model()
        
        if not self.model or not self.voice_state:
            return ""

        print(f"Generating speech for: {text[:50]}...")
        audio = self.model.generate_audio(self.voice_state, text)
        
        # Audio is a 1D torch tensor. Convert to bytes (WAV)
        # Using soundfile or scipy
        buffer = BytesIO()
        sf.write(buffer, audio.numpy(), self.model.sample_rate, format='WAV')
        buffer.seek(0)
        
        # Encode to base64
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return audio_base64
