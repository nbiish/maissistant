import os
import base64
from io import BytesIO
from typing import Optional, Dict, Any
import google.generativeai as genai
from openai import OpenAI
from PIL import Image

class ModelManager:
    def __init__(self):
        pass

    def _decode_image(self, image_base64: str) -> Image.Image:
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        image_data = base64.b64decode(image_base64)
        return Image.open(BytesIO(image_data))

    async def generate_response(
        self, 
        message: str, 
        provider: str, 
        model_name: str, 
        api_key: str, 
        image_base64: Optional[str] = None
    ) -> str:
        
        if not api_key:
            return "Error: API Key is required."

        try:
            if provider.lower() == "gemini":
                return await self._call_gemini(message, model_name, api_key, image_base64)
            elif provider.lower() == "openrouter":
                return await self._call_openai_compatible(
                    message, 
                    model_name, 
                    api_key, 
                    "https://openrouter.ai/api/v1", 
                    image_base64
                )
            elif provider.lower() == "zenmux":
                 return await self._call_openai_compatible(
                    message, 
                    model_name, 
                    api_key, 
                    "https://zenmux.ai/api/v1",  # Check exact base URL
                    image_base64
                )
            else:
                return f"Error: Unknown provider '{provider}'"
        except Exception as e:
            print(f"Error calling {provider}: {e}")
            return f"Error calling {provider}: {str(e)}"

    async def _call_gemini(self, message: str, model_name: str, api_key: str, image_base64: Optional[str]) -> str:
        genai.configure(api_key=api_key)
        # Default to a vision model if image is present, though newer Gemini models are multimodal
        if not model_name:
            model_name = "gemini-1.5-flash" 
        
        model = genai.GenerativeModel(model_name)
        
        content = [message]
        if image_base64:
            image = self._decode_image(image_base64)
            content.append(image)
            
        response = model.generate_content(content)
        return response.text

    async def _call_openai_compatible(
        self, 
        message: str, 
        model_name: str, 
        api_key: str, 
        base_url: str, 
        image_base64: Optional[str]
    ) -> str:
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        messages = []
        if image_base64:
            # Format for vision models (GPT-4o, Claude 3 via OpenRouter, etc.)
            # Note: Not all OpenRouter models support this format, but most vision ones do.
            if "," in image_base64:
                # Ensure data URI scheme
                url = image_base64
            else:
                url = f"data:image/jpeg;base64,{image_base64}"

            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": message},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": url
                        }
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": message})

        if not model_name:
            # Default fallback
            model_name = "openai/gpt-4o-mini" if "openrouter" in base_url else "zenmux/z-ai/glm-4.6v-flash"

        response = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        return response.choices[0].message.content
