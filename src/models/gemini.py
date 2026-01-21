
import os
import google.generativeai as genai
from .base import BaseModel

class GeminiModel(BaseModel):
    def __init__(self, model_name: str, api_key: str = None):
        super().__init__(model_name)
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            # Gemini generation config
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=kwargs.get("max_tokens", 1024),
                temperature=kwargs.get("temperature", 0.7)
            )
            response = self.model.generate_content(prompt, generation_config=generation_config)
            return response.text.strip()
        except Exception as e:
            return f"Error: {e}"
