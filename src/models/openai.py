
import os
from openai import OpenAI
from .base import BaseModel

class OpenAIModel(BaseModel):
    def __init__(self, model_name: str, api_key: str = None):
        super().__init__(model_name)
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", 1024),
                temperature=kwargs.get("temperature", 0.7)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {e}"
