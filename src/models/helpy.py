
import os
import requests
from .base import BaseModel


class HelpyProModel(BaseModel):
    """Helpy Pro Dragon model via mlapi.run"""

    def __init__(self, model_name: str = "helpy-pro", api_key: str = None):
        super().__init__(model_name)
        self.api_key = api_key or os.getenv("ELICE_API_KEY")
        if not self.api_key:
            raise ValueError("ELICE_API_KEY is not set.")
        self.api_url = "https://mlapi.run/5ee9c080-1fdd-401e-9830-1d2733a45b25/v1/chat/completions"
        self.api_model_name = "eliceai/helpy-pro-dragon"

    def generate(self, prompt: str, **kwargs) -> str:
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.api_model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "chat_template_kwargs": {
                "enable_thinking": kwargs.get("enable_thinking", False)
            },
            "max_completion_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7)
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=600)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Error: {e}"


class HelpyEduModel(BaseModel):
    """Helpy Edu DragonFruit model via mlapi.run"""

    def __init__(self, model_name: str = "helpy-edu", api_key: str = None):
        super().__init__(model_name)
        self.api_key = api_key or os.getenv("ELICE_API_KEY")
        if not self.api_key:
            raise ValueError("ELICE_API_KEY is not set.")
        self.api_url = "https://mlapi.run/4efc840a-a50b-46ca-b2d5-6eb7ead6aa37/v1/chat/completions"
        self.api_model_name = "eliceai/helpy-edu-dragonfruit"

    def generate(self, prompt: str, **kwargs) -> str:
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.api_model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "chat_template_kwargs": {
                "enable_thinking": kwargs.get("enable_thinking", False)
            },
            "max_completion_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7)
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=600)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Error: {e}"
