
import os
import requests
from .base import BaseModel


# Model configurations for mlapi.run
MLAPI_MODELS = {
    "gpt-oss-20b": {
        "uuid": "074881af-991b-4237-b58a-5e8a39b225f4",
        "api_model_name": "openai/gpt-oss-20b"
    },
    "gpt-5.2": {
        "uuid": "664ce153-d45c-42a7-903c-d9119cc55b69",
        "api_model_name": "openai/gpt-5.2"
    }
}


class MLApiModel(BaseModel):
    """Model wrapper for mlapi.run hosted models (gpt-oss-20b, GPT-5.2)"""

    def __init__(self, model_name: str, api_key: str = None):
        super().__init__(model_name)
        self.api_key = api_key or os.getenv("ELICE_API_KEY")
        if not self.api_key:
            raise ValueError("ELICE_API_KEY is not set.")

        # Get model config from mapping
        model_config = MLAPI_MODELS.get(model_name)
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MLAPI_MODELS.keys())}")

        self.api_url = f"https://mlapi.run/{model_config['uuid']}/v1/chat/completions"
        self.api_model_name = model_config['api_model_name']

    def generate(self, prompt: str, **kwargs) -> str:
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # OpenAI-compatible chat format
        payload = {
            "model": self.api_model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7)
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=600)
            response.raise_for_status()
            result = response.json()
            # Handle different response formats
            if "choices" in result:
                message = result["choices"][0]["message"]
                content = message.get("content")
                # Some models (like gpt-oss-20b) return content in reasoning_content
                if content is None:
                    content = message.get("reasoning_content", "")
                return content.strip() if content else ""
            elif "text" in result:
                return result["text"].strip()
            else:
                return str(result)
        except Exception as e:
            return f"Error: {e}"
