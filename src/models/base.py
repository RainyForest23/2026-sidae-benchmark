
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_name: str, config: dict = None):
        self.model_name = model_name
        self.config = config or {}

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generates a response for the given prompt.
        
        Args:
            prompt: Input text
            kwargs: Additional generation parameters (max_tokens, temperature, etc.)
            
        Returns:
            Generated text
        """
        pass
