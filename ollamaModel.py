import requests
from typing import Optional

class OllamaLLM:
    def __init__(self, endpoint: str, model: str = "llama3:8B", temperature: float = 0.2, max_tokens: int = 1024):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": self.temperature,
            "num_predict": self.max_tokens
        }
        try:
            response = requests.post(f"{self.endpoint}/api/generate", json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            return f"Error: {e}"
