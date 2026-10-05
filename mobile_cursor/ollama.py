import requests
from .config import CONFIG

class OllamaClient:
    def __init__(self, url: str | None = None, model: str | None = None):
        self.url = (url or CONFIG.ollama_url).rstrip("/")
        self.model = model or CONFIG.ollama_model

    def chat_json(self, system: str, user: str) -> dict:
        r = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        import json
        return json.loads(data["message"]["content"])
