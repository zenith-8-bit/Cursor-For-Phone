import requests

class LocalSTT:
    """Adapter for a local speech-to-text HTTP service."""

    def __init__(self, url="http://127.0.0.1:8766/transcribe"):
        self.url = url

    def transcribe(self, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            r = requests.post(
                self.url,
                files={"audio": f},
                timeout=120,
            )
        r.raise_for_status()
        return r.json()["text"]

class LocalTTS:
    """Adapter for a local text-to-speech HTTP service."""

    def __init__(self, url="http://127.0.0.1:8766/speak"):
        self.url = url

    def speak(self, text: str) -> None:
        r = requests.post(self.url, json={"text": text}, timeout=120)
        r.raise_for_status()
