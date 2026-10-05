from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Config:
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    bridge_url: str = os.getenv("PHONE_BRIDGE_URL", "http://127.0.0.1:8765")
    max_steps: int = int(os.getenv("MAX_STEPS", "9"))
    action_timeout: float = float(os.getenv("ACTION_TIMEOUT", "12"))
    state_timeout: float = float(os.getenv("STATE_TIMEOUT", "3"))
    memory_db: str = os.getenv("MEMORY_DB", "artifacts/mobile_cursor.sqlite3")
    require_confirmation: bool = os.getenv(
        "REQUIRE_CONFIRMATION_FOR_RISKY", "true"
    ).lower() == "true"

CONFIG = Config()
