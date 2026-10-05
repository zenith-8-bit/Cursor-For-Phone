import json
from .models import ScreenState, Plan
from .ollama import OllamaClient
from .prompt import SYSTEM_PROMPT

class Planner:
    def __init__(self, ollama: OllamaClient | None = None):
        self.ollama = ollama or OllamaClient()

    def plan(self, task: str, state: ScreenState, memory: dict, history: list[dict]) -> Plan:
        user = {
            "task": task,
            "current_screen_state": state.model_dump(),
            "known_user_information": memory,
            "recent_action_history": history[-8:],
        }
        raw = self.ollama.chat_json(SYSTEM_PROMPT, json.dumps(user, ensure_ascii=False))
        return Plan.model_validate(raw)
