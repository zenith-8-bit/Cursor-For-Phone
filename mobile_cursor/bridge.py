import time
import requests
from .config import CONFIG
from .models import Action, ScreenState, StepResult

class PhoneBridge:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or CONFIG.bridge_url).rstrip("/")

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=CONFIG.state_timeout)
        r.raise_for_status()
        return r.json()

    def state(self) -> ScreenState:
        r = requests.get(f"{self.base_url}/state", timeout=CONFIG.state_timeout)
        r.raise_for_status()
        return ScreenState.model_validate(r.json())

    def xml(self) -> str:
        r = requests.get(f"{self.base_url}/xml", timeout=CONFIG.state_timeout)
        r.raise_for_status()
        return r.text

    def execute(self, action: Action) -> StepResult:
        before = self.state()
        r = requests.post(
            f"{self.base_url}/action",
            json=action.model_dump(),
            timeout=CONFIG.action_timeout,
        )
        r.raise_for_status()
        result = r.json()
        return StepResult(
            ok=bool(result.get("ok")),
            message=str(result.get("message", "")),
            revision_before=before.revision,
            revision_after=int(result.get("revision", before.revision)),
        )

    def wait_for_revision(self, old_revision: int, timeout: float = 8) -> ScreenState:
        end = time.time() + timeout
        last = self.state()
        while time.time() < end:
            if last.revision != old_revision:
                return last
            time.sleep(0.25)
            last = self.state()
        return last
