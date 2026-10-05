import json
from .bridge import PhoneBridge
from .config import CONFIG
from .memory import Memory
from .models import Action
from .planner import Planner
from .policy import validate_action, PolicyError

class Agent:
    def __init__(self, bridge=None, planner=None, memory=None):
        self.bridge = bridge or PhoneBridge()
        self.planner = planner or Planner()
        self.memory = memory or Memory()

    def _display_state(self, state):
        print(f"\n--- revision={state.revision} package={state.package} "
              f"screen={state.screen} activity={state.activity} ---")
        for e in state.elements:
            label = e.text or e.content_description or e.resource_id
            if not label and not (e.clickable or e.editable or e.scrollable):
                continue
            flags = []
            if e.clickable: flags.append("click")
            if e.editable: flags.append("edit")
            if e.scrollable: flags.append("scroll")
            if e.checked: flags.append("checked")
            print(f"{e.id} | {e.class_name.split('.')[-1]} | "
                  f"{label!r} | {','.join(flags)} | {e.bounds.model_dump()}")

    def run(self, task: str, dry_run=False, visualize=False, max_steps=None):
        max_steps = max_steps or CONFIG.max_steps
        history = []
        context_task = task

        for step in range(max_steps):
            state = self.bridge.state()
            if visualize:
                self._display_state(state)

            plan = self.planner.plan(
                context_task, state, self.memory.all_user_info(), history
            )
            print(f"\n[{step+1}] Qwen: {plan.status} | {plan.message}")

            if plan.status == "done":
                return True, plan.message

            if plan.status in {"failed", "needs_user"}:
                question = plan.message
                if plan.status == "needs_user":
                    answer = input(f"Agent asks: {question}\nYou: ").strip()
                    if not answer:
                        return False, "No answer supplied."
                    context_task += f"\nUser answered the missing-information question: {answer}"
                    continue
                return False, plan.message

            if not plan.actions:
                return False, "Qwen returned no action."

            action = plan.actions[0]
            print(f"    action: {json.dumps(action.model_dump(), ensure_ascii=False)}")

            try:
                validate_action(action, CONFIG.require_confirmation)
            except PolicyError as exc:
                print(f"    policy: {exc}")
                if action.action in {"SEND_MESSAGE", "CALL", "ANSWER_CALL", "HANGUP"}:
                    answer = input("Allow this consequential action? [y/N] ").strip().lower()
                    if answer == "y":
                        action.confirmed = True
                        validate_action(action, CONFIG.require_confirmation)
                    else:
                        return False, "User denied consequential action."

            if dry_run:
                history.append({"action": action.model_dump(), "result": "dry-run"})
                self.memory.log(action.action, "dry-run")
                continue

            before = state.revision
            result = self.bridge.execute(action)
            print(f"    executor: {'OK' if result.ok else 'FAIL'} | {result.message}")
            self.memory.log(action.action, result.message)

            if not result.ok:
                history.append({"action": action.model_dump(), "result": result.message})
                continue

            after = self.bridge.wait_for_revision(before)
            history.append({
                "action": action.model_dump(),
                "result": result.message,
                "revision": after.revision,
            })

            if after.revision == before and action.action not in {"WAIT"}:
                # Prevent infinite loops when an app ignored an action.
                history.append({
                    "warning": "screen_revision_unchanged",
                    "action": action.action,
                })

        return False, f"Stopped after {max_steps} steps."
