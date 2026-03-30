"""
cursor_phone/llm.py
────────────────────
Sends the scene summary + user intent to the Anthropic API.
Returns a structured Action the executor can run.

The model is prompted to respond ONLY with a JSON action block.
We parse that and hand it back to the orchestrator.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from context import Scene


# ── Action types ───────────────────────────────────────────────

@dataclass
class Action:
    type:    str          # tap | type | swipe | key | scroll | long_tap | nothing
    target:  Optional[str] = None   # node index (as str) or None
    x:       Optional[int] = None   # absolute coords (fallback)
    y:       Optional[int] = None
    text:    Optional[str] = None   # for type action
    direction: Optional[str] = None # for swipe/scroll: up|down|left|right
    keycode: Optional[str] = None   # for key action (KEYCODE_HOME etc.)
    reason:  Optional[str] = None   # LLM's reasoning

    def __str__(self):
        if self.type == "tap":
            tgt = f"node[{self.target}]" if self.target else f"({self.x},{self.y})"
            return f"tap {tgt}"
        if self.type == "type":
            return f'type "{self.text}" into node[{self.target}]'
        if self.type == "swipe":
            return f"swipe {self.direction}"
        if self.type == "scroll":
            return f"scroll {self.direction} on node[{self.target}]"
        if self.type == "key":
            return f"key {self.keycode}"
        if self.type == "long_tap":
            return f"long_tap node[{self.target}]"
        if self.type == "nothing":
            return f"nothing ({self.reason})"
        return f"{self.type}"


# ── System prompt ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are an Android UI automation agent.
You receive:
1. The current screen state as a list of interactive elements.
2. A user instruction describing what to do on screen.

Your job: decide the SINGLE best action to take right now.

Respond ONLY with a JSON object — no prose, no markdown fences.
Valid action types and their required fields:

  {"type":"tap",       "target":"<node_index>"}
  {"type":"tap",       "x":<int>, "y":<int>}          <- use if no node matches
  {"type":"long_tap",  "target":"<node_index>"}
  {"type":"type",      "target":"<node_index>", "text":"<string>"}
  {"type":"scroll",    "target":"<node_index>", "direction":"up|down|left|right"}
  {"type":"swipe",     "direction":"up|down|left|right"}
  {"type":"key",       "keycode":"KEYCODE_HOME|KEYCODE_BACK|KEYCODE_ENTER|..."}
  {"type":"nothing",   "reason":"<why no action is needed>"}

Rules:
- Choose the node whose label/id best matches the intent.
- For text fields, use "type" (always tap the field first if it's not focused).
- If the goal is already achieved, return nothing.
- If you must guess coordinates, use the center of the most relevant element.
- Always include an optional "reason" field with a one-line explanation.
"""


# ── Planner ────────────────────────────────────────────────────

class LLMPlanner:
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.model = model
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    def plan(self, intent: str, scene: Scene) -> Optional[Action]:
        prompt = (
            f"SCREEN STATE:\n{scene.summary()}\n\n"
            f"USER INSTRUCTION: {intent}\n\n"
            "What is the single best action to take right now?"
        )

        raw = self._call_api(prompt)
        if raw is None:
            return None

        return self._parse(raw)

    # ── API call ───────────────────────────────────────────────

    def _call_api(self, prompt: str) -> Optional[str]:
        """Call Anthropic /v1/messages. Works in Termux with requests installed."""
        try:
            import requests
        except ImportError:
            print("[llm] 'requests' not installed. Run: pip install requests")
            return None

        if not self._api_key:
            print("[llm] ANTHROPIC_API_KEY not set. Export it first.")
            return None

        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key":         self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json={
                    "model":      self.model,
                    "max_tokens": 256,
                    "system":     SYSTEM_PROMPT,
                    "messages":   [{"role": "user", "content": prompt}],
                },
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"].strip()
        except Exception as e:
            print(f"[llm] API error: {e}")
            return None

    # ── Response parser ────────────────────────────────────────

    def _parse(self, raw: str) -> Optional[Action]:
        # Strip markdown fences if model forgets
        raw = re.sub(r"```[a-z]*", "", raw).strip("`").strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract first {...} block
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                print(f"[llm] Could not parse response: {raw!r}")
                return None
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                print(f"[llm] JSON parse failed on: {m.group(0)!r}")
                return None

        atype = obj.get("type", "nothing")
        return Action(
            type      = atype,
            target    = str(obj["target"]) if "target" in obj else None,
            x         = obj.get("x"),
            y         = obj.get("y"),
            text      = obj.get("text"),
            direction = obj.get("direction"),
            keycode   = obj.get("keycode"),
            reason    = obj.get("reason"),
        )
