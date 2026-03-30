"""
cursor_phone/executor.py
─────────────────────────
Translates Action objects → ADB shell commands.
Supports tap, long_tap, type, swipe, scroll, key events.
"""

import subprocess
import time
from dataclasses import dataclass
from typing import Optional

from context import Scene, Node
from llm     import Action


@dataclass
class ExecResult:
    success: bool
    message: str


# ── ADB primitive wrappers ─────────────────────────────────────

def _adb(*args, timeout: int = 10) -> tuple[bool, str]:
    cmd = ["adb", "shell"] + list(args)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except Exception as e:
        return False, str(e)


def _tap(x: int, y: int) -> ExecResult:
    ok, out = _adb("input", "tap", str(x), str(y))
    return ExecResult(ok, f"tap ({x},{y})" + (f" — {out}" if not ok else ""))


def _long_tap(x: int, y: int, ms: int = 800) -> ExecResult:
    ok, out = _adb("input", "swipe",
                   str(x), str(y), str(x), str(y), str(ms))
    return ExecResult(ok, f"long_tap ({x},{y})" + (f" — {out}" if not ok else ""))


def _type_text(text: str) -> ExecResult:
    # ADB input text doesn't handle spaces or special chars well.
    # We use a per-character approach and escape spaces as %s.
    # For Samsung Android 15, this is the most reliable path.
    escaped = text.replace(" ", "%s").replace("'", "\\'")
    ok, out = _adb("input", "text", escaped)
    return ExecResult(ok, f'type "{text}"' + (f" — {out}" if not ok else ""))


def _swipe_dir(direction: str, sx: int = 540, sy: int = 960) -> ExecResult:
    """Full-screen directional swipe."""
    W, H = 1080, 2400   # Samsung defaults; good enough for swipe direction
    coords = {
        "up":    (W//2, H*3//4, W//2, H//4),
        "down":  (W//2, H//4,   W//2, H*3//4),
        "left":  (W*3//4, H//2, W//4, H//2),
        "right": (W//4,   H//2, W*3//4, H//2),
    }
    if direction not in coords:
        return ExecResult(False, f"unknown direction: {direction}")
    x1, y1, x2, y2 = coords[direction]
    ok, out = _adb("input", "swipe", str(x1), str(y1), str(x2), str(y2), "300")
    return ExecResult(ok, f"swipe {direction}" + (f" — {out}" if not ok else ""))


def _scroll_on_node(node: Node, direction: str) -> ExecResult:
    cx, cy = node.center
    offsets = {
        "up":    (0, -300),
        "down":  (0, 300),
        "left":  (-300, 0),
        "right": (300, 0),
    }
    if direction not in offsets:
        return ExecResult(False, f"unknown scroll direction: {direction}")
    dx, dy = offsets[direction]
    ok, out = _adb("input", "swipe",
                   str(cx), str(cy),
                   str(cx + dx), str(cy + dy), "250")
    return ExecResult(ok, f"scroll {direction} at ({cx},{cy})" +
                         (f" — {out}" if not ok else ""))


def _key(keycode: str) -> ExecResult:
    ok, out = _adb("input", "keyevent", keycode)
    return ExecResult(ok, f"key {keycode}" + (f" — {out}" if not ok else ""))


# ── Node resolver ──────────────────────────────────────────────

def _resolve_node(target_str: Optional[str], scene: Scene) -> Optional[Node]:
    """Find a node by its index (as returned by the LLM)."""
    if target_str is None:
        return None
    try:
        idx = int(target_str)
        for n in scene.nodes:
            if n.index == idx:
                return n
    except ValueError:
        # Maybe it passed a resource-id string
        return scene.find_by_id(target_str)
    return None


# ── Executor ───────────────────────────────────────────────────

class Executor:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def execute(self, action: Action, scene: Scene) -> ExecResult:
        if self.dry_run:
            return ExecResult(True, f"[DRY RUN] would execute: {action}")

        atype = action.type

        # ── tap ───────────────────────────────────────────────
        if atype == "tap":
            node = _resolve_node(action.target, scene)
            if node:
                return _tap(*node.center)
            if action.x is not None and action.y is not None:
                return _tap(action.x, action.y)
            return ExecResult(False, "tap: no node or coordinates found")

        # ── long_tap ──────────────────────────────────────────
        if atype == "long_tap":
            node = _resolve_node(action.target, scene)
            if node:
                return _long_tap(*node.center)
            return ExecResult(False, "long_tap: node not found")

        # ── type ──────────────────────────────────────────────
        if atype == "type":
            node = _resolve_node(action.target, scene)
            if node and node.editable:
                # Focus the field first
                _tap(*node.center)
                time.sleep(0.4)
            elif node:
                _tap(*node.center)
                time.sleep(0.4)
            if action.text is None:
                return ExecResult(False, "type: no text provided")
            return _type_text(action.text)

        # ── swipe ─────────────────────────────────────────────
        if atype == "swipe":
            if action.direction is None:
                return ExecResult(False, "swipe: no direction provided")
            return _swipe_dir(action.direction)

        # ── scroll ────────────────────────────────────────────
        if atype == "scroll":
            node = _resolve_node(action.target, scene)
            if node and node.scrollable:
                return _scroll_on_node(node, action.direction or "down")
            # Fallback: full-screen swipe
            return _swipe_dir(action.direction or "down")

        # ── key ───────────────────────────────────────────────
        if atype == "key":
            kc = action.keycode or "KEYCODE_BACK"
            return _key(kc)

        # ── nothing ───────────────────────────────────────────
        if atype == "nothing":
            return ExecResult(True, f"no-op: {action.reason}")

        return ExecResult(False, f"unknown action type: {atype}")
