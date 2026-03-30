"""
cursor_phone/context.py
───────────────────────
Captures the current screen state via UIAutomator XML dump (ADB),
parses all interactive nodes, and builds a plain-text scene graph
the LLM can reason about.
"""

import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Node:
    index:        int
    node_id:      str          # resource-id or generated
    cls:          str          # android widget class
    text:         str
    content_desc: str
    clickable:    bool
    long_clickable: bool
    scrollable:   bool
    editable:     bool
    enabled:      bool
    checked:      bool
    bounds:       tuple        # (x1,y1,x2,y2)
    center:       tuple        # (cx,cy)
    children:     list = field(default_factory=list)

    def label(self) -> str:
        """Best human-readable label for this node."""
        return (self.text or self.content_desc or
                self.node_id.split("/")[-1] or self.cls.split(".")[-1])

    def is_interactive(self) -> bool:
        return (self.clickable or self.long_clickable or
                self.scrollable or self.editable) and self.enabled


@dataclass
class Scene:
    package:    str
    activity:   str
    nodes:      list[Node]
    raw_xml:    str
    timestamp:  float

    def interactables(self) -> list[Node]:
        return [n for n in self.nodes if n.is_interactive()]

    def find_by_text(self, text: str, fuzzy: bool = True) -> Optional[Node]:
        text_l = text.lower()
        for n in self.nodes:
            lbl = n.label().lower()
            if fuzzy:
                if text_l in lbl or lbl in text_l:
                    return n
            else:
                if lbl == text_l:
                    return n
        return None

    def find_by_id(self, res_id: str) -> Optional[Node]:
        for n in self.nodes:
            if res_id in n.node_id:
                return n
        return None

    def summary(self) -> str:
        """Compact text summary for the LLM prompt."""
        lines = [
            f"APP: {self.package}",
            f"ACTIVITY: {self.activity}",
            f"INTERACTIVE ELEMENTS ({len(self.interactables())}):",
        ]
        for n in self.interactables():
            actions = []
            if n.clickable:      actions.append("click")
            if n.long_clickable: actions.append("long_click")
            if n.scrollable:     actions.append("scroll")
            if n.editable:       actions.append("type")
            lines.append(
                f"  [{n.index}] {n.cls.split('.')[-1]}"
                f" | label={repr(n.label())}"
                f" | id={n.node_id}"
                f" | actions={actions}"
                f" | bounds={n.bounds}"
                f" | center={n.center}"
                + (f" | checked={n.checked}" if n.checked else "")
            )
        return "\n".join(lines)


# ── ADB helpers ────────────────────────────────────────────────

_ADB_DUMP_CMD  = ["adb", "shell", "uiautomator", "dump", "/sdcard/ui.xml"]
_ADB_PULL_CMD  = ["adb", "pull", "/sdcard/ui.xml", "/tmp/ui.xml"]
_ADB_FOCUS_CMD = ["adb", "shell",
                  "dumpsys", "window", "windows"]

def _run(cmd: list[str], timeout: int = 10) -> tuple[bool, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, r.stdout + r.stderr
    except Exception as e:
        return False, str(e)


def _get_foreground_app() -> tuple[str, str]:
    ok, out = _run(_ADB_FOCUS_CMD)
    if not ok:
        return "unknown", "unknown"
    # Parse "mCurrentFocus=Window{... pkg/activity}"
    m = re.search(r"mCurrentFocus=Window\{[^}]+\s+(\S+)/(\S+)\}", out)
    if m:
        return m.group(1), m.group(2)
    # Fallback: mFocusedApp
    m = re.search(r"mFocusedApp.*?(\S+)/(\S+)\}", out)
    if m:
        return m.group(1), m.group(2)
    return "unknown", "unknown"


# ── XML parser ─────────────────────────────────────────────────

def _parse_bounds(bounds_str: str) -> tuple:
    """'[x1,y1][x2,y2]' → (x1,y1,x2,y2)"""
    nums = list(map(int, re.findall(r"\d+", bounds_str)))
    if len(nums) == 4:
        return tuple(nums)
    return (0, 0, 0, 0)


def _center(bounds: tuple) -> tuple:
    x1, y1, x2, y2 = bounds
    return ((x1 + x2) // 2, (y1 + y2) // 2)


_counter = 0

def _parse_node(elem, nodes: list, depth: int = 0) -> Optional[Node]:
    global _counter
    _counter += 1
    idx = _counter

    b     = _parse_bounds(elem.get("bounds", "[0,0][0,0]"))
    node  = Node(
        index        = idx,
        node_id      = elem.get("resource-id", ""),
        cls          = elem.get("class", ""),
        text         = elem.get("text", ""),
        content_desc = elem.get("content-desc", ""),
        clickable    = elem.get("clickable") == "true",
        long_clickable = elem.get("long-clickable") == "true",
        scrollable   = elem.get("scrollable") == "true",
        editable     = elem.get("class", "").endswith("EditText"),
        enabled      = elem.get("enabled", "true") == "true",
        checked      = elem.get("checked") == "true",
        bounds       = b,
        center       = _center(b),
    )
    nodes.append(node)
    for child in elem:
        child_node = _parse_node(child, nodes, depth + 1)
        if child_node:
            node.children.append(child_node)
    return node


# ── Public API ─────────────────────────────────────────────────

class ScreenContext:
    def __init__(self):
        self._last_scene: Optional[Scene] = None

    def capture(self) -> Optional[Scene]:
        import time

        # 1. Dump UI XML to device sdcard
        ok, out = _run(_ADB_DUMP_CMD)
        if not ok or "error" in out.lower():
            return None

        # 2. Pull to local /tmp
        ok, out = _run(_ADB_PULL_CMD)
        if not ok:
            return None

        # 3. Parse XML
        try:
            raw = open("/tmp/ui.xml", "r", encoding="utf-8", errors="replace").read()
        except FileNotFoundError:
            return None

        global _counter
        _counter = 0
        nodes: list[Node] = []
        try:
            root = ET.fromstring(raw)
            _parse_node(root, nodes)
        except ET.ParseError:
            return None

        pkg, act = _get_foreground_app()
        scene = Scene(
            package=pkg, activity=act,
            nodes=nodes, raw_xml=raw,
            timestamp=time.time()
        )
        self._last_scene = scene
        return scene

    @property
    def last(self) -> Optional[Scene]:
        return self._last_scene
