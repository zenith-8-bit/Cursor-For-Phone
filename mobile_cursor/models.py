from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field

ActionName = Literal[
    "HOME", "BACK", "RECENTS", "WAIT",
    "OPEN_APP", "CLICK", "LONG_PRESS",
    "TYPE", "CLEAR", "SCROLL",
    "SEND_MESSAGE", "CALL", "ANSWER_CALL", "HANGUP",
    "SPEAK", "STOP", "ASK_USER"
]

class Bounds(BaseModel):
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0

class UIElement(BaseModel):
    id: str
    class_name: str = ""
    text: str = ""
    content_description: str = ""
    resource_id: str = ""
    package: str = ""
    clickable: bool = False
    long_clickable: bool = False
    editable: bool = False
    scrollable: bool = False
    enabled: bool = True
    focused: bool = False
    selected: bool = False
    checked: bool = False
    visible: bool = True
    bounds: Bounds = Field(default_factory=Bounds)
    depth: int = 0
    parent_id: str | None = None

class ScreenState(BaseModel):
    revision: int = 0
    package: str = "unknown"
    activity: str = ""
    screen: str = "APP"
    timestamp_ms: int = 0
    elements: list[UIElement] = Field(default_factory=list)

class Action(BaseModel):
    action: ActionName
    target_id: str | None = None
    text: str | None = None
    app: str | None = None
    direction: Literal["up", "down", "left", "right"] | None = None
    seconds: float | None = None
    confirmed: bool = False
    reason: str = ""

class Plan(BaseModel):
    status: Literal["continue", "done", "needs_user", "failed"] = "continue"
    message: str = ""
    actions: list[Action] = Field(default_factory=list)

class StepResult(BaseModel):
    ok: bool
    message: str
    revision_before: int = 0
    revision_after: int = 0
