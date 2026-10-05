from .models import Action

RISKY_ACTIONS = {"SEND_MESSAGE","CALL","DELETE","PURCHASE","POST"}

class ActionValidator:
    def validate(self, action: Action):
        if action.name in {"TAP","LONG_PRESS"}:
            if action.target is None and (action.x is None or action.y is None):
                return False, f"{action.name} requires target or coordinates"

        if action.name == "SWIPE":
            if action.direction:
                return True, ""
            if None in (action.x, action.y, action.x2, action.y2):
                return False, "SWIPE requires direction or four coordinates"

        if action.name == "TYPE_TEXT" and action.text is None:
            return False, "TYPE_TEXT requires text"

        if action.name == "OPEN_APP" and not action.package:
            return False, "OPEN_APP requires package"

        if action.name == "CLICK_ELEMENT" and not action.target:
            return False, "CLICK_ELEMENT requires target"

        if action.name == "SCROLL" and action.direction not in {"UP","DOWN","LEFT","RIGHT"}:
            return False, "SCROLL requires UP/DOWN/LEFT/RIGHT"

        return True, ""

    def is_risky(self, action):
        return action.name in RISKY_ACTIONS
