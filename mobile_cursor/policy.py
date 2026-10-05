from .models import Action

RISKY = {"SEND_MESSAGE", "CALL", "ANSWER_CALL", "HANGUP"}

class PolicyError(Exception):
    pass

def validate_action(action: Action, require_confirmation: bool = True) -> None:
    if action.action in RISKY and require_confirmation and not action.confirmed:
        raise PolicyError(
            f"{action.action} requires explicit confirmation. "
            "Set confirmed=true only after the user/application has confirmed it."
        )

    if action.action in {"CLICK", "LONG_PRESS", "TYPE", "CLEAR"} and not action.target_id:
        raise PolicyError(f"{action.action} requires target_id")

    if action.action == "TYPE" and action.text is None:
        raise PolicyError("TYPE requires text")

    if action.action == "OPEN_APP" and not action.app:
        raise PolicyError("OPEN_APP requires app")

    if action.action == "SCROLL" and action.direction not in {"up", "down", "left", "right"}:
        raise PolicyError("SCROLL requires a direction")

    if action.action == "WAIT":
        seconds = action.seconds if action.seconds is not None else 1
        if not 0 <= seconds <= 10:
            raise PolicyError("WAIT is limited to 0..10 seconds")
