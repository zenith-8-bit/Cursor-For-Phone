import pytest
from mobile_cursor.models import Action
from mobile_cursor.policy import validate_action, PolicyError

def test_click_requires_target():
    with pytest.raises(PolicyError):
        validate_action(Action(action="CLICK"))

def test_risky_requires_confirmation():
    with pytest.raises(PolicyError):
        validate_action(Action(action="SEND_MESSAGE", target_id="x", text="hi"))

def test_confirmed_risky_allowed():
    validate_action(
        Action(action="SEND_MESSAGE", target_id="x", text="hi", confirmed=True)
    )
