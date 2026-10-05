from mobile_cursor.models import Action
from mobile_cursor.validator import ActionValidator

def test_tap_needs_target_or_coordinates():
    ok,_=ActionValidator().validate(Action(name="TAP"))
    assert not ok

def test_tap_with_coordinates():
    ok,_=ActionValidator().validate(Action(name="TAP",x=100,y=200))
    assert ok

def test_risky():
    v=ActionValidator()
    assert v.is_risky(Action(name="SEND_MESSAGE"))
    assert not v.is_risky(Action(name="BACK"))
