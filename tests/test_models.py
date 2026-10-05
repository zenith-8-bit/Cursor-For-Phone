from mobile_cursor.models import Action, ScreenState

def test_action():
    a = Action(action="HOME")
    assert a.action == "HOME"

def test_state():
    s = ScreenState()
    assert s.package == "unknown"
