from mobile_cursor.config import Config

def test_config():
    cfg=Config()
    assert cfg.model
    cfg.prepare()
