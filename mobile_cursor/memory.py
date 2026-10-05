import json
import sqlite3
from pathlib import Path
from .config import CONFIG

class Memory:
    def __init__(self, path: str | None = None):
        self.path = Path(path or CONFIG.memory_db)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("""
        CREATE TABLE IF NOT EXISTS user_info (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """)
        self.db.execute("""
        CREATE TABLE IF NOT EXISTS action_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL DEFAULT (strftime('%s','now')),
            action TEXT NOT NULL,
            result TEXT NOT NULL
        )
        """)
        self.db.commit()

    def all_user_info(self) -> dict:
        rows = self.db.execute("SELECT key,value FROM user_info").fetchall()
        out = {}
        for k, v in rows:
            try:
                out[k] = json.loads(v)
            except Exception:
                out[k] = v
        return out

    def set_user_info(self, key: str, value):
        self.db.execute(
            "INSERT OR REPLACE INTO user_info(key,value) VALUES (?,?)",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        self.db.commit()

    def log(self, action: str, result: str):
        self.db.execute(
            "INSERT INTO action_log(action,result) VALUES (?,?)",
            (action, result),
        )
        self.db.commit()
