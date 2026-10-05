import sqlite3
from pathlib import Path

class MemoryStore:
    def __init__(self, path="artifacts/agent.db"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS events "
            "(id INTEGER PRIMARY KEY, ts TEXT, task TEXT, action TEXT, result TEXT)"
        )
        self.db.commit()

    def add(self, ts, task, action, result):
        self.db.execute(
            "INSERT INTO events(ts,task,action,result) VALUES(?,?,?,?)",
            (ts,task,action,result)
        )
        self.db.commit()

    def recent(self, task, limit=8):
        rows = self.db.execute(
            "SELECT action,result FROM events WHERE task=? ORDER BY id DESC LIMIT ?",
            (task,limit)
        ).fetchall()
        return [f"{a}: {r}" for a,r in reversed(rows)]
