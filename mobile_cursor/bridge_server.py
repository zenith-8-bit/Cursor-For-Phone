# Optional PC-side mock bridge for development without a phone.
from fastapi import FastAPI
from .models import ScreenState

app = FastAPI(title="Mobile Cursor Development Bridge")

@app.get("/health")
def health():
    return {"ok": True, "mode": "mock"}

@app.get("/state")
def state():
    return ScreenState(
        revision=1,
        package="mock",
        screen="APP",
        elements=[]
    ).model_dump()

@app.get("/xml")
def xml():
    return "<hierarchy package='mock'></hierarchy>"
