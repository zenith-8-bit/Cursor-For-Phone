from fastapi import FastAPI
from .config import Config

app=FastAPI(title="Mobile Cursor Foundation")
config=Config()

@app.get("/")
def root():
    return {"name":"Mobile Cursor Foundation","model":config.model,"status":"ready"}

@app.get("/health")
def health():
    return {"ok":True}
