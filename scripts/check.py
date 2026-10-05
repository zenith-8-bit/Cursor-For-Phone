import importlib.util
import shutil
import requests

print("Python:", shutil.which("python"))
print("ADB:", shutil.which("adb"))
print("Ollama:", shutil.which("ollama"))

try:
    r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
    print("Ollama:", "OK", r.status_code)
except Exception as e:
    print("Ollama: unavailable:", e)

print("Python package check:")
for name in ["requests", "pydantic", "dotenv"]:
    print(" ", name, bool(importlib.util.find_spec(name)))
