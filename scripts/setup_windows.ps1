$ErrorActionPreference = "Stop"

python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host ""
Write-Host "Environment created."
Write-Host "Next:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  ollama pull qwen2.5:7b"
Write-Host "  adb devices"
Write-Host "  adb reverse tcp:8765 tcp:8765"
