# Roadmap

## 1. Foundation
- Qwen via Ollama
- structured actions
- ADB executor
- screenshots
- OCR adapter
- verification-loop skeleton
- safety policy
- SQLite history

## 2. Accessibility
Build an Android AccessibilityService and normalize the UI tree.

## 3. Screen classifier
Add UI-element detection with normalized labels, bounding boxes and confidence.

## 4. Target resolver
Combine accessibility, OCR and vision matches. Prefer structured UI.

## 5. Recovery
After each action, compare expected vs observed state and retry/alternative/ask.

## 6. Voice
Whisper -> task -> Qwen, plus Android TTS.

## 7. VLA
Add VLA only when accessibility + classifier confidence is insufficient.
