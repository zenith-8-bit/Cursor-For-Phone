# Mobile Cursor — Qwen 2.5 + Android Accessibility Groundbase

A from-scratch groundbase for a "Cursor for Phone" agent.

## Core design

```text
Android AccessibilityService
        │
        │ semantic UI tree / XML-like state
        ▼
localhost bridge (adb reverse)
        │
        ▼
Python controller
        │
        ├── task + user memory
        ├── policy / action validator
        ▼
Ollama → Qwen 2.5 7B
        │
        ▼
bounded JSON action
        │
        ▼
Android AccessibilityService
        │
        ▼
new UI state / revision
```

There is deliberately **no OCR and no screenshot reasoning in the normal control loop**.

Qwen decides **what** should happen. Android Accessibility decides **how** to perform it. The validator decides **whether** the requested action is allowed.

## What this build includes

- Android AccessibilityService with live `AccessibilityNodeInfo` tree inspection.
- Stable per-screen semantic element IDs.
- `/state`, `/xml`, `/health`, and `/action` local endpoints.
- Real two-way Android action transport using an embedded Android `ServerSocket` plus `adb reverse`.
- Click, long press, type, clear, scroll, home, back, recents, wait, open-app.
- Qwen 2.5 7B through Ollama.
- Strict JSON action format.
- Action validation and risky-action confirmation gates.
- Interactive missing-information questions.
- SQLite user-memory store.
- Terminal screen-state visualization.
- Loop protection against repeating an action on an unchanged screen.
- Optional localhost STT/TTS interfaces for a future voice/call controller.
- InCallService skeleton with explicit Android/default-phone-app limitations documented.

## Important limitation

Accessibility-only control works very well for many native Android interfaces, but it cannot guarantee complete coverage of every app. Some apps use custom rendering, protected surfaces, games, or accessibility-hostile views.

This build intentionally does not add OCR/VLA fallback yet. That can be layered on later without changing the semantic planner.

## Requirements

### PC

- Windows 10/11
- Python 3.11+ (3.13 should work)
- Ollama
- Qwen 2.5 7B model
- Android SDK / platform tools
- `adb`
- Android Studio for building the Android project

### Phone

- Android device with USB debugging enabled
- Accessibility permission granted to Mobile Cursor
- For call automation: additional Telecom/default-phone-app permissions may be required

## 1. Install Python dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Prepare Ollama

```powershell
ollama pull qwen2.5:7b
ollama serve
```

If Ollama is already running, do not start a second server.

## 3. Build Android side

Open the `android` folder in Android Studio.

Build and install the `app` module on the phone.

After installation:

1. Open **Settings → Accessibility**.
2. Enable **Mobile Cursor**.
3. Return to the app and check the bridge status.
4. Connect the phone over USB.

Then run:

```powershell
adb devices
adb reverse tcp:8765 tcp:8765
```

The Android AccessibilityService listens on device localhost port `8765`. `adb reverse` maps PC `127.0.0.1:8765` to it.

## 4. Test the raw screen state

```powershell
python -m mobile_cursor.cli state
```

Pretty state viewer:

```powershell
python -m mobile_cursor.cli watch
```

Raw hierarchy XML:

```powershell
python -m mobile_cursor.cli xml
```

## 5. Dry-run Qwen

```powershell
python -m mobile_cursor.cli run --task "Open the home screen" --dry-run
```

Dry-run asks Qwen what it would do but does not send the action.

## 6. Live task

```powershell
python -m mobile_cursor.cli run --task "Open Amazon and search for black hoodies"
```

The controller executes one bounded action at a time and waits for the Android UI revision to change.

## 7. Interactive missing information

If Qwen decides that a required value is missing, the controller prints a question and waits for your answer.

Example:

```text
Agent: What size should I search for?
You: medium
```

The answer becomes part of the task context and the loop continues.

## Example tasks

```text
Open WhatsApp and open the chat with Alice.
```

```text
Open Amazon and search for black hoodies.
```

```text
Open Chrome and search for the latest Android AccessibilityService documentation.
```

```text
Open Settings and find the Accessibility settings.
```

For actions with real-world consequences, the default policy requires confirmation.

## Risk policy

The default validator blocks:

- sending messages
- placing calls
- disconnecting calls
- purchases
- account/security changes

until explicitly confirmed by the user/application policy.

This is intentional. The planner is not allowed to bypass this by emitting arbitrary shell, ADB, Python, JavaScript, or OS commands.

## Voice / phone calls

The Python side includes interfaces for localhost STT/TTS and a call controller.

The Android side includes an `InCallService` skeleton.

Actual phone-call audio capture/injection depends on Android version, device, Telecom role, permissions, and whether the application is the default phone app. The project therefore separates:

1. call state/control,
2. local speech-to-text,
3. Qwen conversation logic,
4. local text-to-speech,
5. device call-audio routing.

Do not treat the call module as a guaranteed bypass of Android Telecom restrictions.

## Files

```text
mobile_cursor_full/
├── mobile_cursor/
│   ├── agent.py
│   ├── bridge.py
│   ├── bridge_server.py
│   ├── calls.py
│   ├── cli.py
│   ├── config.py
│   ├── memory.py
│   ├── models.py
│   ├── ollama.py
│   ├── planner.py
│   ├── policy.py
│   ├── prompt.py
│   └── voice.py
├── android/
│   └── app/src/main/java/com/mobilecursor/
├── docs/
└── tests/
```
