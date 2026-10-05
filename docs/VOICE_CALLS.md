# Voice and phone-call extension

The architecture is:

```text
Telecom / InCallService
        ↓
call state
        ↓
local STT
        ↓
Qwen
        ↓
local TTS
        ↓
device call-audio path
```

The repository provides the Python controller and Android InCallService skeleton.

Android call handling is intentionally not presented as universally available.
Depending on the Android version/device, the application may need to become
the default phone app and receive Telecom permissions.

Speech-to-text and text-to-speech should run locally where possible. The Python
adapters expect HTTP endpoints at localhost:

```text
POST /transcribe
POST /speak
```

The audio transport into an active cellular call remains a device-specific
Android integration point.
