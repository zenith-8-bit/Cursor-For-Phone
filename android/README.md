# Android side

Open this folder in Android Studio.

After installation:

1. Enable **Mobile Cursor** under Accessibility.
2. Connect the phone by USB.
3. On the PC run:

```powershell
adb devices
adb reverse tcp:8765 tcp:8765
```

The AccessibilityService starts a tiny HTTP server on device localhost.

The Python controller then calls:

```text
GET  /state
GET  /xml
GET  /health
POST /action
```

No screenshot or OCR is required for the normal loop.

## Accessibility permission

The service requires:

- retrieve window content
- perform gestures
- view IDs
- interactive window reporting

Some apps still expose incomplete trees. That is an Android/app limitation, not a planner limitation.

## Calls

`CursorInCallService` is intentionally conservative. Android may require the application to hold the appropriate Telecom role/default-phone-app status before call control is available.
