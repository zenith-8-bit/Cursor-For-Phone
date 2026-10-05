# Local bridge protocol

PC:

```text
http://127.0.0.1:8765
```

Android:

```text
127.0.0.1:8765
```

Transport:

```powershell
adb reverse tcp:8765 tcp:8765
```

## GET /health

Returns bridge/service status.

## GET /state

Returns:

```json
{
  "revision": 12,
  "package": "com.example",
  "activity": "MainActivity",
  "screen": "APP",
  "elements": [
    {
      "id": "e123",
      "text": "Search",
      "clickable": true,
      "editable": false,
      "bounds": {"left":0,"top":0,"right":500,"bottom":100}
    }
  ]
}
```

## GET /xml

Returns an XML-like serialization of the current Accessibility tree.

## POST /action

Example:

```json
{
  "action": "CLICK",
  "target_id": "e123",
  "confirmed": false
}
```

or:

```json
{
  "action": "TYPE",
  "target_id": "e456",
  "text": "black hoodies"
}
```

The Android service resolves the ID against the current node map. It never accepts an arbitrary code string.
