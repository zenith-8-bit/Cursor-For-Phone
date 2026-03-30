# Cursor-for-Phone 📱

A budget Cursor-style AI agent for Android. Reads your screen's
UI XML, understands what's on it, and taps/types/swipes to
complete tasks described in plain English — all from Termux,
no PC required.

## How it works

```
YAML Script  ──► agent.py
                    │
                    ▼
             context.py          ← ADB dumps screen XML
             (UIAutomator XML)     parses all interactive nodes
                    │
                    ▼
             llm.py              ← sends scene summary + intent
             (Claude API)          gets back: {"type":"tap","target":"12"}
                    │
                    ▼
             executor.py         ← translates to:
             (ADB commands)        adb shell input tap 540 960
                    │
                    ▼
             verify (re-capture) ← if state changed → next step
                    │
                    ▼
             failsafe.py         ← if stuck → asks you what to do
```

## Prerequisites

- Samsung running Android 15
- Termux (from F-Droid, NOT Play Store)
- Developer Options enabled
- Wireless Debugging turned on
- Anthropic API key

## Install

```bash
# In Termux:
cd ~
git clone <this-repo> cursor_phone   # or copy files manually
cd cursor_phone
bash setup.sh
```

The setup script:
1. Installs `python`, `android-tools` (ADB), `requests`, `pyyaml`
2. Guides you through wireless ADB pairing (no PC needed)
3. Tests UIAutomator XML dump

## Usage

### Run a script

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python agent.py scripts/open_youtube_search.yaml
```

### Interactive (REPL) mode

```bash
python agent.py --interactive
>>> open Chrome and go to reddit.com
>>> scroll down three times
>>> tap the first post
>>> quit
```

### Dry run (see what it would do, no actual taps)

```bash
python agent.py scripts/wifi_toggle.yaml --dry-run
```

## Writing scripts

Scripts are `.yaml` files with a list of steps:

```yaml
name: "My Automation"
config:
  model: "claude-haiku-4-5-20251001"   # cheaper/faster
  max_retries: 3
  default_wait: 2.0                    # seconds between steps

steps:
  - do: "press Home button"
    wait: 1.0

  - do: "open Settings app"
    wait: 2.5

  - do: "tap on Battery"
    # uses default_wait if not specified

  - do: "tap the Battery saver toggle to turn it on"
```

Each `do:` is plain English. The LLM reads the live screen state
and figures out which element to interact with.

## Failsafe

If a step fails after `max_retries` attempts, you get a menu:

```
⚠  FAILSAFE — could not complete step:
   Intent: tap the send button

Current screen state:
APP: com.whatsapp
INTERACTIVE ELEMENTS (8):
  [3] ImageButton | label='Send' | actions=['click'] | ...

Options:
  [r] Retry with a new instruction
  [s] Skip this step
  [a] Abort the script
  [d] Describe what you see and let me re-plan
```

## File structure

```
cursor_phone/
├── agent.py      — orchestrator / entry point
├── context.py    — ADB + XML parser → Scene
├── llm.py        — Claude API → Action
├── executor.py   — Action → adb shell commands
├── failsafe.py   — stuck-state recovery
├── logger.py     — coloured terminal output
├── setup.sh      — one-time install + ADB pairing
└── scripts/
    ├── open_youtube_search.yaml
    ├── send_whatsapp.yaml
    ├── wifi_toggle.yaml
    └── circle_to_search.yaml
```

## Tips

- **Wider steps beat narrow ones.** Write `"tap the search icon"` not
  `"tap the element at position 3"` — let the LLM match it.
- **Add `wait:` generously.** Slow apps need 2–3 seconds after opening.
- **Use `--dry-run` first** to check the plan before it touches anything.
- **Keep the screen on** — Android may lock while the script runs.
  Settings → Display → Screen timeout → 10 minutes.
- **ADB disconnects?** Re-run `adb connect <ip>:<port>` in Termux.
