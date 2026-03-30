#!/data/data/com.termux/files/usr/bin/bash
# ─────────────────────────────────────────────────────────────
#  Cursor-for-Phone — Termux Setup Script
#  Run this once inside Termux on your Samsung.
# ─────────────────────────────────────────────────────────────

set -e
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║        Cursor-for-Phone — Setup              ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# 1. Update packages
echo "[1/6] Updating Termux packages..."
pkg update -y && pkg upgrade -y

# 2. Install Python + ADB
echo "[2/6] Installing Python and ADB..."
pkg install -y python android-tools

# 3. Install pip deps
echo "[3/6] Installing Python dependencies..."
pip install requests pyyaml

# 4. ADB over WiFi (no PC needed — phone connects to itself)
echo ""
echo "[4/6] Setting up wireless ADB..."
echo "  This requires Developer Options to be enabled."
echo "  Go to: Settings → About phone → tap Build number 7 times."
echo "  Then: Settings → Developer options → Wireless debugging → ON"
echo ""
read -p "  Press Enter when Wireless debugging is ON..."

# Try to pair via adb tcpip
echo "  Getting device IP..."
DEVICE_IP=$(ip route get 8.8.8.8 | awk '{print $7; exit}')
echo "  Your IP: $DEVICE_IP"

echo ""
echo "  In Developer Options → Wireless debugging → 'Pair device with pairing code'"
echo "  Enter the PORT and PAIRING CODE shown there."
echo ""
read -p "  Pairing port (e.g. 37291): " PAIR_PORT
read -p "  Pairing code (6 digits):   " PAIR_CODE

adb pair "${DEVICE_IP}:${PAIR_PORT}" "${PAIR_CODE}"

echo ""
read -p "  Connection port from 'Wireless debugging' screen (e.g. 42817): " CONN_PORT
adb connect "${DEVICE_IP}:${CONN_PORT}"

echo ""
echo "[5/6] Testing ADB connection..."
adb devices

# 5. Test UIAutomator
echo ""
echo "[6/6] Testing UIAutomator XML dump..."
adb shell uiautomator dump /sdcard/ui_test.xml
adb pull /sdcard/ui_test.xml /tmp/ui_test.xml
NODES=$(grep -c "clickable=\"true\"" /tmp/ui_test.xml || echo 0)
echo "  ✓ Got XML dump. Found ${NODES} clickable nodes on current screen."

echo ""
echo "══════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Export your API key:"
echo "    export ANTHROPIC_API_KEY=sk-ant-..."
echo "  (Add this to ~/.bashrc to make it permanent)"
echo ""
echo "  Run a script:"
echo "    cd cursor_phone"
echo "    python agent.py scripts/open_youtube_search.yaml"
echo ""
echo "  Interactive mode:"
echo "    python agent.py --interactive"
echo ""
echo "  Dry-run (no actual taps):"
echo "    python agent.py scripts/open_youtube_search.yaml --dry-run"
echo "══════════════════════════════════════════════"
