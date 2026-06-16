#!/usr/bin/env bash
# Launch the whole Voice OS. Run this IN YOUR OWN TERMINAL (so the clean
# ⚙ tool-call output is visible for screen recording, and so the accessibility
# keeper is "trusted").
#
#   cd .../voice-os && ./start.sh
#
# Hold LEFT OPTION + Z (⌥Z) anywhere to talk. Ctrl-C to quit.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
set -a; [ -f .env ] && source .env; set +a

# 1. keep Claude Desktop's accessibility tree forced on (for the ask_claude beat)
pkill -f ax_keeper.py 2>/dev/null || true
python ax_keeper.py >/dev/null 2>&1 &

# 2. black-and-white waveform overlay (top of screen while you talk)
pkill -f overlay.py 2>/dev/null || true
python overlay.py >/dev/null 2>&1 &

# 3. the voice OS, in the foreground so you SEE every ⚙ tool call / ✓ result.
#    (set VOICEOS_MIC to target a specific mic, e.g. VOICEOS_MIC=Scarlett ./start.sh)
pkill -f voice_app.py 2>/dev/null || true
exec python voice_app.py --combo opt+z ${VOICEOS_MIC:+--mic "$VOICEOS_MIC"}
