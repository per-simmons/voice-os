#!/usr/bin/env bash
# Hold-to-talk launcher. HOLD the Right Control (⌃) key ANYWHERE to talk, release
# to send. macOS will prompt for Input Monitoring the first time — allow it.
# Pick a specific mic with VOICEOS_MIC, e.g.  VOICEOS_MIC=Scarlett ./ptt.sh
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
set -a; source .env; set +a
exec python src/voice_agent.py --hotkey right_ctrl ${VOICEOS_MIC:+--mic "$VOICEOS_MIC"}
