#!/usr/bin/env bash
# run.sh — bootstrap + launch the voice OS.
# Creates a venv, installs deps, loads .env, checks the control layer, then
# starts the gpt-realtime-2 voice loop. Idempotent; safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

# 1. Python venv + deps
if [ ! -d .venv ]; then
  echo "→ creating venv"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip >/dev/null
pip install -q -r requirements.txt

# 2. Control layer (agent-desktop)
if ! command -v agent-desktop >/dev/null 2>&1; then
  echo "→ installing agent-desktop (npm global)"
  npm install -g agent-desktop
fi
echo "→ agent-desktop permissions:"
agent-desktop permissions || true

# 3. API key
if [ -f .env ]; then
  set -a; # shellcheck disable=SC1091
  source .env; set +a
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "✗ OPENAI_API_KEY not set. Copy .env.example → .env and add a valid key."
  exit 1
fi

# 4. Go
if [[ " $* " == *" --local "* ]]; then
  echo "→ installing local wake engine (one-time)"
  pip install -q -r requirements-local.txt
  echo "→ launching LOCAL wake-word listener (\$0 idle)"
  exec python wake_listener.py "${@/--local/}"
fi
if [[ " $* " != *" --push-to-talk "* && " $* " != *" --hotkey "* ]]; then
  set -- --push-to-talk "$@"
fi
echo "→ launching voice agent"
exec python voice_agent.py "$@"
