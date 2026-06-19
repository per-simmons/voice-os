# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run (creates venv, installs deps, launches)
./run.sh                        # push-to-talk: press ENTER to talk
./run.sh --local                # local on-device wake word (OpenWakeWord + faster-whisper, $0 idle)
./run.sh --hotkey               # hold Right Control anywhere to talk
./run.sh --wake                 # cloud wake word "hey chat" (streams continuously — NOT $0 idle)

# Mic selection (any mode)
VOICEOS_MIC=Scarlett ./run.sh

# Run retrospective manually (learn from last N sessions)
cd src && python retrospective.py
cd src && python retrospective.py --sessions 3

# Test individual tools without OpenAI
cd src && python actions.py run_applescript 'tell application "Spotify" to play'
cd src && python actions.py open_url "https://example.com"
# (also works with legacy recipe names: open_app, play_music, web_search, etc.)

# Tests
pip install -r requirements-dev.txt
pytest                           # all tests
pytest tests/test_retrieval.py   # capability retrieval grounding tests (needs embedding model)
pytest tests/test_retrospective.py  # dreaming logic, pure — no network needed
python tests/test_loop.py "open Spotify"  # live end-to-end (needs OPENAI_API_KEY)
```

All Python is run from within the `.venv` that `run.sh` creates. Scripts in `src/` import each other directly (no package install needed; they find siblings by path).

## Architecture

### The main flow

```
voice → gpt-realtime-2 transcribes → retrieval.py embeds query + searches capabilities.json
→ RETRIEVED CAPABILITIES injected as system context → model calls a primitive tool
→ actions.py executes it → model speaks confirmation → session_log.py writes JSONL
→ (on Ctrl-C) retrospective.py reads log, calls gpt-4.1-mini, writes capabilities.user.json
```

The model has **no hardcoded routing rules**. It receives retrieved capability templates as recipes and fills in the parameters.

### Entry points

- **`src/voice_agent.py`** — cloud modes (PTT / hotkey / wake word "hey chat"). Opens a WebSocket to `wss://api.openai.com/v1/realtime`, streams audio, gates on the wake word regex, injects capability context, and dispatches tool calls to `actions.py`. Reconnects automatically on the 60-min API session cap.
- **`src/wake_listener.py`** — local `--local` mode. Two-stage pipeline: OpenWakeWord scores every 80ms frame ($0 CPU), then faster-whisper transcribes only post-wake audio. Once the command text is known, it opens a short-lived Realtime WebSocket to execute it. Imports `INSTRUCTIONS`, `MODEL`, `TOOLS`, `dispatch_tool` from `voice_agent.py`.

### Tools / actions (`src/actions.py`)

Two layers:

1. **5 primitives** (the only functions exposed to the model via `TOOLS` dict):
   - `run_applescript(script)` — arbitrary osascript
   - `press_key(combo, app, repeat?)` — CGEvent keystrokes via agent-desktop
   - `read_screen(app?)` — accessibility tree text extraction
   - `open_url(url)` — opens in configured browser
   - `obs_call(request_type, request_data)` — OBS WebSocket

2. **Recipe functions** (higher-level, used in the legacy `_LEGACY_TOOLS` dict for standalone CLI testing): `open_app`, `play_music`, `premiere_control`, `obs_scene`, `ask_claude`, `web_search`, `click_link`, `take_note`, etc.

The model uses the 5 primitives to implement whatever the capability template describes.

### Capability store and retrieval (`src/retrieval.py`)

- `memory/capabilities.json` — shipped capability templates (each has `id`, `description`, `examples[]`, `primitive`, `template`)
- `memory/capabilities.user.json` — your learned capabilities (gitignored), written by the retrospective
- On startup, `CapabilityIndex` embeds all example phrases with `sentence-transformers/all-MiniLM-L6-v2` (~22 MB, runs fully locally). The embedding cache (`memory/embeddings.npy` + `memory/embedding_ids.json`) auto-invalidates when capabilities change.
- Per-turn: the transcript is embedded, cosine similarity ranks capabilities, top-3 are injected as `RETRIEVED CAPABILITIES (grounding: STRONG|WEAK)` into the conversation before `response.create` fires.
- Grounding is STRONG when top score ≥ 0.52, or ≥ 0.40 with clear dominance over the runner-up. The system prompt tells the model to refuse ambiguous weak-grounding commands.

### Dreaming loop (`src/retrospective.py`)

Post-session (on Ctrl-C): reads JSONL session log, pairs successful tool calls with the spoken phrase that triggered them, calls `gpt-4.1-mini` to either (A) add the user's phrasing to an existing capability or (B) create a new one. Writes result to `capabilities.user.json` and refreshes the in-process retrieval index.

### Session logging (`src/session_log.py`)

Writes typed JSONL events to `memory/sessions/<timestamp>.jsonl` (gitignored). Event types: `heard`, `wake`, `ignored`, `tool_call` (with latency + ok/fail), `spoken`, `error`.

### Configuration (`src/config.py`)

Single source of truth for all constants, all overridable via env vars or `.env`. Key vars: `OPENAI_API_KEY`, `VOICEOS_TRANSCRIBE_MODEL` (default `gpt-4o-transcribe`), `VOICEOS_WHISPER` (local Whisper size), `VOICEOS_MIC`, `VOICEOS_BROWSER`, `VOICEOS_USER_NAME`, `VOICEOS_PREMIERE_APP`, `VOICEOS_SPOTIFY_FAVORITES`, `VOICEOS_CLAUDE_PROJECT`.

### App-specific quirks

- **Adobe Premiere Pro** — AX tree is empty (custom UI). `premiere_control` uses CoreGraphics to get the window rect, clicks the Program Monitor's video area to force panel keyboard focus, then sends keys via agent-desktop CGEvent. Adding a new Premiere shortcut = one line in `_PREMIERE_KEYS`.
- **Claude Desktop (Electron)** — AX tree is empty by default. `_force_electron_ax` calls `AXUIElementSetAttributeValue(..., "AXManualAccessibility", True)` to enable it. `ask_claude` navigates to a configured project, types a question, and polls the tree for the response.
- **Wake word regex** — `_WAKE_RE` in `voice_agent.py` handles NZ-accent mishears of "hey chat" (chut, chit, jet, jat, etc.) from gpt-4o-transcribe.

## Adding a new capability

Edit `memory/capabilities.json` (or `capabilities.user.json`) and restart — the embedding cache auto-regenerates. See `docs/ADD-AN-APP.md` for the full recipe: snapshot the app with `agent-desktop snapshot --app "AppName" --compact`, write a tool function in `actions.py`, register it in both `TOOLS` dicts, add a `TOOLS` entry in `voice_agent.py`.
