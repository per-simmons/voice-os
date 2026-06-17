# Voice OS — run your Mac with your voice

Talk to your Mac and it actually does things — open apps, play music, control
Premiere, start a recording, read the screen back. A small, hackable starting
point you can clone and extend. **It learns your workflows over time.**

- **Brain:** OpenAI `gpt-realtime-2` (speech-to-speech + tool calling)
- **Hands:** 5 primitive tools — AppleScript, key presses, screen reading, URL opening, OBS WebSocket
- **Memory:** local semantic retrieval — capabilities embedded with `sentence-transformers`, searched per-turn, no API cost
- **Learning:** post-session "dreaming" — the model reflects on what worked and writes new capability templates for next time

> Built for the "run your entire computer with your voice" video. Clone it, hand
> it to your coding agent, and say _"build this out for me."_
> See **[ADD-AN-APP.md](ADD-AN-APP.md)**.

---

## Quickstart

**Requirements:** a Mac, [Node](https://nodejs.org) (for agent-desktop),
Python 3.10+, and an **OpenAI API key with Realtime access**.

```bash
# 1. install the "hands"
npm install -g agent-desktop
#    grant Accessibility: System Settings → Privacy & Security → Accessibility

# 2. add your key
cp .env.example .env          # paste OPENAI_API_KEY into .env

# 3. run it (creates a venv, installs deps, launches)
./run.sh                      # push-to-talk: press ENTER, talk, it acts
```

Then talk: _"open Spotify," "play some jazz," "cut here," "what's on my screen?"_

---

## Ways to talk to it

| Mode                           | Command            | Notes                                             |
| ------------------------------ | ------------------ | ------------------------------------------------- |
| **Push-to-talk** (recommended) | `./run.sh`         | Press ENTER, talk. 100% reliable, $0 idle.        |
| **Hold-to-talk hotkey**        | `./ptt.sh`         | Hold **Right Control** anywhere to talk.          |
| **Wake word "hey chat"**       | `./run.sh --local` | Local on-device wake word. Great in a quiet room. |

Pick a specific mic with `VOICEOS_MIC=Scarlett ./ptt.sh`.

---

## How it works

```
your voice
    │
    ▼
gpt-realtime-2 transcribes the command
    │
    ▼
retrieval.py embeds the transcript and searches capabilities.json
returns top-3 matching capability templates (locally, ~2ms)
    │
    ▼
context injected: "RETRIEVED CAPABILITIES: cut in Premiere → press_key cmd+k"
    │
    ▼
model calls the right primitive with filled-in parameters
    │
    ├─ run_applescript(script)       AppleScript for any scriptable app
    ├─ press_key(combo, app)         hotkeys for Premiere, etc.
    ├─ read_screen(app)              accessibility-tree text extraction
    ├─ open_url(url)                 browser / Spotify URI schemes
    └─ obs_call(requestType, data)   OBS WebSocket
    │
    ▼
app does the thing → model speaks a short confirmation
    │
    ▼  (on Ctrl-C / session end)
retrospective.py reflects on what worked → writes new templates to
memory/capabilities.user.json → re-embedded next startup
```

The model has **no hardcoded routing rules**. It receives the retrieved
capability as a recipe and fills in the parameters. New capabilities are
added by editing `memory/capabilities.json` — or by just using the OS and
letting the retrospective learn them for you.

---

## The capability store

`memory/capabilities.json` — shipped generic capabilities (app launching,
Spotify, Premiere editing, OBS, web search, notes, etc.).

`memory/capabilities.user.json` — your personal capabilities, written by the
dreaming loop after each session. Gitignored. Format:

```json
[
  {
    "id": "premiere-cut",
    "description": "Cut/razor at the playhead in Premiere Pro",
    "examples": ["cut here", "razor at playhead", "add edit", "split here"],
    "primitive": "press_key",
    "template": { "combo": "cmd+k", "app": "Adobe Premiere Pro" }
  }
]
```

**Adding a new capability:** add an entry to either JSON file and restart. The
embedding cache auto-regenerates. No Python needed.

---

## The dreaming loop

At the end of every session (Ctrl-C), the system runs a retrospective:

```bash
python retrospective.py              # reflect on the last session
python retrospective.py --sessions 3 # reflect on the last 3 sessions
```

It reads the structured session log, identifies patterns in what worked, and
appends new capability entries to `memory/capabilities.user.json`. Next startup,
those entries are embedded and retrievable. The OS learns your specific workflow
without you having to write any code.

---

## Session history

Every session is logged as structured JSONL in `memory/sessions/` (gitignored):

```
memory/sessions/2026-06-18T00-30-00.jsonl
```

Each line is a typed event — `heard`, `wake`, `tool_call` (with latency and
ok/fail), `spoken`, `error`. Useful for debugging, cost auditing, or feeding
into the retrospective manually.

---

## Configuration

All tuneable values live in `.env` (copy `.env.example` to get started):

| Variable                      | Default              | Description                                               |
| ----------------------------- | -------------------- | --------------------------------------------------------- |
| `OPENAI_API_KEY`              | —                    | Required. Realtime-capable key.                           |
| `VOICEOS_BROWSER`             | `Safari`             | Browser for web searches.                                 |
| `VOICEOS_USER_NAME`           | `the user`           | Your name in the system prompt.                           |
| `VOICEOS_USER_HINTS`          | —                    | Free-text hints e.g. accent, preferences.                 |
| `VOICEOS_PREMIERE_APP`        | `Adobe Premiere Pro` | Exact app name (update yearly).                           |
| `VOICEOS_SPOTIFY_FAVORITES`   | —                    | Path to JSON file of phrase → spotify URI mappings.       |
| `VOICEOS_CLAUDE_PROJECT`      | —                    | Claude Desktop project name for `ask_claude`.             |
| `VOICEOS_CLAUDE_PROJECT_HINT` | —                    | Phrase from the project's system prompt (skip-nav check). |
| `VOICEOS_EMBED_MODEL`         | `all-MiniLM-L6-v2`   | Local sentence-transformers model for retrieval.          |

---

## Cost & privacy

- **Cost:** `gpt-realtime-2` is ~$32/$64 per 1M audio tokens — roughly **a few
  cents per command**. Push-to-talk and local wake word are **$0 idle**.
- **Retrieval:** runs fully locally via `sentence-transformers`. No API calls, no
  cost, works offline.
- **Privacy:** mic audio is sent to OpenAI only during an active command. Session
  logs and learned capabilities stay on your machine.

---

## Files

| File                            | Role                                                 |
| ------------------------------- | ---------------------------------------------------- |
| `voice_agent.py`                | Realtime loop — mic ↔ model ↔ tools, all input modes |
| `actions.py`                    | The 5 primitive tools the model calls                |
| `retrieval.py`                  | Local capability embedding index and cosine search   |
| `session_log.py`                | Structured per-session JSONL event logger            |
| `retrospective.py`              | Post-session dreaming loop — learns new capabilities |
| `wake_listener.py`              | Local ($0-idle) wake-word variant                    |
| `config.py`                     | All tuneable constants, read from env                |
| `overlay.py`                    | Waveform HUD                                         |
| `memory/capabilities.json`      | Shipped capability templates                         |
| `memory/capabilities.user.json` | Your learned capabilities (gitignored)               |
| `memory/sessions/`              | Session logs (gitignored)                            |

License: MIT. Have fun — go tell your agent to build it out.
