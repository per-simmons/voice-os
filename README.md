# Voice OS — run your Mac with your voice

Talk to your Mac and it actually does things — open apps, play music, run
terminal commands, read the screen back, control Premiere, start a recording.
A small, hackable starting point you can clone and extend to **any** app.

- **Brain:** OpenAI `gpt-realtime-2` (speech-to-speech + tool calling)
- **Hands:** [`agent-desktop`](https://github.com/per-simmons/agent-desktop) — drives Mac apps via the macOS accessibility tree (no screenshots)
- **Glue:** this repo — a handful of voice "tools" the model can call

> Built for the "run your entire computer with your voice" video. It's meant to
> be taken apart: clone it, hand it to your coding agent, and say *"build this
> out for me / add my app."* See **[ADD-AN-APP.md](ADD-AN-APP.md)**.

---

## Quickstart

**Requirements:** a Mac, [Node](https://nodejs.org) (for agent-desktop),
Python 3.10+, and an **OpenAI API key with Realtime access** (billed ~pennies
per command — see Cost below).

```bash
# 1. install the "hands"
npm install -g agent-desktop
#    then grant Accessibility: System Settings → Privacy & Security → Accessibility

# 2. add your key
cp .env.example .env          # paste your OPENAI_API_KEY into .env

# 3. run it (creates a venv, installs deps, launches)
./run.sh                      # push-to-talk: press ENTER, talk, it acts
```

Then talk: *"open Spotify," "play some Tchaikovsky," "what's on my screen?"*

---

## Ways to talk to it

| Mode | Command | Notes |
|------|---------|-------|
| **Push-to-talk** (recommended) | `./run.sh` | Press ENTER, talk. 100% reliable, $0 idle. |
| **Hold-to-talk hotkey** | `./ptt.sh` | Hold **Right Control** anywhere to talk. Needs Input Monitoring permission. |
| **Wake word "hey chat"** | `./run.sh --local` | Local wake word (free, on-device). Great in a quiet room; finicky in noise — see below. |
| **Cloud wake word** | (default of `voice_agent.py`) | Simplest, but streams the mic the whole time (~$1/hr idle). |

Pick a specific mic anywhere with `VOICEOS_MIC=Scarlett ./ptt.sh`.

**Optional waveform HUD (experimental):** `python overlay.py` shows a black-and-
white waveform at the top of the screen while you talk. ⚠️ The current tkinter
version is always-on-top and can intercept clicks near the top of the screen — a
native click-through rebuild is planned. Run it only while demoing.

---

## What it can do (the tools)

`open_app` · `play_music` (Spotify) · `run_terminal` (opens Terminal + your CLI
agent) · `read_screen_aloud` · `start_obs_recording` · `premiere_control`
(play/pause + nudge the playhead a frame).

**These are just examples.** Adding your own app is the whole point —
see **[ADD-AN-APP.md](ADD-AN-APP.md)**. Short version: every tool is a ~15-line
Python function in `actions.py`; the model calls it when your spoken intent
matches its description. You can even tell your coding agent to write new ones
for you (it can inspect any app via `agent-desktop snapshot`).

---

## Cost & privacy

- **Cost:** `gpt-realtime-2` is ~$32 / $64 per 1M audio in/out tokens — roughly
  **a few cents per command**. Push-to-talk and local wake word are **$0 when
  idle**; the always-on cloud wake word streams continuously (~$1/hr). Watch real
  spend at platform.openai.com/usage.
- **Privacy (local wake word):** the mic is processed **on your Mac and
  discarded** — nothing leaves the machine until you say the wake word. Push-to-
  talk sends audio only while you hold the key.

---

## Wake word, honestly

Catching a *short* phrase like "hey chat" with general speech-to-text is the
fiddly part — in a noisy room it gets misheard. Three ways to make the trigger
solid:
1. **Picovoice** — free, train a custom "hey ___" wake word (~2 min), very
   accurate even in noise. Best for a custom phrase.
2. **openWakeWord** — free, pretrained phrases like "Hey Jarvis" that work
   out of the box (but you take their phrase).
3. **Push-to-talk / hotkey** — a button instead of a wake word. 100% reliable.
   What this repo ships with by default.

General speech-to-text is great for the *command*; a dedicated wake engine (or a
button) is what makes the always-on *trigger* reliable.

---

## How it works

```
your voice ─▶ gpt-realtime-2 (decides which tool) ─▶ actions.py
                                                        ├─ agent-desktop  (click/read any app via the accessibility tree)
                                                        └─ AppleScript    (deeper control for scriptable apps)
                                                      ─▶ app does the thing ─▶ model speaks a confirmation
```

- `voice_agent.py` — the realtime loop (mic ↔ model ↔ tools), with push-to-talk,
  hotkey, and wake-word modes.
- `wake_listener.py` — the local ($0-idle) wake-word variant.
- `actions.py` — the tools (the hands). Runnable standalone:
  `python actions.py open_app Spotify`.
- `overlay.py` — the waveform HUD.

License: MIT. Have fun — go tell your agent to build it out.
