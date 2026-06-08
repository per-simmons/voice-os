#!/usr/bin/env python3
"""
actions.py — the "hands" of the voice OS.

Each function here is a single, reliable, high-level intent that maps to one of
the beats in the GPT-Realtime-2 cold-open script. The voice agent
(voice_agent.py) exposes these to gpt-realtime-2 as function tools; when you
speak an intent, the model picks the matching tool and we execute it here.

Design choice: we do NOT hand the realtime model the raw agent-desktop API and
ask it to drive a snapshot->click loop live on camera (too slow, too fragile).
Instead each tool is a deterministic recipe built from the most reliable path
for that specific app:
  - Spotify        -> AppleScript (has a real scripting dictionary)
  - launch/focus   -> agent-desktop launch (accessibility, no screenshots)
  - Terminal+Claude-> AppleScript to open + type a Claude Code prompt
  - read screen    -> agent-desktop accessibility-tree text extraction
  - OBS            -> launch + global hotkey via AppleScript keystroke

Every function returns a small JSON-able dict {status, ...}. The voice agent
JSON-encodes that as the function_call_output, and the model speaks a summary.

This module is fully runnable WITHOUT OpenAI — test any action directly:
    python actions.py open_app Spotify
    python actions.py play_music "Tchaikovsky"
    python actions.py read_screen_aloud
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time

AGENT_DESKTOP = shutil.which("agent-desktop") or "agent-desktop"


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------
def _run(cmd: list[str], timeout: int = 20) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _osa(script: str, timeout: int = 20) -> subprocess.CompletedProcess:
    return _run(["osascript", "-e", script], timeout=timeout)


def _ad(args: list[str], timeout: int = 20) -> dict:
    """Call agent-desktop and parse its structured JSON."""
    p = _run([AGENT_DESKTOP, *args], timeout=timeout)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "raw": (p.stdout or p.stderr)[:500]}


# ---------------------------------------------------------------------------
# tools exposed to gpt-realtime-2
# ---------------------------------------------------------------------------
def open_app(name: str) -> dict:
    """Launch or focus a macOS app by name (Spotify, OBS, Google Chrome, ...)."""
    res = _ad(["launch", name])
    ok = bool(res.get("ok"))
    return {
        "status": "ok" if ok else "error",
        "app": name,
        "title": res.get("data", {}).get("title"),
        "detail": None if ok else res,
    }


def play_music(query: str = "") -> dict:
    """
    Open Spotify and start playback. If `query` is given, search and play the
    first matching track/artist; otherwise just resume playback.
    """
    open_app("Spotify")
    time.sleep(1.5)
    if query:
        # Navigate Spotify to the search results for `query` via its URI scheme.
        # `open spotify:search:<q>` is far more reliable on camera than injecting
        # keystrokes into Spotify's Electron search box.
        #
        # NOTE: this reliably SHOWS the right results and resumes playback, but
        # `play` plays the active context, not guaranteed to be the top search
        # hit. To play an EXACT track every time, wire the Spotify Web API
        # (OAuth) and `play track "spotify:track:<id>"`. Documented in SETUP-LOG.
        from urllib.parse import quote

        _run(["open", f"spotify:search:{quote(query)}"])
        time.sleep(1.5)
    _osa('tell application "Spotify" to play')
    time.sleep(0.6)
    now = _osa('tell application "Spotify" to get name of current track')
    artist = _osa('tell application "Spotify" to get artist of current track')
    return {
        "status": "ok",
        "now_playing": (now.stdout or "").strip(),
        "artist": (artist.stdout or "").strip(),
        "query": query,
    }


def run_terminal(prompt: str) -> dict:
    """
    Open Terminal and start a Claude Code session with `prompt` as the request.
    Used for the 'ask Claude to write me a better intro' beat. Types the command
    and presses Return so it runs live on camera.
    """
    safe = prompt.replace('"', '\\"')
    # Open Terminal with a new window already cd'd to home, then run claude.
    script = f'''
    tell application "Terminal"
        activate
        do script "claude \\"{safe}\\""
    end tell
    '''
    p = _osa(script)
    ok = p.returncode == 0
    return {
        "status": "ok" if ok else "error",
        "prompt": prompt,
        "detail": None if ok else (p.stderr or "")[:300],
    }


def read_screen_aloud(app: str = "Terminal") -> dict:
    """
    Read back the text currently visible in `app` (default Terminal) using the
    accessibility tree — the 'Chat, what did Claude just say?' beat. Returns the
    extracted text so the model can speak it.
    """
    _ad(["launch", app])
    time.sleep(0.5)
    res = _ad(["snapshot", "--app", app, "--compact"], timeout=25)
    text = _extract_text(res.get("data", {}))
    # keep it speakable: last ~600 chars of visible text
    spoken = text[-600:].strip() if text else ""
    return {
        "status": "ok" if spoken else "empty",
        "app": app,
        "screen_text": spoken,
    }


def premiere_control(action: str = "pause") -> dict:
    """
    Control Adobe Premiere Pro playback for the cold-open 'cut, cut, cut / pause'
    beat. action: 'pause' | 'play' | 'stop'. Premiere's spacebar toggles play/
    pause; we bring Premiere to the front and send it. (For real razor cuts /
    markers / timeline ops, the deeper path is the premiere-pro MCP — see notes.)
    """
    # find the running Premiere process name (version-agnostic)
    p = _osa(
        'tell application "System Events" to get name of (first process whose '
        'name contains "Premiere")'
    )
    proc = (p.stdout or "").strip()
    if not proc:
        return {"status": "error", "error": "Premiere Pro is not running"}
    _osa(f'tell application "{proc}" to activate')
    time.sleep(0.4)
    if action in ("left", "back", "frame_back", "previous"):
        # left arrow = step playhead back one frame
        _osa("tell application \"System Events\" to key code 123")
    elif action in ("right", "forward", "frame_forward", "next"):
        # right arrow = step playhead forward one frame
        _osa("tell application \"System Events\" to key code 124")
    else:  # pause / play / stop all toggle playback with space
        _osa('tell application "System Events" to keystroke " "')
    return {"status": "ok", "app": proc, "action": action}


def start_obs_recording() -> dict:
    """Open OBS and start recording (global hotkey). The 'open OBS' beat."""
    open_app("OBS")
    time.sleep(2.5)
    # OBS default has no global record hotkey set; we click via accessibility.
    # Most reliable cross-setup path: bring OBS front, then send Cmd? -> instead
    # we look for a 'Start Recording' button in the tree.
    res = _ad(["find", "--role", "button", "--app", "OBS"], timeout=20)
    btn = _find_button(res.get("data", {}), ("start recording", "record"))
    if btn:
        snap = res.get("data", {}).get("snapshot_id")
        click_args = ["click", btn]
        if snap:
            click_args += ["--snapshot", snap]
        _ad(click_args)
        return {"status": "ok", "action": "clicked Start Recording", "ref": btn}
    return {
        "status": "opened",
        "note": "OBS is open and focused; assign a Start Recording hotkey or click it on camera.",
    }


# ---------------------------------------------------------------------------
# text extraction helpers for accessibility trees
# ---------------------------------------------------------------------------
def _extract_text(node) -> str:
    out: list[str] = []

    def walk(n):
        if isinstance(n, dict):
            for k in ("value", "name", "title", "text"):
                v = n.get(k)
                if isinstance(v, str) and v.strip():
                    out.append(v.strip())
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(node)
    # de-dup consecutive repeats, join
    seen: list[str] = []
    for s in out:
        if not seen or seen[-1] != s:
            seen.append(s)
    return "\n".join(seen)


def _find_button(node, needles) -> str | None:
    found = {"ref": None}

    def walk(n):
        if found["ref"]:
            return
        if isinstance(n, dict):
            name = " ".join(
                str(n.get(k, "")) for k in ("name", "title", "value")
            ).lower()
            if n.get("ref_id") and any(x in name for x in needles):
                found["ref"] = n["ref_id"]
                return
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(node)
    return found["ref"]


# tool registry the voice agent imports
TOOLS = {
    "open_app": open_app,
    "play_music": play_music,
    "run_terminal": run_terminal,
    "read_screen_aloud": read_screen_aloud,
    "start_obs_recording": start_obs_recording,
    "premiere_control": premiere_control,
}


# ---------------------------------------------------------------------------
# CLI for standalone testing (no OpenAI needed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in TOOLS:
        print("usage: python actions.py <tool> [args...]")
        print("tools:", ", ".join(TOOLS))
        sys.exit(1)
    fn = TOOLS[sys.argv[1]]
    kwargs = {}
    rest = sys.argv[2:]
    if rest:
        # positional -> first param name
        import inspect

        params = list(inspect.signature(fn).parameters)
        kwargs[params[0]] = " ".join(rest)
    print(json.dumps(fn(**kwargs), indent=2))
