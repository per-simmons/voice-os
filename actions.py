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

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time

AGENT_DESKTOP = shutil.which("agent-desktop") or "agent-desktop"

CLAUDE_LOG = "/tmp/voiceos-claude.log"
_clog_t0 = [0.0]


def _clog(msg: str):
    """Step-level logging for the Claude Desktop flow (so we can see inside it)."""
    el = time.monotonic() - _clog_t0[0] if _clog_t0[0] else 0.0
    line = f"[+{el:5.1f}s] {msg}"
    print(line, flush=True)
    try:
        with open(CLAUDE_LOG, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


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
# Map spoken / mis-transcribed app names to the real macOS app name.
APP_ALIASES = {
    "claude desktop": "Claude",
    "cloud desktop": "Claude",
    "claude": "Claude",
    "chrome": "Google Chrome",
    "google chrome": "Google Chrome",
    "premiere": "Adobe Premiere Pro 2025",
    "premiere pro": "Adobe Premiere Pro 2025",
    "obs studio": "OBS",
    "spotify app": "Spotify",
}


def open_app(name: str) -> dict:
    """Launch or focus a macOS app by name and bring it to the FRONT (above
    whatever's currently active, e.g. Premiere)."""
    name = APP_ALIASES.get((name or "").strip().lower(), name)
    res = _ad(["launch", name])
    ok = bool(res.get("ok"))
    # force it to the foreground — agent-desktop launches/focuses but the current
    # frontmost app can stay on top; `activate` + `open -a` make it the main app.
    _osa(f'tell application "{name}" to activate')
    _run(["open", "-a", name])
    return {
        "status": "ok" if ok else "error",
        "app": name,
        "title": res.get("data", {}).get("title"),
        "detail": None if ok else res,
    }


# Pre-programmed exact tracks: if the spoken query contains one of these phrases,
# play that EXACT Spotify track (reliable on camera). Add your own:
#   "phrase" : "spotify:track:<id>"   (grab the id from the song's Spotify URL)
FAVORITES = {
    "herbie hancock": "spotify:track:38xcUjiTP1ivfb7ObwjyGA",  # Watermelon Man (Remastered 2007, Takin' Off)
    "watermelon man": "spotify:track:38xcUjiTP1ivfb7ObwjyGA",
}


def play_music(query: str = "") -> dict:
    """
    Open Spotify and start playback. Exact-track favorites (see FAVORITES) play
    the precise song; otherwise navigate to the search and resume.
    """
    open_app("Spotify")
    time.sleep(1.5)
    ql = (query or "").lower()
    for phrase, uri in FAVORITES.items():
        if phrase in ql:
            # shuffle ON makes `play track` shuffle the album and land on a random
            # neighbor — force it off so we get the EXACT track every time.
            _osa('tell application "Spotify" to set shuffling to false')
            _osa(f'tell application "Spotify" to play track "{uri}"')
            time.sleep(0.6)
            now = _osa('tell application "Spotify" to get name of current track')
            artist = _osa('tell application "Spotify" to get artist of current track')
            return {
                "status": "ok",
                "now_playing": (now.stdout or "").strip(),
                "artist": (artist.stdout or "").strip(),
                "query": query,
                "matched": phrase,
            }
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


# Claude Desktop is pre-programmed (via its project instructions) to reply with
# this exact script when asked for a rewrite. We return it as the read-back text
# (Claude Desktop is Electron — its reply isn't readable via the accessibility
# tree — so we use the known deterministic output, which matches what's on screen).
CLAUDE_DESKTOP_RESPONSE = (
    "This is GPT-Realtime 2 in action.\n\n"
    "And in this video, I'm going to show you exactly how to build this yourself "
    "— everything from opening your apps to fully commanding them, just by talking.\n\n"
    "You'll get a glimpse into the future of a new kind of operating system — one "
    "you run entirely with your own voice.\n\n"
    "And the best part? No coding or technical knowledge is required. All it takes "
    "is a few prompts to Claude Code."
)


def _force_electron_ax(app_name: str = "Claude"):
    """Force an Electron app (Claude Desktop) to build its accessibility tree, so
    agent-desktop can read/navigate it. Without this, Claude's tree is empty."""
    try:
        pids = subprocess.run(["pgrep", "-x", app_name], capture_output=True, text=True).stdout.split()
        if not pids:
            return
        from ApplicationServices import (
            AXUIElementCreateApplication,
            AXUIElementSetAttributeValue,
        )
        AXUIElementSetAttributeValue(
            AXUIElementCreateApplication(int(pids[0])), "AXManualAccessibility", True
        )
    except Exception:  # noqa: BLE001
        pass


def _claude_snapshot() -> dict:
    return _ad(["snapshot", "--app", "Claude", "--max-depth", "35"], timeout=25).get("data", {})


def _claude_click_exact(name: str) -> bool:
    """Find + click a Claude element whose name == `name` exactly (full snapshot)."""
    data = _claude_snapshot()
    sid = data.get("snapshot_id")
    target = name.lower()
    found = {"ref": None}

    def walk(n):
        if found["ref"]:
            return
        if isinstance(n, dict):
            if n.get("ref_id") and (n.get("name") or "").strip().lower() == target:
                found["ref"] = n["ref_id"]
                return
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(data)
    if found["ref"] and sid:
        _ad(["click", found["ref"], "--snapshot", sid])
        return True
    return False


def _find_link_by_subtext(data: dict, needle: str):
    """Find a Claude project card: a `link` whose SUBTREE text contains needle
    (project cards have empty names; their title lives in a child statictext)."""
    needle = needle.lower()
    found = {"ref": None}

    def subtext(n, acc):
        if isinstance(n, dict):
            for k in ("name", "value", "title", "text"):
                v = n.get(k)
                if isinstance(v, str):
                    acc.append(v)
            for v in n.values():
                subtext(v, acc)
        elif isinstance(n, list):
            for v in n:
                subtext(v, acc)

    def walk(n):
        if found["ref"]:
            return
        if isinstance(n, dict):
            if n.get("role") == "link" and n.get("ref_id"):
                acc = []
                subtext(n, acc)
                if any(needle in (t or "").lower() for t in acc):
                    found["ref"] = n["ref_id"]
                    return
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(data)
    return found["ref"]


def _claude_in_project(name: str) -> bool:
    """True if Claude is ALREADY showing the named project (so we can skip the
    navigation + its visible bounce). The project's instructions/header show on
    screen when you're inside it."""
    blob = json.dumps(_claude_snapshot()).lower()
    if '"name": "new chat - claude"' in blob or '"name": "projects - claude"' in blob:
        return False
    # the project's own instruction text shows in the project view
    return "you are helping with a youtube script" in blob


def _claude_open_project(name: str) -> bool:
    """Navigate Claude Desktop to a project by name: click Projects, then the card.
    Retries finding the project card — the Projects panel renders into the
    accessibility tree on a delay, so the card isn't always there on first look."""
    _clog("nav: clicking Projects")
    _claude_click_exact("Projects")
    for i in range(10):
        time.sleep(0.8)
        _force_electron_ax("Claude")
        data = _claude_snapshot()
        sid = data.get("snapshot_id")
        ref = _find_link_by_subtext(data, name)
        # DIAGNOSTICS: element count (is the tree populated?) + does 'youtube' appear?
        blob = json.dumps(data).lower()
        n_el = blob.count('"role"')
        has_yt = "youtube" in blob
        _clog(f"nav: try {i + 1} -> ref={ref or 'none'} | elements={n_el} | 'youtube' present={has_yt}")
        if ref and sid:
            _ad(["click", ref, "--snapshot", sid])
            time.sleep(1.3)
            _force_electron_ax("Claude")
            _clog("nav: clicked project card -> IN PROJECT")
            return True
        if i in (3, 6):  # the Projects click may not have registered — re-click it
            _clog("nav: re-clicking Projects")
            _claude_click_exact("Projects")
    _clog("nav: project card NEVER found -> will fall back to current view")
    return False


def _claude_focus_compose():
    """Click the compose textfield so keystrokes land in it."""
    data = _claude_snapshot()
    sid = data.get("snapshot_id")
    found = {"ref": None}

    def walk(n):
        if found["ref"]:
            return
        if isinstance(n, dict):
            if n.get("role") == "textfield" and n.get("ref_id"):
                found["ref"] = n["ref_id"]
                return
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(data)
    if found["ref"] and sid:
        _ad(["click", found["ref"], "--snapshot", sid])
        time.sleep(0.4)


def _claude_all_text() -> list:
    out = []

    def walk(n):
        if isinstance(n, dict):
            if n.get("role") in ("statictext", "paragraph"):
                v = (n.get("name") or n.get("value") or "").strip()
                if v:
                    out.append(v)
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(_claude_snapshot())
    return out


def _read_claude_response(timeout: float = 25.0) -> str:
    """Poll Claude's tree until the response finishes, then return its text."""
    deadline = time.time() + timeout
    last = ""
    n = 0
    while time.time() < deadline:
        _force_electron_ax("Claude")
        texts = _claude_all_text()
        # the app labels the assistant turn as "Claude responded: <text>"
        responded = [t[len("Claude responded:"):].strip()
                     for t in texts if t.startswith("Claude responded:")]
        if responded:
            last = responded[-1]
        finished = any("finished the response" in t.lower() for t in texts)
        n += 1
        if finished and last:
            _clog(f"read: finished after {n} polls -> {last[:50]!r}")
            return last
        time.sleep(0.7)
    _clog(f"read: TIMED OUT after {n} polls, last={last[:50]!r}")
    return last


def ask_claude(question: str = "",
               project: str = "YouTube Script") -> dict:
    """
    Open the YouTube Script project in Claude Desktop, type the question into the
    project's compose box on screen, send it, and read Claude's ACTUAL reply back
    for chat to speak. The project's instructions make Claude answer with the
    locked script. Navigation: force the accessibility tree on, click Projects,
    click the project card (matched by its child text), type, send, read.
    """
    _clog_t0[0] = time.monotonic()
    q = (question or "").strip()
    _clog(f"ask_claude START — project={project!r} question={q[:50]!r}")
    open_app("Claude")  # bring Claude Desktop to the front
    time.sleep(1.0)
    _force_electron_ax("Claude")
    time.sleep(0.5)
    if project and _claude_in_project(project):
        _clog("already in project -> skipping navigation (no bounce)")
        in_project = True
    elif project:
        in_project = _claude_open_project(project)
    else:
        in_project = False
    _clog(f"project_opened={in_project}")
    if not q:
        # "open my YouTube script project" with no question — just open it, fast.
        _clog(f"no question -> opened project only, done in {time.monotonic()-_clog_t0[0]:.1f}s")
        return {
            "status": "ok",
            "project_opened": in_project,
            "question": "",
            "response": "Opened the YouTube Script project." if in_project
            else "Opened Claude.",
        }
    _claude_focus_compose()  # focus the compose box (project's, or current view's)
    _clog("typing question into compose")
    safe = q.replace("\\", "").replace('"', '\\"')
    _osa(f'tell application "System Events" to keystroke "{safe}"')
    time.sleep(0.4)
    _osa("tell application \"System Events\" to key code 36")  # Return = send
    _clog("sent; waiting for Claude's reply…")
    reply = _read_claude_response(timeout=22.0)
    _clog(f"ask_claude DONE in {time.monotonic()-_clog_t0[0]:.1f}s  in_project={in_project}")
    return {
        "status": "ok",
        "project_opened": in_project,
        "question": q,
        "response": reply or CLAUDE_DESKTOP_RESPONSE,
    }


def _find_clickable_by_subtext(data, needle, roles=("cell", "row", "button", "link")):
    """Find an element of `roles` whose SUBTREE text contains needle (the label
    is often a child statictext with no ref of its own)."""
    needle = needle.lower()
    found = {"ref": None}

    def subtext(n, acc):
        if isinstance(n, dict):
            for k in ("name", "value", "title", "text"):
                v = n.get(k)
                if isinstance(v, str):
                    acc.append(v)
            for v in n.values():
                subtext(v, acc)
        elif isinstance(n, list):
            for v in n:
                subtext(v, acc)

    def walk(n):
        if found["ref"]:
            return
        if isinstance(n, dict):
            if n.get("role") in roles and n.get("ref_id"):
                acc = []
                subtext(n, acc)
                if any(needle in (t or "").lower() for t in acc):
                    found["ref"] = n["ref_id"]
                    return
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(data)
    return found["ref"]


# ---- OBS control via its built-in WebSocket (reliable; not tree-dependent) ----
OBS_WS_CONFIG = os.path.expanduser(
    "~/Library/Application Support/obs-studio/plugin_config/obs-websocket/config.json"
)


def _obs_password() -> str:
    try:
        return json.load(open(OBS_WS_CONFIG)).get("server_password", "")
    except Exception:  # noqa: BLE001
        return ""


async def _obs_call_async(reqs):
    import base64
    import hashlib

    import websockets

    async with websockets.connect("ws://127.0.0.1:4455", max_size=None) as ws:
        hello = json.loads(await ws.recv())["d"]
        ident = {"op": 1, "d": {"rpcVersion": 1}}
        if "authentication" in hello:
            pw = _obs_password()
            salt = hello["authentication"]["salt"]
            ch = hello["authentication"]["challenge"]
            secret = base64.b64encode(hashlib.sha256((pw + salt).encode()).digest()).decode()
            ident["d"]["authentication"] = base64.b64encode(
                hashlib.sha256((secret + ch).encode()).digest()
            ).decode()
        await ws.send(json.dumps(ident))
        await ws.recv()  # Identified
        out = []
        for rt, rd in reqs:
            await ws.send(json.dumps({"op": 6, "d": {
                "requestType": rt, "requestId": rt, "requestData": rd or {}}}))
            while True:
                m = json.loads(await ws.recv())
                if m.get("op") == 7 and m["d"].get("requestId") == rt:
                    out.append(m["d"])
                    break
        return out


def _obs_call(reqs):
    return asyncio.run(_obs_call_async(reqs))


def _ensure_obs():
    """Make sure OBS is running WITHOUT stealing focus (open -ga = background)."""
    _run(["open", "-ga", "OBS"])
    time.sleep(0.2)


def obs_scene(name: str = "YouTube Talking Head") -> dict:
    """Switch OBS to a scene by name (fuzzy). Brings OBS to the front, then uses
    the OBS WebSocket to set the program scene — reliable, not tree-dependent."""
    _ensure_obs()
    try:
        scenes = [s["sceneName"] for s in
                  _obs_call([("GetSceneList", {})])[0]["responseData"]["scenes"]]
        nl = (name or "").lower()
        target = next((s for s in scenes if nl and (nl in s.lower() or s.lower() in nl)), None)
        if not target:
            import difflib
            m = difflib.get_close_matches(name, scenes, n=1, cutoff=0.4)
            target = m[0] if m else None
        if not target:
            return {"status": "error", "error": f"no scene matching {name!r}", "scenes": scenes}
        r = _obs_call([("SetCurrentProgramScene", {"sceneName": target})])
        ok = r[0]["requestStatus"]["result"]
        return {"status": "ok" if ok else "error", "scene": target}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}


# Premiere actions → key combos. Add a line to teach a new editing command.
_PREMIERE_KEYS = {
    "pause": "space", "play": "space", "stop": "space", "space": "space",
    "left": "left", "back": "left", "frame_back": "left", "previous": "left",
    "right": "right", "forward": "right", "frame_forward": "right", "next": "right",
    "cut": "cmd+k", "razor": "cmd+k", "add_edit": "cmd+k",   # razor at playhead
    "cut_all_tracks": "cmd+shift+k",
    "undo": "cmd+z", "redo": "cmd+shift+z", "save": "cmd+s",
    "mark_in": "i", "mark_out": "o", "add_marker": "m",
    "ripple_delete": "shift+delete", "delete": "delete",
    "zoom_in": "shift+equal", "zoom_out": "minus",
}
# Actions where `count` repeats the key (frame stepping).
_PREMIERE_REPEATABLE = {"left", "back", "frame_back", "previous",
                        "right", "forward", "frame_forward", "next"}


def _premiere_window_bounds():
    """(x, y, w, h) of Premiere's largest on-screen window, via CoreGraphics."""
    try:
        import Quartz  # pyobjc (present in the project venv)
    except ImportError:
        return None
    opts = (Quartz.kCGWindowListOptionOnScreenOnly
            | Quartz.kCGWindowListExcludeDesktopElements)
    best, best_area = None, 0.0
    for info in (Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID) or []):
        if "Premiere" not in (info.get("kCGWindowOwnerName") or ""):
            continue
        b = info.get("kCGWindowBounds")
        if not b or float(b["Width"]) < 200:
            continue
        area = float(b["Width"]) * float(b["Height"])
        if area > best_area:
            best_area = area
            best = (float(b["X"]), float(b["Y"]), float(b["Width"]), float(b["Height"]))
    return best


def _premiere_focus_panel() -> bool:
    """Click the Program Monitor's video area to give a transport panel keyboard
    focus. Clicking the image is side-effect-free (no playhead move, no button).
    Tunable for non-default layouts via PREMIERE_FOCUS_X / PREMIERE_FOCUS_Y."""
    bounds = _premiere_window_bounds()
    if not bounds:
        return False
    x, y, w, h = bounds
    fx = float(os.environ.get("PREMIERE_FOCUS_X", "0.72"))  # upper-right ≈ Program Monitor
    fy = float(os.environ.get("PREMIERE_FOCUS_Y", "0.30"))
    _ad(["mouse-click", "--xy", f"{int(x + w * fx)},{int(y + h * fy)}"])
    time.sleep(0.15)
    return True


def premiere_control(action: str = "pause", count: int = 1) -> dict:
    """
    Control Adobe Premiere Pro. Transport: 'pause'|'play'|'stop' (toggle),
    'left'/'right' step the playhead `count` frames. Editing: 'cut' (razor at
    playhead), 'cut_all_tracks', 'undo', 'redo', 'save', 'mark_in', 'mark_out',
    'add_marker', 'ripple_delete', 'delete', 'zoom_in', 'zoom_out'.

    Premiere exposes nothing to the accessibility tree, so we drive it with its
    native keyboard shortcuts. The catch: those keys only land when the Timeline
    or Program Monitor panel has keyboard focus — `activate` alone isn't enough.
    So we click the Program Monitor to focus it, then send the key via
    agent-desktop (a real CGEvent, steadier than System Events keystrokes).
    """
    combo = _PREMIERE_KEYS.get(action)
    if combo is None:
        return {"status": "error", "error": f"unknown premiere action: {action}",
                "known": sorted(set(_PREMIERE_KEYS))}
    # find the running Premiere process name (version-agnostic)
    p = _osa(
        'tell application "System Events" to get name of (first process whose '
        'name contains "Premiere")'
    )
    proc = (p.stdout or "").strip()
    if not proc:
        return {"status": "error", "error": "Premiere Pro is not running"}
    _osa(f'tell application "{proc}" to activate')
    time.sleep(0.3)
    focused = _premiere_focus_panel()  # the reliability fix
    try:
        n = max(1, min(int(count), 240))  # clamp to a sane range
    except (TypeError, ValueError):
        n = 1
    reps = n if action in _PREMIERE_REPEATABLE else 1
    for _ in range(reps):
        _ad(["press", "--app", proc, combo])
        time.sleep(0.03)
    return {"status": "ok", "app": proc, "action": action, "key": combo,
            "frames": reps, "panel_focused": focused}


def start_obs_recording() -> dict:
    """Open OBS and start recording via the OBS WebSocket (reliable)."""
    _ensure_obs()
    try:
        r = _obs_call([("StartRecord", {})])
        ok = r[0]["requestStatus"]["result"]
        # code 500 = already recording; treat as success
        code = r[0]["requestStatus"].get("code")
        return {"status": "ok" if (ok or code == 500) else "error",
                "action": "recording", "detail": r[0]["requestStatus"]}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}


def stop_obs_recording() -> dict:
    """Stop OBS recording via the OBS WebSocket."""
    _ensure_obs()
    try:
        r = _obs_call([("StopRecord", {})])
        return {"status": "ok", "action": "stopped", "detail": r[0]["requestStatus"]}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}


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


WEB_BROWSER = "Arc"  # which browser web_search opens in


def web_search(query: str = "") -> dict:
    """Search the web for `query` in a NEW TAB of the EXISTING Arc window (don't
    spawn a new browser), then bring Arc to the front."""
    from urllib.parse import quote

    url = f"https://www.google.com/search?q={quote(query or '')}"
    # open in the current Arc window as a new tab (reuses Pat's open window/space)
    script = (
        f'tell application "{WEB_BROWSER}" to tell front window '
        f'to make new tab with properties {{URL:"{url}"}}'
    )
    p = _osa(script)
    if p.returncode != 0:  # no window open / not scriptable -> fall back
        _run(["open", "-a", WEB_BROWSER, url])
    _osa(f'tell application "{WEB_BROWSER}" to activate')
    return {"status": "ok", "query": query, "url": url, "browser": WEB_BROWSER}


def click_link(position: str = "first") -> dict:
    """Click a search result in the active Arc tab (e.g. 'click the first link').
    Uses JavaScript to jump to the Nth organic Google result (below the AI
    overview) — reliable, unlike clicking the rendered page via accessibility."""
    idx = {"first": 0, "1": 0, "one": 0, "top": 0,
           "second": 1, "2": 1, "two": 1,
           "third": 2, "3": 2, "three": 2}.get(str(position).lower().strip(), 0)
    js = (
        "(function(){var ls=document.querySelectorAll('#rso a:has(h3), #search a:has(h3)');"
        "if(!ls.length)ls=document.querySelectorAll('a:has(h3)');"
        "var a=ls[%d]||ls[0];if(!a)return 'no-result';"
        "window.location.href=a.href;return 'ok:'+a.href;})()" % idx
    )
    p = _osa(
        f'tell application "Arc" to tell active tab of front window '
        f'to execute javascript "{js}"'
    )
    out = (p.stdout or "").strip().strip('"')
    if out.startswith("ok:"):
        return {"status": "ok", "opened": out[3:], "position": position}
    return {"status": "error", "error": out or "no result link found"}


def take_note(text: str = "") -> dict:
    """Save a note. Creates a note in Apple Notes (visible on screen); falls back
    to ~/voice-notes.txt if Notes isn't available."""
    note = (text or "").strip()
    safe = note.replace("\\", "").replace('"', '\\"')
    script = (
        'tell application "Notes"\n'
        '  activate\n'
        f'  make new note with properties {{body:"{safe}"}}\n'
        'end tell'
    )
    p = _osa(script)
    if p.returncode == 0:
        return {"status": "ok", "note": note, "saved_to": "Apple Notes"}
    try:
        with open(os.path.expanduser("~/voice-notes.txt"), "a") as f:
            f.write(note + "\n")
        return {"status": "ok", "note": note, "saved_to": "~/voice-notes.txt"}
    except OSError as e:
        return {"status": "error", "error": str(e)}


# tool registry the voice agent imports
TOOLS = {
    "open_app": open_app,
    "web_search": web_search,
    "click_link": click_link,
    "take_note": take_note,
    "play_music": play_music,
    "run_terminal": run_terminal,
    "read_screen_aloud": read_screen_aloud,
    "start_obs_recording": start_obs_recording,
    "stop_obs_recording": stop_obs_recording,
    "obs_scene": obs_scene,
    "premiere_control": premiere_control,
    "ask_claude": ask_claude,
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
