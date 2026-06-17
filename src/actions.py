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

import config

AGENT_DESKTOP = shutil.which("agent-desktop") or "agent-desktop"

CLAUDE_LOG = config.CLAUDE_LOG
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


def _ax_collect_text(node, acc: list) -> None:
    """Recursively collect all string values from an accessibility-tree node."""
    if isinstance(node, dict):
        for k in ("name", "value", "title", "text"):
            v = node.get(k)
            if isinstance(v, str):
                acc.append(v)
        for v in node.values():
            _ax_collect_text(v, acc)
    elif isinstance(node, list):
        for v in node:
            _ax_collect_text(v, acc)


def _ax_node_matches(
    node: dict,
    allowed_roles,
    name_lower: str | None,
    sub_lower: str | None,
    needles: list[str] | None,
) -> bool:
    """Return True if node satisfies all provided match criteria."""
    if allowed_roles and node.get("role") not in allowed_roles:
        return False
    if name_lower and (node.get("name") or "").strip().lower() != name_lower:
        return False
    if sub_lower or needles:
        acc: list[str] = []
        _ax_collect_text(node, acc)
        flat = " ".join(acc).lower()
        if sub_lower and sub_lower not in flat:
            return False
        if needles and not any(nd in flat for nd in needles):
            return False
    return True


def _tree_find(
    node,
    *,
    role: str | None = None,
    roles: tuple | None = None,
    name: str | None = None,
    subtext: str | None = None,
    needles: list[str] | None = None,
) -> str | None:
    """Generic accessibility-tree search. Returns the first matching ref_id.

    Matching rules (all provided must hold):
      role / roles  — element role (role is shorthand for roles=(role,))
      name          — element's 'name' attribute == name (exact, case-insensitive)
      subtext       — subtree text contains subtext (case-insensitive)
      needles       — any needle string appears in combined name/title/value
    """
    if role:
        allowed_roles: tuple | None = (role,)
    elif roles:
        allowed_roles = tuple(roles)
    else:
        allowed_roles = None
    name_lower = name.lower() if name else None
    sub_lower = subtext.lower() if subtext else None

    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, dict):
            ref = n.get("ref_id")
            if ref and _ax_node_matches(n, allowed_roles, name_lower, sub_lower, needles):
                return ref
            stack.extend(n.values())
        elif isinstance(n, list):
            stack.extend(n)
    return None


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
    "premiere": config.PREMIERE_APP,
    "premiere pro": config.PREMIERE_APP,
    "obs studio": "OBS",
    "spotify app": "Spotify",
}


def open_app(name: str) -> dict:
    """Launch or focus a macOS app by name and bring it to the FRONT."""
    name = APP_ALIASES.get((name or "").strip().lower(), name)
    _ad(["launch", name])  # best-effort via agent-desktop
    # `activate` + `open -a` are the reliable foreground path; we use both
    # because agent-desktop alone sometimes leaves the current app on top.
    _osa(f'tell application "{name}" to activate')
    p = _run(["open", "-a", name])
    ok = p.returncode == 0
    if not ok:
        return {
            "status": "error",
            "error": f"Could not open '{name}' — app may not be installed or the name is wrong.",
            "app": name,
        }
    return {"status": "ok", "app": name}


def _load_favorites() -> dict:
    """Load Spotify favorites from VOICEOS_SPOTIFY_FAVORITES JSON file.
    Returns empty dict if unset or file is missing/malformed."""
    path = config.SPOTIFY_FAVORITES_FILE
    if not path:
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {k.lower(): v for k, v in data.items()}
    except FileNotFoundError:
        return {}
    except Exception as e:  # noqa: BLE001
        print(f"[voice-os] warning: could not load favorites from {path}: {e}", flush=True)
        return {}


FAVORITES: dict = _load_favorites()


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
    Types the command and presses Return so it runs live on camera.
    """
    if not shutil.which("claude"):
        return {
            "status": "error",
            "error": "'claude' CLI not found in PATH. Install it with: npm install -g @anthropic-ai/claude-code",
        }
    safe = prompt.replace('"', '\\"')
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
    ref = _tree_find(data, name=name)
    if ref and sid:
        _ad(["click", ref, "--snapshot", sid])
        return True
    return False


def _find_link_by_subtext(data: dict, needle: str):
    """Find a Claude project card: a `link` whose SUBTREE text contains needle."""
    return _tree_find(data, role="link", subtext=needle)


def _claude_in_project(name: str) -> bool:
    """True if Claude is ALREADY showing the named project (so we can skip the
    navigation + its visible bounce). Requires VOICEOS_CLAUDE_PROJECT_HINT to be
    set to a short unique phrase from the project's system prompt."""
    hint = config.CLAUDE_PROJECT_HINT.strip().lower()
    if not hint:
        return False  # no hint configured — always navigate to be safe
    blob = json.dumps(_claude_snapshot()).lower()
    if '"name": "new chat - claude"' in blob or '"name": "projects - claude"' in blob:
        return False
    return hint in blob


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
    ref = _tree_find(data, role="textfield")
    if ref and sid:
        _ad(["click", ref, "--snapshot", sid])
        time.sleep(0.4)


def _claude_all_text() -> list:
    acc: list[str] = []
    _ax_collect_text(_claude_snapshot(), acc)
    return [t.strip() for t in acc if t.strip()]


def _read_claude_response(timeout: float = 25.0) -> str:
    """Poll Claude's accessibility tree until the response finishes.

    Markers used (Claude Desktop Electron accessibility labels):
      - start:  any text node begins with "Claude responded:"
      - finish: any text node contains "finished the response"
    Poll interval starts at 0.5 s and backs off to 1.0 s after 6 polls
    to reduce the number of expensive full-tree snapshots at the end.
    """
    deadline = time.time() + timeout
    last = ""
    n = 0
    while time.time() < deadline:
        _force_electron_ax("Claude")
        texts = _claude_all_text()
        responded = [t[len("Claude responded:"):].strip()
                     for t in texts if t.startswith("Claude responded:")]
        if responded:
            last = responded[-1]
        if last and any("finished the response" in t.lower() for t in texts):
            _clog(f"read: finished after {n} polls -> {last[:50]!r}")
            return last
        n += 1
        time.sleep(0.5 if n < 6 else 1.0)
    _clog(f"read: TIMED OUT after {n} polls, last={last[:50]!r}")
    return last


def ask_claude(question: str = "",
               project: str = "") -> dict:
    """
    Open the YouTube Script project in Claude Desktop, type the question into the
    project's compose box on screen, send it, and read Claude's ACTUAL reply back
    for chat to speak. The project's instructions make Claude answer with the
    locked script. Navigation: force the accessibility tree on, click Projects,
    click the project card (matched by its child text), type, send, read.
    """
    _clog_t0[0] = time.monotonic()
    q = (question or "").strip()
    project = (project or config.CLAUDE_PROJECT).strip()
    _clog(f"ask_claude START — project={project!r} question={q[:50]!r}")
    open_app("Claude")  # bring Claude Desktop to the front
    time.sleep(1.5)
    _force_electron_ax("Claude")
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
    _osa("tell application \"System Events\" to key code 36")
    _clog("sent; waiting for Claude's reply…")
    reply = _read_claude_response(timeout=22.0)
    _clog(f"ask_claude DONE in {time.monotonic()-_clog_t0[0]:.1f}s  in_project={in_project}")
    if not reply:
        return {
            "status": "error",
            "error": "Claude did not produce a readable response within the timeout.",
            "project_opened": in_project,
            "question": q,
        }
    return {
        "status": "ok",
        "project_opened": in_project,
        "question": q,
        "response": reply,
    }


def _find_clickable_by_subtext(data, needle, roles=("cell", "row", "button", "link")):
    """Find an element of `roles` whose SUBTREE text contains needle."""
    return _tree_find(data, roles=tuple(roles), subtext=needle)


# ---- OBS control via its built-in WebSocket (reliable; not tree-dependent) ----
OBS_WS_CONFIG = os.path.expanduser(
    "~/Library/Application Support/obs-studio/plugin_config/obs-websocket/config.json"
)


def _obs_password() -> str:
    try:
        with open(OBS_WS_CONFIG) as f:
            return json.load(f).get("server_password", "")
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


_KEY_RAZOR = "cmd+k"  # Premiere: razor/add-edit at playhead

# Premiere actions → key combos. Add a line to teach a new editing command.
_PREMIERE_KEYS = {
    "pause": "space", "play": "space", "stop": "space", "space": "space",
    "left": "left", "back": "left", "frame_back": "left", "previous": "left",
    "right": "right", "forward": "right", "frame_forward": "right", "next": "right",
    "cut": _KEY_RAZOR, "razor": _KEY_RAZOR, "add_edit": _KEY_RAZOR,
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
    acc: list[str] = []
    _ax_collect_text(node, acc)
    stripped = [s.strip() for s in acc if s.strip()]
    seen: list[str] = []
    for s in stripped:
        if not seen or seen[-1] != s:
            seen.append(s)
    return "\n".join(seen)


def _find_button(node, needles) -> str | None:
    return _tree_find(node, needles=list(needles))


WEB_BROWSER = config.WEB_BROWSER


def web_search(query: str = "") -> dict:
    """Search the web for `query` in a NEW TAB of the EXISTING browser window (don't
    spawn a new window), then bring the browser to the front."""
    from urllib.parse import quote

    url = f"https://www.google.com/search?q={quote(query or '')}"
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
    """Click a search result in the active browser tab.
    Uses JavaScript to jump to the Nth organic Google result (below the AI
    overview) — reliable, unlike clicking the rendered page via accessibility."""
    _BROWSER = WEB_BROWSER
    _JS_CAPABLE = {"Arc", "Google Chrome", "Safari", "Microsoft Edge", "Brave Browser"}
    if _BROWSER not in _JS_CAPABLE:
        return {
            "status": "error",
            "error": f"click_link does not support browser '{_BROWSER}'. "
                     f"Supported: {sorted(_JS_CAPABLE)}.",
        }
    idx = {"first": 0, "1": 0, "one": 0, "top": 0,
           "second": 1, "2": 1, "two": 1,
           "third": 2, "3": 2, "three": 2}.get(str(position).lower().strip(), 0)
    js = (
        "(function(){var ls=document.querySelectorAll('#rso a:has(h3), #search a:has(h3)');"
        "if(!ls.length)ls=document.querySelectorAll('a:has(h3)');"
        "var a=ls[%d]||ls[0];if(!a)return 'no-result';"
        "window.location.href=a.href;return 'ok:'+a.href;})()" % idx
    )
    if _BROWSER == "Safari":
        script = (
            f'tell application "Safari" to do JavaScript "{js}" '
            f'in current tab of front window'
        )
    else:
        script = (
            f'tell application "{_BROWSER}" to tell active tab of front window '
            f'to execute javascript "{js}"'
        )
    p = _osa(script)
    out = (p.stdout or "").strip().strip('"')
    if out.startswith("ok:"):
        return {"status": "ok", "opened": out[3:], "position": position}
    err = out or "no result link found — make sure a Google search is open in the browser"
    return {"status": "error", "error": err}


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


# ---------------------------------------------------------------------------
# The 5 primitives — these are the only tools exposed to the model.
# The recipe functions above become their implementation layer.
# The model composes these based on capabilities retrieved per-turn.
# ---------------------------------------------------------------------------

def primitive_run_applescript(script: str) -> dict:
    """Execute arbitrary AppleScript. Use for any app that has a scripting
    dictionary (Spotify, Notes, Finder, Terminal, System Events, etc.)."""
    if not script or not script.strip():
        return {"status": "error", "error": "script must not be empty"}
    p = _osa(script.strip())
    ok = p.returncode == 0
    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()
    return {
        "status": "ok" if ok else "error",
        "output": out or None,
        "error": err[:300] if not ok else None,
    }


def primitive_press_key(combo: str, app: str, repeat: int = 1) -> dict:
    """Send a keyboard shortcut to a named macOS app via CGEvent.
    combo examples: 'space', 'cmd+k', 'cmd+shift+z', 'left'.
    repeat sends the key N times (for frame-stepping etc.)."""
    if not combo:
        return {"status": "error", "error": "combo must not be empty"}
    if not app:
        return {"status": "error", "error": "app name required"}
    # find running process (version-agnostic name match)
    p = _osa(
        f'tell application "System Events" to get name of '
        f'(first process whose name contains "{app}")'
    )
    proc = (p.stdout or "").strip()
    if not proc:
        return {"status": "error", "error": f"'{app}' is not running"}
    _osa(f'tell application "{proc}" to activate')
    time.sleep(0.25)
    n = max(1, min(int(repeat), 240))
    for _ in range(n):
        result = _ad(["press", "--app", proc, combo])
        time.sleep(0.03)
    ok = result.get("ok", True)  # agent-desktop returns ok on success
    return {
        "status": "ok" if ok else "error",
        "app": proc,
        "combo": combo,
        "repeat": n,
    }


def primitive_read_screen(app: str = "Terminal") -> dict:
    """Read the text currently visible in a macOS app via the accessibility
    tree. Returns the last ~800 chars of visible text so the model can
    summarise or speak it."""
    _ad(["launch", app])
    time.sleep(0.4)
    res = _ad(["snapshot", "--app", app, "--compact"], timeout=25)
    text = _extract_text(res.get("data", {}))
    spoken = text[-800:].strip() if text else ""
    if not spoken:
        return {"status": "error", "error": f"no readable text found in '{app}'"}
    return {"status": "ok", "app": app, "text": spoken}


def primitive_open_url(url: str) -> dict:
    """Open a URL in the configured browser. Use for web searches, docs,
    any http/https URL. Construct Google search URLs as:
    https://www.google.com/search?q=<url-encoded-query>"""
    if not url or not url.startswith(("http://", "https://", "spotify:", "x-apple")):
        return {"status": "error", "error": f"invalid or unsupported URL scheme: {url!r}"}
    browser = WEB_BROWSER
    script = (
        f'tell application "{browser}" to tell front window '
        f'to make new tab with properties {{URL:"{url}"}}'
    )
    p = _osa(script)
    if p.returncode != 0:
        _run(["open", "-a", browser, url])
    _osa(f'tell application "{browser}" to activate')
    return {"status": "ok", "url": url, "browser": browser}


def primitive_obs_call(request_type: str, request_data: dict | None = None) -> dict:
    """Send a request to the OBS WebSocket API.
    Common request_types: StartRecord, StopRecord, GetSceneList,
    SetCurrentProgramScene (requestData: {sceneName: "..."}).
    OBS must be running with WebSocket server enabled (Tools → WebSocket Server)."""
    _ensure_obs()
    try:
        results = _obs_call([(request_type, request_data or {})])
        status = results[0].get("requestStatus", {})
        ok = status.get("result", False)
        # code 500 = already in that state (e.g. already recording) — treat as ok
        if not ok and status.get("code") == 500:
            ok = True
        return {
            "status": "ok" if ok else "error",
            "requestType": request_type,
            "response": results[0].get("responseData"),
            "error": None if ok else status.get("comment", "OBS request failed"),
        }
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}


# tool registry — 5 primitives exposed to the model
TOOLS = {
    "run_applescript": primitive_run_applescript,
    "press_key": primitive_press_key,
    "read_screen": primitive_read_screen,
    "open_url": primitive_open_url,
    "obs_call": primitive_obs_call,
}

# Legacy registry — used by the standalone CLI tester only
_LEGACY_TOOLS = {
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
    _all = {**TOOLS, **_LEGACY_TOOLS}
    if len(sys.argv) < 2 or sys.argv[1] not in _all:
        print("usage: python actions.py <tool> [args...]")
        print("primitives:", ", ".join(TOOLS))
        print("recipes:   ", ", ".join(_LEGACY_TOOLS))
        sys.exit(1)
    fn = _all[sys.argv[1]]
    kwargs = {}
    rest = sys.argv[2:]
    if rest:
        # positional -> first param name
        import inspect

        params = list(inspect.signature(fn).parameters)
        kwargs[params[0]] = " ".join(rest)
    print(json.dumps(fn(**kwargs), indent=2))
