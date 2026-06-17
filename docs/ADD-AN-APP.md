# Teach your agent a new app

This voice OS can already **open any app** out of the box (just say "open Apple
Music"). This guide is for when you want it to do something *specific* inside an
app — "play my Discover Weekly," "skip this track," "start a new note" — for an
app we didn't pre-build. You're going to add a **tool**. It's ~15 lines, and you
can even have your coding agent (Claude Code / Cursor) do it for you using the
steps below.

> The Python modules referenced below (`actions.py`, `voice_agent.py`, …) live in
> the **`src/`** directory.

---

## First, the one idea that makes this possible: the accessibility tree

Every Mac app publishes a hidden, structured outline of its window — "here's a
button called *Play*, here's a search field, here's a menu called *File*." It
exists so screen readers can describe apps to blind users. It's been in macOS
for decades.

Our agent's "hands" — a free tool called **agent-desktop** — read that same tree.
So the AI sees an app as a tidy list of buttons and fields (as JSON) instead of
staring at pixels. **That's why this works on almost any app without anyone
writing a special integration for it.** If you can click it, the agent can
usually find it in the tree and click it too.

There are two ways to drive an app. Use whichever is easier for your app:

| Way | When to use | Tool |
|-----|-------------|------|
| **Accessibility tree** | Works on almost any app. Click buttons, read text, fill fields. | `agent-desktop` |
| **AppleScript** | Some apps (Music, Notes, Spotify, Finder…) have a clean "scripting dictionary" — often the simplest path for media control. | `osascript` |

---

## The 3-step recipe to add an app

### Step 1 — look at the app through the tree
Open the app, then ask agent-desktop what it sees:

```bash
agent-desktop launch "Music"
agent-desktop snapshot --app "Music" -i --compact
```

You'll get JSON with the interactive elements and **refs** like `@e5`. Find the
thing you want (e.g. a "Play" button). You can click it directly to confirm:

```bash
agent-desktop click @e5            # did it play? great, you found your action
```

(Prefer AppleScript? Check if the app is scriptable: open **Script Editor →
File → Open Dictionary** and pick the app. If it lists commands like `play`,
you can script it.)

### Step 2 — add a tool function in `actions.py`
A tool is just a Python function that does the thing and returns a small dict.
Copy this template and adapt it:

```python
def play_apple_music(query: str = "") -> dict:
    """Play music in Apple Music. Empty query = resume; otherwise play that."""
    open_app("Music")
    time.sleep(1.2)
    if query:
        from urllib.parse import quote
        _run(["open", f"music://search?term={quote(query)}"])
        time.sleep(1.2)
    # Apple Music is AppleScript-scriptable:
    _osa('tell application "Music" to play')
    now = _osa('tell application "Music" to get name of current track')
    return {"status": "ok", "now_playing": (now.stdout or "").strip(), "query": query}
```

If your app isn't scriptable, do it through the tree instead — the helpers are
already in `actions.py`:

```python
def skip_track_in_someapp() -> dict:
    _ad(["launch", "SomeApp"]); time.sleep(0.8)
    res = _ad(["find", "--role", "button", "--app", "SomeApp"])
    ref = _find_button(res.get("data", {}), ("next", "skip"))
    if ref:
        _ad(["click", ref, "--snapshot", res.get("data", {}).get("snapshot_id", "")])
        return {"status": "ok", "action": "skipped"}
    return {"status": "error", "error": "no skip button found"}
```

Then register it in the `TOOLS` dict at the bottom of `actions.py`:

```python
TOOLS = {
    # ...existing tools...
    "play_apple_music": play_apple_music,
}
```

### Step 3 — tell the model the tool exists (`voice_agent.py`)
Add a matching entry to the `TOOLS` list near the top of `voice_agent.py` so
gpt-realtime-2 knows it can call it and what it does:

```python
{
    "type": "function",
    "name": "play_apple_music",
    "description": "Play music in Apple Music. Pass a search query, or empty to resume.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": [],
    },
},
```

Restart the agent. Now say (or hold-to-talk): **"play some jazz in Apple
Music."** Done.

---

## The fast way: let your coding agent do it

Open this repo in Claude Code (or Cursor) and say:

> "Add a tool for **Apple Music** to control playback. Follow `ADD-AN-APP.md`:
> snapshot the app with agent-desktop to find the controls, write the tool
> function in `actions.py`, register it in both the `actions.py` TOOLS dict and
> the `voice_agent.py` TOOLS list. Use AppleScript if the app is scriptable."

Because the agent can run `agent-desktop snapshot` itself, it will literally look
at the app, find the right buttons, and write the tool for you.

---

## Case study: a *hard* app (Adobe Premiere Pro)

Some apps — Adobe's especially — draw their own custom UI and publish **almost
nothing** to the accessibility tree. Snapshot Premiere and you get basically an
empty tree:

```bash
agent-desktop find --app "Adobe Premiere Pro 2025" --role button --count   # → 0
```

Zero buttons. So "snapshot the tree, click the Play button" is **impossible** here.
When you hit this, fall back to the app's **native keyboard shortcuts** — space =
play/pause, ←/→ = step a frame, ⌘K = razor cut at the playhead, etc.

But there's a gotcha that makes naive keyboard sending flaky, and it's worth
knowing because it bites every Adobe app:

> **Transport keys only land when the right panel has keyboard focus.**
> Bringing Premiere to the front with `activate` is *not* enough — if the Project
> bin or an effect field had focus, the spacebar goes nowhere. That's the classic
> "it pauses sometimes and not others." (Premiere also reports **0 windows to
> System Events**, so you can't even enumerate its panels that way.)

**The fix (what `premiere_control` does):**

1. `activate` Premiere.
2. **Click the Program Monitor's video area to force a transport panel into
   focus.** Clicking the *image* is side-effect-free — no playhead move, no button
   toggle. Get the window rectangle from CoreGraphics (`Quartz.CGWindowListCopyWindowInfo`,
   which *does* see Premiere's windows even though System Events doesn't), then
   click ~72% across / ~30% down (upper-right ≈ Program Monitor in the default
   layout; tune with `PREMIERE_FOCUS_X` / `PREMIERE_FOCUS_Y`).
3. Send the key with `agent-desktop press --app "<proc>" <combo>` — a real CGEvent,
   steadier than `osascript ... keystroke` (which can silently no-op if the
   controlling process lacks Accessibility).

Adding a new Premiere command is then **one line** — drop it in the `_PREMIERE_KEYS`
map in `actions.py`:

```python
_PREMIERE_KEYS = {
    "cut": "cmd+k",            # razor / add edit at the playhead
    "undo": "cmd+z", "redo": "cmd+shift+z", "save": "cmd+s",
    "mark_in": "i", "mark_out": "o", "add_marker": "m",
    "ripple_delete": "shift+delete", "zoom_in": "shift+equal", "zoom_out": "minus",
    # add yours here ↓
}
```

Then add the action name to the `enum` in `voice_agent.py`. Say **"cut"** and it
razors at the playhead.

**Takeaway for any hard app:** if the AX tree is empty, switch to keyboard
shortcuts, and remember that keyboard input needs the *correct inner panel*
focused — a single safe focus-click usually fixes the "works sometimes" problem.

---

## Tips
- **Naming:** keep tool names short and verb-y (`open_app`, `play_music`). The
  model matches your spoken intent to the tool `description`, so write clear
  descriptions.
- **Reliability on camera:** AppleScript/`open` URLs are steadier than clicking
  tree refs (refs can change between snapshots). Prefer them when available.
- **Permissions:** agent-desktop needs **Accessibility** permission (System
  Settings → Privacy & Security → Accessibility). Grant it once.
- **Test a tool without voice:** every tool runs standalone —
  `python src/actions.py play_apple_music "miles davis"`.
