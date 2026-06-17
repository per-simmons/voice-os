"""
retrospective.py — post-session "dreaming" loop for voice-os.

After a session ends, this module:
  1. Reads the session log (structured JSONL)
  2. Asks the LLM to identify reusable patterns in what was said + done
  3. Writes new capability entries to memory/capabilities.user.json
  4. Triggers a retrieval index refresh so new patterns are live next session

This is the "dreaming" idea: the OS consolidates episodic memory (what happened
this session) into semantic memory (generalised, retrievable capabilities).

Only successful tool calls (status: ok) become templates — the system learns
what works, not what failed.

Run automatically at session end, or manually:
    python retrospective.py                  # reflect on last session
    python retrospective.py --sessions 3     # reflect on last 3 sessions
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from session_log import SessionLog

# src/ lives one level under the project root; memory/ sits at the root.
_MEMORY_DIR = Path(__file__).resolve().parent.parent / "memory"
_USER_CAPS_PATH = _MEMORY_DIR / "capabilities.user.json"
_BUILTIN_CAPS_PATH = _MEMORY_DIR / "capabilities.json"

# Strips a leading wake word ("hey chat, ...") so learned phrasings match the
# wake-word-free style of the builtin examples used at retrieval time.
_WAKE_STRIP = re.compile(r"^\s*(hey|hay|hi|ok|okay|yo)\s+\w+\s*[,.]?\s*", re.IGNORECASE)

_RETROSPECTIVE_PROMPT = """\
You are the long-term memory of a voice-controlled macOS assistant. You are shown
(1) the capabilities it ALREADY knows and (2) what the user actually said this
session alongside the tool that ran successfully.

Your goal: make the assistant respond INSTANTLY next time. There are two moves:

A) ADD PHRASINGS (preferred). If a successful command was just a new way of asking
   for something already in KNOWN CAPABILITIES, return that capability's EXACT id
   with the user's new phrasing(s) in "examples". This teaches retrieval to match
   that phrasing next time. Omit "template" when reusing an existing id.

B) CREATE a capability. Only for a genuinely new action that nothing existing covers.

Rules:
- Strongly prefer A. Reuse an existing id whenever the intent already exists.
- Use the user's ACTUAL spoken phrasings as examples — that is what makes matching work.
- Never list an existing id as if it were new.
- For a NEW capability give a "description", 3-8 varied "examples", a "primitive",
  and the exact "template" (use {placeholders} for variable parts).

Primitives available:
  run_applescript(script)            — arbitrary AppleScript
  press_key(combo, app, repeat?)     — CGEvent keystroke, optional repeat count
  read_screen(app)                   — accessibility tree text
  open_url(url)                      — open URL in browser
  obs_call(requestType, requestData) — OBS WebSocket call

Output a JSON object: {"updates": [ ... ]}. Each update object:
{
  "id": "existing-or-new-slug",
  "examples": ["the user's phrasing", "another phrasing"],
  "primitive": "run_applescript | press_key | read_screen | open_url | obs_call",
  "template": "ONLY for a new id",
  "description": "ONLY for a new id"
}
If nothing is worth remembering, return {"updates": []}.

KNOWN CAPABILITIES:
{existing}

SESSION LOG ('said' is what the user spoke; only successful commands shown):
{log}
"""


def _strip_wake(text: str) -> str:
    return _WAKE_STRIP.sub("", text or "").strip()


def _collect_turns(paths: list[Path]) -> list[dict]:
    """Pair each successful tool call with the phrasing the user spoke for it.

    Walks events in order, tracking the most recent heard/wake transcript, and
    attaches it to the tool call(s) it triggered. The spoken phrasing is the
    signal the retrospective learns from."""
    turns: list[dict] = []
    for path in paths:
        last_query: str | None = None
        for ev in SessionLog.read_session(path):
            event = ev.get("event")
            if event in ("wake", "heard"):
                q = _strip_wake(ev.get("transcript") or "")
                if q:
                    last_query = q
            elif event == "tool_call" and ev.get("ok"):
                turns.append({
                    "query": last_query,
                    "name": ev.get("name"),
                    "args": ev.get("args", {}),
                    "result": ev.get("result", {}),
                })
    return turns


def _format_turns_for_prompt(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        args_str = json.dumps(t.get("args", {}), ensure_ascii=False)
        said = t.get("query") or "(unknown)"
        lines.append(f'  said="{said}" → {t.get("name")}({args_str})')
    return "\n".join(lines) if lines else "(no successful commands)"


def _format_existing_for_prompt(caps: list[dict]) -> str:
    lines = []
    for c in caps:
        ex = "; ".join(c.get("examples", [])[:4])
        lines.append(f"  [{c['id']}] {c.get('description', '')} — e.g. {ex}")
    return "\n".join(lines) if lines else "(none)"


def _load_caps(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001
        return []


def _load_user_caps() -> list[dict]:
    return _load_caps(_USER_CAPS_PATH)


def _save_user_caps(caps: list[dict]) -> None:
    _USER_CAPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_USER_CAPS_PATH, "w") as f:
        json.dump(caps, f, indent=2, ensure_ascii=False)


def _merge_updates(
    user_caps: list[dict],
    builtin_by_id: dict[str, dict],
    updates: list[dict],
) -> tuple[list[dict], int, int]:
    """Apply learned updates to the user-overlay capability list.

      - id already in the user file -> union new example phrasings in place
      - id is a builtin             -> create an overlay (full builtin copy +
                                       new phrasings) so the original template
                                       and examples are preserved
      - brand-new id                -> add as a learned capability

    Returns (merged_user_caps, new_capability_count, new_example_count).
    """
    by_id = {c["id"]: c for c in user_caps}
    new_caps = 0
    new_examples = 0

    def _union(target: dict, examples: list[str]) -> None:
        nonlocal new_examples
        have = {e.lower().strip() for e in target.get("examples", [])}
        for ex in examples:
            key = ex.lower().strip()
            if key and key not in have:
                target.setdefault("examples", []).append(ex)
                have.add(key)
                new_examples += 1

    for up in updates:
        cid = (up.get("id") or "").strip()
        examples = [e for e in up.get("examples", []) if isinstance(e, str)]
        if not cid:
            continue
        if cid in by_id:
            _union(by_id[cid], examples)
        elif cid in builtin_by_id:
            overlay = dict(builtin_by_id[cid])
            overlay["examples"] = list(overlay.get("examples", []))
            overlay["source"] = "learned"
            _union(overlay, examples)
            by_id[cid] = overlay
        else:
            # genuinely new: a usable capability needs a primitive + template
            if not up.get("primitive") or not up.get("template"):
                continue
            by_id[cid] = {
                "id": cid,
                "description": up.get("description", ""),
                "examples": examples,
                "primitive": up["primitive"],
                "template": up["template"],
                "source": "learned",
            }
            new_caps += 1

    return list(by_id.values()), new_caps, new_examples


def _parse_updates(parsed: object) -> list[dict]:
    """Normalise the model's JSON into a flat list of update objects, tolerating
    {"updates": [...]}, {"capabilities": [...]}, or a bare [...]."""
    if isinstance(parsed, dict):
        updates = parsed.get("updates") or parsed.get("capabilities") or parsed.get("items") or []
        if not updates:
            for v in parsed.values():
                if isinstance(v, list):
                    updates = v
                    break
        return updates if isinstance(updates, list) else []
    return parsed if isinstance(parsed, list) else []


def run_retrospective(
    session_log_path: Path | None = None,
    sessions: int = 1,
    verbose: bool = True,
) -> int:
    """
    Run the retrospective. Returns the number of new capabilities added.
    If session_log_path is given, use that session; otherwise use the last N.
    """
    if session_log_path:
        paths = [session_log_path]
    else:
        paths = SessionLog.list_sessions(limit=sessions)

    if not paths:
        if verbose:
            print("[retrospective] no sessions found", flush=True)
        return 0

    # pair each successful tool call with the phrasing the user spoke for it
    turns = _collect_turns(paths)
    if not turns:
        if verbose:
            print("[retrospective] no successful commands to reflect on", flush=True)
        return 0

    if verbose:
        print(f"[retrospective] reflecting on {len(turns)} successful command(s) "
              f"across {len(paths)} session(s)…", flush=True)

    # call the LLM
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("[retrospective] OPENAI_API_KEY not set — skipping", flush=True)
        return 0

    builtin_caps = _load_caps(_BUILTIN_CAPS_PATH)
    user_caps = _load_user_caps()
    builtin_by_id = {c["id"]: c for c in builtin_caps}
    # user overrides builtin by id, for the "known capabilities" view
    existing_view = list({**builtin_by_id, **{c["id"]: c for c in user_caps}}.values())

    # NB: plain .replace, not .format — the prompt contains literal { } in its
    # JSON example, which str.format would try to parse as fields.
    prompt = (
        _RETROSPECTIVE_PROMPT
        .replace("{existing}", _format_existing_for_prompt(existing_view))
        .replace("{log}", _format_turns_for_prompt(turns))
    )

    try:
        import urllib.request

        body = json.dumps({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.4,
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read())
        if verbose:
            print(f"[retrospective] LLM responded in {time.monotonic()-t0:.1f}s", flush=True)

        content = raw["choices"][0]["message"]["content"]
        updates = _parse_updates(json.loads(content))

    except Exception as e:  # noqa: BLE001
        print(f"[retrospective] LLM call failed: {e}", flush=True)
        return 0

    if not updates:
        if verbose:
            print("[retrospective] nothing new worth remembering this session", flush=True)
        return 0

    merged, new_caps, new_examples = _merge_updates(user_caps, builtin_by_id, updates)
    _save_user_caps(merged)

    if verbose:
        print(f"[retrospective] {new_caps} new capability(s), {new_examples} new "
              f"phrasing(s) added to {_USER_CAPS_PATH}", flush=True)
        for up in updates:
            cid = (up.get("id") or "").strip()
            mark = "NEW" if cid not in builtin_by_id and cid not in {c["id"] for c in user_caps} else "UPD"
            exs = ", ".join(up.get("examples", [])[:3])
            print(f"  [{mark}] {cid}: +{exs}", flush=True)

    # refresh the in-process retrieval index if it's loaded
    try:
        import retrieval
        if retrieval._index is not None:
            retrieval._index.refresh()
            if verbose:
                print("[retrospective] retrieval index refreshed", flush=True)
    except Exception:  # noqa: BLE001
        pass

    return new_caps + new_examples


if __name__ == "__main__":
    import argparse
    from pathlib import Path as _Path

    # load .env if present (.env lives at the project root, one level above src/)
    env_path = _Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    parser = argparse.ArgumentParser(description="voice-os retrospective / dreaming loop")
    parser.add_argument("--sessions", type=int, default=1,
                        help="number of recent sessions to reflect on (default: 1)")
    parser.add_argument("--session-file", type=str, default=None,
                        help="path to a specific session JSONL file")
    args = parser.parse_args()

    path = Path(args.session_file) if args.session_file else None
    added = run_retrospective(session_log_path=path, sessions=args.sessions, verbose=True)
    print(f"\n[retrospective] done — {added} new capabilities added")
