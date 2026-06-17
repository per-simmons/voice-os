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
import sys
import time
from pathlib import Path

_USER_CAPS_PATH = Path(__file__).parent / "memory" / "capabilities.user.json"

_RETROSPECTIVE_PROMPT = """\
You are the memory system for a voice-controlled macOS assistant.
Below is a log of what the user said and what tools were called in a recent session.
Your job: identify 1-5 reusable patterns worth remembering as new capabilities.

Rules:
- Only create entries for things that WORKED (they are already filtered to successful calls).
- Focus on patterns that are likely to recur — personal workflows, compound actions, shortcuts.
- Do NOT duplicate capabilities that are already obvious generic ones (open app, web search, etc).
- Each entry must have 3-8 varied example phrasings covering how a user might say it.
- The template should be the exact primitive invocation needed.

Primitives available:
  run_applescript(script)          — arbitrary AppleScript
  press_key(combo, app, repeat?)   — CGEvent keystroke, optional repeat count
  read_screen(app)                 — accessibility tree text
  open_url(url)                    — open URL in browser
  obs_call(requestType, requestData) — OBS WebSocket call

Output a JSON array of new capability objects. Each object:
{
  "id": "unique-slug",
  "description": "one sentence describing what it does",
  "examples": ["spoken phrase 1", "spoken phrase 2", ...],
  "primitive": "run_applescript | press_key | read_screen | open_url | obs_call",
  "template": "the script/key/url/args — use {placeholders} for variable parts"
}

If you find no reusable patterns worth remembering, return an empty array: []

Session log (successful tool calls only):
{log}
"""


def _format_log_for_prompt(calls: list[dict]) -> str:
    lines = []
    for c in calls:
        args_str = json.dumps(c.get("args", {}), ensure_ascii=False)
        result_str = json.dumps(c.get("result", {}), ensure_ascii=False)[:120]
        lines.append(f"  [{c.get('name')}] args={args_str} → {result_str}")
    return "\n".join(lines) if lines else "(no successful tool calls)"


def _load_user_caps() -> list[dict]:
    if not _USER_CAPS_PATH.exists():
        return []
    try:
        with open(_USER_CAPS_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001
        return []


def _save_user_caps(caps: list[dict]) -> None:
    _USER_CAPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_USER_CAPS_PATH, "w") as f:
        json.dump(caps, f, indent=2, ensure_ascii=False)


def _merge_caps(existing: list[dict], new: list[dict]) -> tuple[list[dict], int]:
    """Merge new caps into existing, overwriting by id. Returns (merged, added_count)."""
    by_id = {c["id"]: c for c in existing}
    added = 0
    for cap in new:
        cap["source"] = "learned"
        if cap["id"] not in by_id:
            added += 1
        by_id[cap["id"]] = cap
    return list(by_id.values()), added


def run_retrospective(
    session_log_path: Path | None = None,
    sessions: int = 1,
    verbose: bool = True,
) -> int:
    """
    Run the retrospective. Returns the number of new capabilities added.
    If session_log_path is given, use that session; otherwise use the last N.
    """
    from session_log import SessionLog

    if session_log_path:
        paths = [session_log_path]
    else:
        paths = SessionLog.list_sessions(limit=sessions)

    if not paths:
        if verbose:
            print("[retrospective] no sessions found", flush=True)
        return 0

    # collect successful tool calls across sessions
    all_calls: list[dict] = []
    for path in paths:
        for ev in SessionLog.read_session(path):
            if ev.get("event") == "tool_call" and ev.get("ok"):
                all_calls.append(ev)

    if not all_calls:
        if verbose:
            print("[retrospective] no successful tool calls to reflect on", flush=True)
        return 0

    if verbose:
        print(f"[retrospective] reflecting on {len(all_calls)} successful calls "
              f"across {len(paths)} session(s)…", flush=True)

    # call the LLM
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("[retrospective] OPENAI_API_KEY not set — skipping", flush=True)
        return 0

    import config
    log_str = _format_log_for_prompt(all_calls)
    prompt = _RETROSPECTIVE_PROMPT.format(log=log_str)

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
        parsed = json.loads(content)
        # handle both {"capabilities": [...]} and bare [...]
        if isinstance(parsed, dict):
            new_caps = parsed.get("capabilities") or parsed.get("items") or []
            # fallback: first list value in the dict
            if not new_caps:
                for v in parsed.values():
                    if isinstance(v, list):
                        new_caps = v
                        break
        else:
            new_caps = parsed if isinstance(parsed, list) else []

    except Exception as e:  # noqa: BLE001
        print(f"[retrospective] LLM call failed: {e}", flush=True)
        return 0

    if not new_caps:
        if verbose:
            print("[retrospective] no new patterns identified this session", flush=True)
        return 0

    existing = _load_user_caps()
    merged, added = _merge_caps(existing, new_caps)
    _save_user_caps(merged)

    if verbose:
        print(f"[retrospective] {added} new capability/capabilities added "
              f"to {_USER_CAPS_PATH}", flush=True)
        for cap in new_caps:
            marker = "NEW" if cap["id"] not in {c["id"] for c in existing} else "UPD"
            print(f"  [{marker}] {cap['id']}: {cap['description']}", flush=True)

    # refresh the in-process retrieval index if it's loaded
    try:
        import retrieval
        if retrieval._index is not None:
            retrieval._index.refresh()
            if verbose:
                print("[retrospective] retrieval index refreshed", flush=True)
    except Exception:  # noqa: BLE001
        pass

    return added


if __name__ == "__main__":
    import argparse
    from pathlib import Path as _Path

    # load .env if present
    env_path = _Path(__file__).parent / ".env"
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
