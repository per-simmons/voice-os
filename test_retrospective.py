#!/usr/bin/env python3
"""
test_retrospective.py — unit tests for the dreaming/retrospective loop.

These are pure (no network, no model): they exercise the logic that turns a
session log into memory updates — pairing spoken phrasings with tool calls,
stripping the wake word, and the smart merge that either appends a new phrasing
to an existing template or mints a brand-new capability.

    pytest test_retrospective.py
"""
from __future__ import annotations

import json

import retrospective as r


# --- wake-word stripping --------------------------------------------------

def test_strip_wake_removes_leading_wake_word():
    assert r._strip_wake("hey chat, open spotify") == "open spotify"
    assert r._strip_wake("Hey Chat play music") == "play music"


def test_strip_wake_leaves_plain_commands_untouched():
    assert r._strip_wake("pause the music") == "pause the music"
    assert r._strip_wake("") == ""


# --- pairing a spoken phrase with the tool it triggered -------------------

def test_collect_turns_pairs_query_with_tool_and_skips_failures(tmp_path):
    log = tmp_path / "session.jsonl"
    events = [
        {"event": "session_start", "user": "x"},
        {"event": "wake", "transcript": "hey chat, open spotify"},
        {"event": "tool_call", "name": "run_applescript", "args": {"script": "a"},
         "result": {"status": "ok"}, "ok": True},
        {"event": "wake", "transcript": "hey chat, do the broken thing"},
        {"event": "tool_call", "name": "run_applescript", "args": {"script": "b"},
         "result": {"status": "error"}, "ok": False},      # failed -> ignored
        {"event": "heard", "transcript": "pause"},
        {"event": "tool_call", "name": "run_applescript", "args": {"script": "c"},
         "result": {"status": "ok"}, "ok": True},
    ]
    log.write_text("\n".join(json.dumps(e) for e in events))

    turns = r._collect_turns([log])

    assert [t["query"] for t in turns] == ["open spotify", "pause"]   # wake stripped, failure dropped
    assert [t["name"] for t in turns] == ["run_applescript", "run_applescript"]


# --- the smart merge ------------------------------------------------------

def test_merge_appends_phrasing_to_existing_user_cap():
    user = [{"id": "my-thing", "description": "d", "examples": ["foo"],
             "primitive": "open_url", "template": "http://x", "source": "learned"}]
    merged, new_caps, new_examples = r._merge_updates(
        user, {}, [{"id": "my-thing", "examples": ["bar"]}]
    )
    cap = next(c for c in merged if c["id"] == "my-thing")
    assert new_caps == 0 and new_examples == 1
    assert cap["examples"] == ["foo", "bar"]


def test_merge_overlays_builtin_preserving_template_and_examples():
    builtin = {"app-open": {
        "id": "app-open", "description": "open an app",
        "examples": ["open Spotify", "launch Chrome"],
        "primitive": "run_applescript", "template": "tell application ...",
    }}
    merged, new_caps, new_examples = r._merge_updates(
        [], builtin, [{"id": "app-open", "examples": ["fire up the spotify app"]}]
    )
    overlay = next(c for c in merged if c["id"] == "app-open")
    assert new_caps == 0 and new_examples == 1
    # the original template + examples survive — we only ADD the new phrasing
    assert overlay["template"] == "tell application ..."
    assert "open Spotify" in overlay["examples"]
    assert "fire up the spotify app" in overlay["examples"]
    assert overlay["source"] == "learned"


def test_merge_dedups_phrasings_case_insensitively():
    builtin = {"app-open": {"id": "app-open", "description": "d",
                            "examples": ["Open Spotify"],
                            "primitive": "run_applescript", "template": "t"}}
    _, _, new_examples = r._merge_updates(
        [], builtin, [{"id": "app-open", "examples": ["open spotify"]}]
    )
    assert new_examples == 0   # already known, just different case


def test_merge_new_capability_requires_primitive_and_template():
    # missing template -> not usable, skipped
    merged, new_caps, _ = r._merge_updates([], {}, [{"id": "x", "examples": ["a"]}])
    assert new_caps == 0 and merged == []
    # complete -> added as a learned capability
    merged, new_caps, _ = r._merge_updates(
        [], {}, [{"id": "x", "examples": ["a"], "primitive": "open_url", "template": "http://x"}]
    )
    assert new_caps == 1 and merged[0]["source"] == "learned"


# --- response parsing + prompt building -----------------------------------

def test_parse_updates_tolerates_shapes():
    assert r._parse_updates({"updates": [1]}) == [1]
    assert r._parse_updates({"capabilities": [2]}) == [2]
    assert r._parse_updates({"weird_key": [3]}) == [3]   # first list value
    assert r._parse_updates([4, 5]) == [4, 5]
    assert r._parse_updates("nonsense") == []


def test_prompt_builds_without_str_format_crash():
    # regression: the prompt has literal { } in its JSON example; str.format
    # would raise KeyError('\n  "id"'). We use .replace, which must not.
    prompt = (r._RETROSPECTIVE_PROMPT
              .replace("{existing}", "KNOWN")
              .replace("{log}", "SAID"))
    assert "KNOWN" in prompt and "SAID" in prompt
    assert '"id": "existing-or-new-slug"' in prompt


if __name__ == "__main__":
    # runnable without pytest: execute every test_* with a tmp dir where needed
    import tempfile
    from pathlib import Path

    failures = 0
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"  ok  {name}")
        except AssertionError as e:  # noqa: PERF203
            failures += 1
            print(f"FAIL  {name}: {e}")
    raise SystemExit(1 if failures else 0)
