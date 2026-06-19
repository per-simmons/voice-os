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
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import retrospective as r  # noqa: E402


# ---------------------------------------------------------------------------
# Wake-word stripping
# ---------------------------------------------------------------------------

def test_strip_wake_removes_leading_wake_word():
    assert r._strip_wake("hey chat, open spotify") == "open spotify"
    assert r._strip_wake("Hey Chat play music") == "play music"


def test_strip_wake_handles_variant_openers():
    assert r._strip_wake("ok chat, pause the music") == "pause the music"
    assert r._strip_wake("yo chat resume") == "resume"
    assert r._strip_wake("hi chat open terminal") == "open terminal"


def test_strip_wake_leaves_plain_commands_untouched():
    assert r._strip_wake("pause the music") == "pause the music"
    assert r._strip_wake("") == ""


def test_strip_wake_handles_comma_and_period_separators():
    assert r._strip_wake("hey chat. open Spotify") == "open Spotify"
    assert r._strip_wake("hey chat, play some jazz") == "play some jazz"


# ---------------------------------------------------------------------------
# Pairing a spoken phrase with the tool it triggered
# ---------------------------------------------------------------------------

def _make_log(events: list[dict]) -> "Path":
    raise NotImplementedError("use the tmp_path fixture version below")


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

    assert [t["query"] for t in turns] == ["open spotify", "pause"]
    assert [t["name"] for t in turns] == ["run_applescript", "run_applescript"]


def test_collect_turns_from_multiple_session_files(tmp_path):
    log_a = tmp_path / "a.jsonl"
    log_b = tmp_path / "b.jsonl"
    log_a.write_text("\n".join(json.dumps(e) for e in [
        {"event": "wake", "transcript": "hey chat, open spotify"},
        {"event": "tool_call", "name": "run_applescript", "args": {}, "result": {"status": "ok"}, "ok": True},
    ]))
    log_b.write_text("\n".join(json.dumps(e) for e in [
        {"event": "heard", "transcript": "pause"},
        {"event": "tool_call", "name": "run_applescript", "args": {}, "result": {"status": "ok"}, "ok": True},
    ]))

    turns = r._collect_turns([log_a, log_b])
    assert len(turns) == 2
    assert turns[0]["query"] == "open spotify"
    assert turns[1]["query"] == "pause"


def test_collect_turns_one_query_triggers_multiple_tool_calls(tmp_path):
    log = tmp_path / "session.jsonl"
    events = [
        {"event": "wake", "transcript": "hey chat, open spotify and play jazz"},
        {"event": "tool_call", "name": "run_applescript", "args": {"script": "open"},
         "result": {"status": "ok"}, "ok": True},
        {"event": "tool_call", "name": "run_applescript", "args": {"script": "play"},
         "result": {"status": "ok"}, "ok": True},
    ]
    log.write_text("\n".join(json.dumps(e) for e in events))

    turns = r._collect_turns([log])
    # both tool calls are attributed to the same spoken phrase
    assert len(turns) == 2
    assert turns[0]["query"] == "open spotify and play jazz"
    assert turns[1]["query"] == "open spotify and play jazz"


def test_collect_turns_with_empty_session(tmp_path):
    log = tmp_path / "empty.jsonl"
    log.write_text("")
    turns = r._collect_turns([log])
    assert turns == []


def test_collect_turns_with_only_non_tool_events(tmp_path):
    log = tmp_path / "session.jsonl"
    events = [
        {"event": "session_start", "user": "x"},
        {"event": "heard", "transcript": "hello"},
        {"event": "session_end", "duration_s": 10.0},
    ]
    log.write_text("\n".join(json.dumps(e) for e in events))
    assert r._collect_turns([log]) == []


def test_collect_turns_tolerates_corrupt_session_file(tmp_path):
    log = tmp_path / "bad.jsonl"
    log.write_text("this is not json at all\n{bad json too\n")
    turns = r._collect_turns([log])
    assert turns == []


def test_collect_turns_recovers_valid_turn_around_corrupt_line(tmp_path):
    # a corrupt line in the middle must not drop the valid command that follows
    log = tmp_path / "mixed.jsonl"
    log.write_text(
        json.dumps({"event": "wake", "transcript": "hey chat, open spotify"}) + "\n"
        "GARBAGE LINE\n"
        + json.dumps({"event": "tool_call", "name": "run_applescript",
                      "args": {}, "result": {"status": "ok"}, "ok": True}) + "\n"
    )
    turns = r._collect_turns([log])
    assert len(turns) == 1
    assert turns[0]["query"] == "open spotify"


def test_collect_turns_tool_without_preceding_query_has_none_query(tmp_path):
    log = tmp_path / "session.jsonl"
    events = [
        {"event": "tool_call", "name": "run_applescript", "args": {},
         "result": {"status": "ok"}, "ok": True},
    ]
    log.write_text("\n".join(json.dumps(e) for e in events))
    turns = r._collect_turns([log])
    # tool call with no preceding heard/wake -> query is None, which _strip_wake handles
    assert len(turns) == 1
    assert turns[0]["query"] is None


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def test_format_turns_for_prompt_includes_said_and_tool(tmp_path):
    turns = [
        {"query": "pause the music", "name": "run_applescript",
         "args": {"script": "tell app..."}, "result": {"status": "ok"}},
    ]
    output = r._format_turns_for_prompt(turns)
    assert 'said="pause the music"' in output
    assert "run_applescript" in output


def test_format_turns_for_prompt_with_no_turns():
    output = r._format_turns_for_prompt([])
    assert "(no successful commands)" in output


def test_format_existing_for_prompt_includes_id_and_description():
    caps = [{"id": "app-open", "description": "Launch an app", "examples": ["open Spotify"]}]
    output = r._format_existing_for_prompt(caps)
    assert "app-open" in output
    assert "Launch an app" in output


def test_format_existing_for_prompt_with_no_caps():
    output = r._format_existing_for_prompt([])
    assert "(none)" in output


# ---------------------------------------------------------------------------
# The smart merge
# ---------------------------------------------------------------------------

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
    assert new_examples == 0


def test_merge_new_capability_requires_primitive_and_template():
    # missing template -> not usable, skipped
    merged, new_caps, _ = r._merge_updates([], {}, [{"id": "x", "examples": ["a"]}])
    assert new_caps == 0 and merged == []
    # complete -> added as a learned capability
    merged, new_caps, _ = r._merge_updates(
        [], {}, [{"id": "x", "examples": ["a"], "primitive": "open_url", "template": "http://x"}]
    )
    assert new_caps == 1 and merged[0]["source"] == "learned"


def test_merge_empty_id_is_skipped():
    merged, new_caps, new_examples = r._merge_updates(
        [], {}, [{"id": "", "examples": ["something"]}]
    )
    assert new_caps == 0 and new_examples == 0 and merged == []


def test_merge_multiple_updates_to_same_id_accumulate():
    user = [{"id": "thing", "description": "d", "examples": ["a"],
             "primitive": "open_url", "template": "http://x", "source": "learned"}]
    merged, _, new_examples = r._merge_updates(
        user, {},
        [
            {"id": "thing", "examples": ["b"]},
            {"id": "thing", "examples": ["c"]},
        ]
    )
    cap = next(c for c in merged if c["id"] == "thing")
    assert new_examples == 2
    assert set(cap["examples"]) == {"a", "b", "c"}


def test_merge_preserves_unrelated_user_caps():
    user = [
        {"id": "cap-a", "description": "d", "examples": ["a"], "primitive": "open_url", "template": "t"},
        {"id": "cap-b", "description": "d", "examples": ["b"], "primitive": "open_url", "template": "t"},
    ]
    merged, _, _ = r._merge_updates(user, {}, [{"id": "cap-a", "examples": ["new"]}])
    ids = {c["id"] for c in merged}
    assert "cap-b" in ids


def test_merge_update_with_no_examples_adds_zero():
    user = [{"id": "thing", "description": "d", "examples": ["a"],
             "primitive": "open_url", "template": "http://x", "source": "learned"}]
    _, _, new_examples = r._merge_updates(user, {}, [{"id": "thing", "examples": []}])
    assert new_examples == 0


# ---------------------------------------------------------------------------
# Response parsing + prompt building
# ---------------------------------------------------------------------------

def test_parse_updates_tolerates_shapes():
    assert r._parse_updates({"updates": [1]}) == [1]
    assert r._parse_updates({"capabilities": [2]}) == [2]
    assert r._parse_updates({"weird_key": [3]}) == [3]   # first list value
    assert r._parse_updates([4, 5]) == [4, 5]
    assert r._parse_updates("nonsense") == []


def test_parse_updates_empty_dict_returns_empty():
    assert r._parse_updates({}) == []


def test_prompt_builds_without_str_format_crash():
    prompt = (r._RETROSPECTIVE_PROMPT
              .replace("{existing}", "KNOWN")
              .replace("{log}", "SAID"))
    assert "KNOWN" in prompt and "SAID" in prompt
    assert '"id": "existing-or-new-slug"' in prompt


if __name__ == "__main__":
    # runnable without pytest
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
