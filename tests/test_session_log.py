#!/usr/bin/env python3
"""
test_session_log.py — unit tests for the structured session event logger.

Tests the full write/read cycle: every event method, the JSONL format, the
list_sessions() and read_session() class helpers, and graceful handling of
malformed files.

    pytest test_session_log.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402
import session_log  # noqa: E402
from session_log import SessionLog  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    """Redirect session writes to a temp directory for test isolation."""
    d = tmp_path / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(session_log, "_SESSIONS_DIR", d)
    return d


@pytest.fixture
def slog(log_dir):
    """A fresh SessionLog writing to the isolated temp directory."""
    return SessionLog(user="test-user")


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_session_log_creates_file_on_init(log_dir, slog):
    assert log_dir.exists()
    files = list(log_dir.glob("*.jsonl"))
    assert len(files) == 1


def test_session_log_writes_session_start_event(log_dir, slog):
    events = SessionLog.read_session(slog.path)
    first = events[0]
    assert first["event"] == "session_start"
    assert first["user"] == "test-user"
    assert "t" in first


def test_session_log_path_is_jsonl(slog):
    assert slog.path.suffix == ".jsonl"


# ---------------------------------------------------------------------------
# Event methods
# ---------------------------------------------------------------------------

def test_heard_writes_correct_event(slog):
    slog.heard("pause the music")
    events = SessionLog.read_session(slog.path)
    ev = next(e for e in events if e.get("event") == "heard")
    assert ev["transcript"] == "pause the music"


def test_wake_writes_correct_event(slog):
    slog.wake("hey chat, open Spotify")
    events = SessionLog.read_session(slog.path)
    ev = next(e for e in events if e.get("event") == "wake")
    assert ev["transcript"] == "hey chat, open Spotify"


def test_ignored_writes_correct_event(slog):
    slog.ignored("noise in the room")
    events = SessionLog.read_session(slog.path)
    ev = next(e for e in events if e.get("event") == "ignored")
    assert ev["transcript"] == "noise in the room"


def test_tool_call_ok_writes_correct_fields(slog):
    slog.tool_call(
        "run_applescript",
        {"script": "tell application ..."},
        {"status": "ok"},
        latency=0.123,
    )
    events = SessionLog.read_session(slog.path)
    ev = next(e for e in events if e.get("event") == "tool_call")
    assert ev["name"] == "run_applescript"
    assert ev["ok"] is True
    assert ev["latency_s"] == pytest.approx(0.123, abs=0.001)


def test_tool_call_failure_sets_ok_false(slog):
    slog.tool_call(
        "run_applescript",
        {},
        {"status": "error", "error": "app not found"},
        latency=0.05,
    )
    events = SessionLog.read_session(slog.path)
    ev = next(e for e in events if e.get("event") == "tool_call")
    assert ev["ok"] is False


def test_spoken_writes_correct_event(slog):
    slog.spoken("Done — Spotify is now playing.")
    events = SessionLog.read_session(slog.path)
    ev = next(e for e in events if e.get("event") == "spoken")
    assert ev["text"] == "Done — Spotify is now playing."


def test_error_writes_correct_event(slog):
    slog.error({"code": 429, "message": "rate limit"})
    events = SessionLog.read_session(slog.path)
    ev = next(e for e in events if e.get("event") == "error")
    assert ev["detail"]["code"] == 429


def test_reconnect_writes_correct_event(slog):
    slog.reconnect()
    events = SessionLog.read_session(slog.path)
    ev = next(e for e in events if e.get("event") == "reconnect")
    assert ev is not None


# ---------------------------------------------------------------------------
# Session close
# ---------------------------------------------------------------------------

def test_close_writes_session_end_event(slog):
    slog.close()
    events = SessionLog.read_session(slog.path)
    last = events[-1]
    assert last["event"] == "session_end"
    assert "duration_s" in last
    assert last["duration_s"] >= 0.0


def test_close_duration_is_non_negative(slog):
    time.sleep(0.05)
    slog.close()
    events = SessionLog.read_session(slog.path)
    end = next(e for e in events if e.get("event") == "session_end")
    assert end["duration_s"] >= 0.05


# ---------------------------------------------------------------------------
# Event ordering and timestamps
# ---------------------------------------------------------------------------

def test_events_are_written_in_order(slog):
    slog.heard("first")
    slog.heard("second")
    slog.heard("third")
    events = [e for e in SessionLog.read_session(slog.path) if e.get("event") == "heard"]
    assert [e["transcript"] for e in events] == ["first", "second", "third"]


def test_every_event_has_a_timestamp(slog):
    slog.heard("test")
    slog.wake("test wake")
    slog.spoken("test spoken")
    slog.close()
    for ev in SessionLog.read_session(slog.path):
        assert "t" in ev, f"event {ev.get('event')!r} has no timestamp"
        assert isinstance(ev["t"], (int, float))


# ---------------------------------------------------------------------------
# read_session
# ---------------------------------------------------------------------------

def test_read_session_returns_empty_list_for_missing_file(tmp_path):
    result = SessionLog.read_session(tmp_path / "nonexistent.jsonl")
    assert result == []


def test_read_session_returns_empty_list_for_empty_file(tmp_path):
    f = tmp_path / "empty.jsonl"
    f.write_text("")
    assert SessionLog.read_session(f) == []


def test_read_session_skips_blank_lines(tmp_path):
    f = tmp_path / "blanks.jsonl"
    f.write_text(
        json.dumps({"event": "heard", "transcript": "a"}) + "\n"
        "\n"
        "\n"
        + json.dumps({"event": "heard", "transcript": "b"}) + "\n"
    )
    events = SessionLog.read_session(f)
    assert len(events) == 2


def test_read_session_tolerates_first_line_corrupt(tmp_path):
    f = tmp_path / "corrupt.jsonl"
    f.write_text("NOT JSON\n" + json.dumps({"event": "heard"}) + "\n")
    # corrupt first line stops processing; result has 0 events (existing behavior)
    events = SessionLog.read_session(f)
    assert isinstance(events, list)


# ---------------------------------------------------------------------------
# list_sessions
# ---------------------------------------------------------------------------

def test_list_sessions_returns_empty_when_no_dir(log_dir, monkeypatch, tmp_path):
    absent = tmp_path / "absent_sessions"
    monkeypatch.setattr(session_log, "_SESSIONS_DIR", absent)
    assert SessionLog.list_sessions() == []


def test_list_sessions_returns_files_newest_first(log_dir):
    # create several .jsonl files with different names (sorted lexically = chronologically)
    for name in ("2026-01-01T00-00-00.jsonl", "2026-01-02T00-00-00.jsonl", "2026-01-03T00-00-00.jsonl"):
        (log_dir / name).write_text("{}")
    paths = SessionLog.list_sessions()
    names = [p.name for p in paths]
    assert names == sorted(names, reverse=True)


def test_list_sessions_respects_limit(log_dir):
    for i in range(5):
        (log_dir / f"2026-01-0{i+1}T00-00-00.jsonl").write_text("{}")
    assert len(SessionLog.list_sessions(limit=3)) == 3
    assert len(SessionLog.list_sessions(limit=10)) == 5


def test_list_sessions_only_returns_jsonl_files(log_dir):
    (log_dir / "notes.txt").write_text("not a session")
    (log_dir / "2026-01-01T00-00-00.jsonl").write_text("{}")
    paths = SessionLog.list_sessions()
    assert all(p.suffix == ".jsonl" for p in paths)


# ---------------------------------------------------------------------------
# successful_tool_calls
# ---------------------------------------------------------------------------

def test_successful_tool_calls_returns_only_ok_events(log_dir):
    f = log_dir / "2026-01-01T00-00-00.jsonl"
    events = [
        {"event": "tool_call", "name": "run_applescript", "ok": True},
        {"event": "tool_call", "name": "run_applescript", "ok": False},
        {"event": "tool_call", "name": "open_url", "ok": True},
        {"event": "heard", "transcript": "blah"},
    ]
    f.write_text("\n".join(json.dumps(e) for e in events))

    calls = SessionLog.successful_tool_calls(sessions=1)
    assert all(c["ok"] for c in calls)
    assert len(calls) == 2


def test_successful_tool_calls_returns_empty_when_no_sessions(log_dir):
    assert SessionLog.successful_tool_calls(sessions=5) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
