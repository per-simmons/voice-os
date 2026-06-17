"""
session_log.py — structured per-session event log for voice-os.

Every session writes a JSONL file to memory/sessions/<ISO-timestamp>.jsonl.
Events are structured (not just text) so the retrospective can reason over them.

Event types:
  heard        — transcript received from the model
  wake         — wake word detected, response triggered
  ignored      — transcript received but wake word absent
  tool_call    — a tool was dispatched
  spoken       — model spoke a response (transcript.done)
  error        — an error event from the API
  reconnect    — session reconnected (60-min cap or network drop)
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

# src/ lives one level under the project root; memory/ sits at the root.
_SESSIONS_DIR = Path(__file__).resolve().parent.parent / "memory" / "sessions"


class SessionLog:
    def __init__(self, user: str = "default"):
        _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        self._path = _SESSIONS_DIR / f"{ts}.jsonl"
        self._user = user
        self._start = time.time()
        self._write({"event": "session_start", "user": user})

    # ------------------------------------------------------------------
    # Public log methods
    # ------------------------------------------------------------------

    def heard(self, transcript: str) -> None:
        self._write({"event": "heard", "transcript": transcript})

    def wake(self, transcript: str) -> None:
        self._write({"event": "wake", "transcript": transcript})

    def ignored(self, transcript: str) -> None:
        self._write({"event": "ignored", "transcript": transcript})

    def tool_call(self, name: str, args: dict, result: dict, latency: float) -> None:
        self._write({
            "event": "tool_call",
            "name": name,
            "args": args,
            "result": result,
            "latency_s": round(latency, 3),
            "ok": result.get("status") == "ok",
        })

    def spoken(self, text: str) -> None:
        self._write({"event": "spoken", "text": text})

    def error(self, detail: dict) -> None:
        self._write({"event": "error", "detail": detail})

    def reconnect(self) -> None:
        self._write({"event": "reconnect"})

    def close(self) -> None:
        duration = round(time.time() - self._start, 1)
        self._write({"event": "session_end", "duration_s": duration})

    # ------------------------------------------------------------------
    # Path accessor (for retrospective)
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(self, event: dict) -> None:
        event["t"] = round(time.time(), 3)
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Class-level helpers for reading past sessions
    # ------------------------------------------------------------------

    @classmethod
    def list_sessions(cls, limit: int = 50) -> list[Path]:
        """Return paths to the most recent session files, newest first."""
        if not _SESSIONS_DIR.exists():
            return []
        files = sorted(_SESSIONS_DIR.glob("*.jsonl"), reverse=True)
        return files[:limit]

    @classmethod
    def read_session(cls, path: Path) -> list[dict]:
        events = []
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except Exception:  # noqa: BLE001
            pass
        return events

    @classmethod
    def successful_tool_calls(cls, sessions: int = 5) -> list[dict]:
        """Return all successful tool_call events from the last N sessions."""
        calls = []
        for path in cls.list_sessions(limit=sessions):
            for ev in cls.read_session(path):
                if ev.get("event") == "tool_call" and ev.get("ok"):
                    calls.append(ev)
        return calls
