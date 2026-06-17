"""
config.py — single source of truth for all Voice OS tuneable values.

Every value can be overridden via an environment variable (or .env).
Import this module; do not duplicate constants across files.
"""
from __future__ import annotations

import os
import tempfile

# ---------------------------------------------------------------------------
# OpenAI / model
# ---------------------------------------------------------------------------
MODEL: str = os.environ.get("VOICEOS_MODEL", "gpt-realtime-2")
VOICE: str = os.environ.get("VOICEOS_VOICE", "marin")
URL: str = f"wss://api.openai.com/v1/realtime?model={MODEL}"

# Input transcription model (drives wake-word detection + command parsing).
# whisper-1 is fast but mishears short wake words; gpt-4o-transcribe and
# gpt-4o-mini-transcribe are markedly more accurate. Pick the bigger model if
# the wake word / commands are getting mistranscribed:
#   VOICEOS_TRANSCRIBE_MODEL=gpt-4o-transcribe        (best accuracy)
#   VOICEOS_TRANSCRIBE_MODEL=gpt-4o-mini-transcribe   (cheaper, still good)
#   VOICEOS_TRANSCRIBE_MODEL=whisper-1                (fastest, least accurate)
TRANSCRIBE_MODEL: str = os.environ.get("VOICEOS_TRANSCRIBE_MODEL", "gpt-4o-transcribe")
TRANSCRIBE_LANGUAGE: str = os.environ.get("VOICEOS_TRANSCRIBE_LANGUAGE", "en")

# ---------------------------------------------------------------------------
# Audio — output (shared by both entry points)
# ---------------------------------------------------------------------------
OUT_RATE: int = 24_000          # gpt-realtime-2 output sample rate (Hz)
OUT_CHANNELS: int = 1
OUT_BLOCK: int = 4_800          # ~200 ms at 24 kHz
PRIME_BYTES: int = OUT_RATE * 2 * 300 // 1_000  # 300 ms pre-roll before playback

# ---------------------------------------------------------------------------
# Audio — capture (voice_agent.py cloud path)
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 24_000       # cloud mode streams at 24 kHz
BLOCK: int = 2_400              # ~100 ms at 24 kHz

# ---------------------------------------------------------------------------
# Audio — capture (wake_listener.py local path)
# ---------------------------------------------------------------------------
CAP_RATE: int = 16_000          # OWW + Whisper both want 16 kHz
FRAME_MS: int = 80              # OWW requires multiples of 80 ms
FRAME: int = CAP_RATE * FRAME_MS // 1_000       # 1 280 samples
SILENCE_TAIL_MS: int = 450      # silence duration that ends a command utterance
MAX_UTTER_MS: int = 6_000       # hard cap on a single utterance

# ---------------------------------------------------------------------------
# Wake word
# ---------------------------------------------------------------------------
WAKE_WORD: str = os.environ.get("VOICEOS_WAKE_WORD", "hey chat")

# Biasing hint fed to the transcriber. Seeding the wake word (and any unusual app
# names you use) measurably cuts mishears of short commands. Set to "" to disable.
TRANSCRIBE_PROMPT: str = os.environ.get(
    "VOICEOS_TRANSCRIBE_PROMPT",
    f'Commands to a Mac voice assistant. They usually start with the wake word "{WAKE_WORD}".',
)

# OpenWakeWord — built-in options: "hey_jarvis", "hey_mycroft", "alexa"
# Set to an absolute path to load a custom .onnx / .tflite model.
OWW_MODEL: str = os.environ.get("VOICEOS_OWW_MODEL", "hey_jarvis")
# Detection threshold: lower = more sensitive (more false triggers); higher = stricter.
OWW_THRESHOLD: float = float(os.environ.get("VOICEOS_OWW_THRESHOLD", "0.5"))
OWW_FRAME_MS: int = 80
OWW_FRAME: int = CAP_RATE * OWW_FRAME_MS // 1_000  # 1 280 samples

# Local Whisper (faster-whisper) model for command transcription AFTER the wake
# word fires. This runs on-device and is only invoked per-command, so a bigger,
# more accurate model costs CPU briefly but NOTHING at idle — pick a bigger one
# if commands get mistranscribed. Roughly smallest→best:
#   tiny.en  base.en  small.en  medium.en  distil-large-v3  large-v3
# small.en is a good accuracy/speed default on Apple Silicon (int8).
WHISPER_SIZE: str = os.environ.get("VOICEOS_WHISPER", "small.en")

# ---------------------------------------------------------------------------
# Microphone selection
# ---------------------------------------------------------------------------
# Substring match against sounddevice device names (e.g. "Scarlett", "MacBook").
MIC_NAME: str | None = os.environ.get("VOICEOS_MIC")

# ---------------------------------------------------------------------------
# Logging / IPC
# ---------------------------------------------------------------------------
_tmp = tempfile.gettempdir()
EVENT_LOG: str = os.path.join(_tmp, "voiceos-events.log")
CLAUDE_LOG: str = os.path.join(_tmp, "voiceos-claude.log")
HUD_FILE: str = os.path.join(_tmp, "voiceos-hud.json")

# ---------------------------------------------------------------------------
# Browser / external tools
# ---------------------------------------------------------------------------
# Default browser used by web_search (must be AppleScript-scriptable on macOS).
WEB_BROWSER: str = os.environ.get("VOICEOS_BROWSER", "Safari")

# ---------------------------------------------------------------------------
# User / persona (keep personal values in .env, not in source)
# ---------------------------------------------------------------------------
# The user's name — used in the system prompt so the model addresses them correctly.
USER_NAME: str = os.environ.get("VOICEOS_USER_NAME", "the user")

# Free-text hints appended to the system prompt, e.g. accent, preferences.
# Example: "The user has a New Zealand accent."
USER_HINTS: str = os.environ.get("VOICEOS_USER_HINTS", "")

# ---------------------------------------------------------------------------
# App versions (override when you upgrade)
# ---------------------------------------------------------------------------
# Exact macOS app name for Adobe Premiere Pro (changes each year).
PREMIERE_APP: str = os.environ.get("VOICEOS_PREMIERE_APP", "Adobe Premiere Pro")

# ---------------------------------------------------------------------------
# Spotify favorites (keep personal track URIs in .env, not in source)
# ---------------------------------------------------------------------------
# Path to a JSON file mapping spoken phrases to spotify:track:<id> URIs.
# Format: {"phrase": "spotify:track:ABC123", ...}
# If unset, only generic search is used.
SPOTIFY_FAVORITES_FILE: str = os.environ.get("VOICEOS_SPOTIFY_FAVORITES", "")

# ---------------------------------------------------------------------------
# Claude Desktop project
# ---------------------------------------------------------------------------
# The name of the Claude Desktop project ask_claude navigates into.
CLAUDE_PROJECT: str = os.environ.get("VOICEOS_CLAUDE_PROJECT", "")

# A short unique phrase from the project's system prompt that confirms we're
# already inside the right project (Claude Desktop accessibility tree check).
# If unset, the in-project check is skipped and navigation always runs.
CLAUDE_PROJECT_HINT: str = os.environ.get("VOICEOS_CLAUDE_PROJECT_HINT", "")
