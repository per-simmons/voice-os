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

# OpenWakeWord — built-in options: "hey_jarvis", "hey_mycroft", "alexa"
# Set to an absolute path to load a custom .onnx / .tflite model.
OWW_MODEL: str = os.environ.get("VOICEOS_OWW_MODEL", "hey_jarvis")
# Detection threshold: lower = more sensitive (more false triggers); higher = stricter.
OWW_THRESHOLD: float = float(os.environ.get("VOICEOS_OWW_THRESHOLD", "0.5"))
OWW_FRAME_MS: int = 80
OWW_FRAME: int = CAP_RATE * OWW_FRAME_MS // 1_000  # 1 280 samples

# Whisper model size for command transcription (after OWW fires).
# Options: tiny.en, base.en, small.en  — base.en is the default sweet-spot.
WHISPER_SIZE: str = os.environ.get("VOICEOS_WHISPER", "base.en")

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
