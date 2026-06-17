#!/usr/bin/env python3
"""
wake_listener.py — LOCAL wake word, $0 idle.

Two-stage pipeline:
  1. OpenWakeWord runs a tiny neural net on every 80ms audio frame — sub-ms CPU,
     no network, scores the wake phrase continuously.
  2. When OWW fires, faster-whisper transcribes only the post-wake audio to extract
     the command text.  Whisper never sees audio unless the wake word already fired.

Pipeline:
  mic ─▶ OpenWakeWord (local, ~1ms/frame) ─▶ wake score > threshold?
        └ no  → ignored, $0                    │ yes
                                               ▼
                faster-whisper base.en (local) ─▶ command text
                                               ▼
              open gpt-realtime-2 ─▶ tool call ─▶ agent-desktop ─▶ speak

Run:  ./run.sh --local     (or)   python wake_listener.py [--mic Scarlett]
Ctrl-C to quit. First run downloads models once (~75MB Whisper + ~5MB OWW).
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
import sys
import threading
import time

try:
    import numpy as np
    import sounddevice as sd
    import websockets
    from faster_whisper import WhisperModel
    import openwakeword
    from openwakeword.model import Model as OWWModel
except ImportError as e:
    sys.exit(f"Missing dep ({e}). Run: ./run.sh --local  (installs local wake engine).")

import config
from voice_agent import (
    INSTRUCTIONS,
    MODEL,
    TOOLS,
    URL,
    dispatch_tool,
    is_wake,
)


def _arg_value(flag, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


MIC_NAME = _arg_value("--mic", config.MIC_NAME)
WHISPER_SIZE = config.WHISPER_SIZE

CAP_RATE = config.CAP_RATE
FRAME_MS = config.FRAME_MS
FRAME = config.FRAME
SILENCE_TAIL_MS = config.SILENCE_TAIL_MS
MAX_UTTER_MS = config.MAX_UTTER_MS

OWW_MODEL = config.OWW_MODEL
OWW_THRESHOLD = config.OWW_THRESHOLD
OWW_FRAME_MS = config.OWW_FRAME_MS
OWW_FRAME = config.OWW_FRAME
OUT_RATE = config.OUT_RATE
OUT_BLOCK = config.OUT_BLOCK
PRIME_BYTES = config.PRIME_BYTES
VOICE = config.VOICE
EVENT_LOG = config.EVENT_LOG

# ---- audio plumbing ----
cap_q: "queue.Queue[bytes]" = queue.Queue()   # 16k int16 frames from mic
cmd_q: "queue.Queue[str]" = queue.Queue()     # locally-detected wake commands
play_q: "queue.Queue[bytes]" = queue.Queue()  # 24k int16 from the model
_play_buf = bytearray()
_primed = False
_speaking = False  # model talking -> pause local capture so we don't hear it


def _log(m: str):
    try:
        with open(EVENT_LOG, "a") as f:
            f.write(m + "\n")
    except OSError:
        pass


def _mic_cb(indata, frames, t, status):
    cap_q.put(bytes(indata))


def _spk_cb(outdata, frames, t, status):
    global _primed
    need = len(outdata)
    while True:
        try:
            _play_buf.extend(play_q.get_nowait())
        except queue.Empty:
            break
    if not _primed:
        if len(_play_buf) >= PRIME_BYTES:
            _primed = True
        else:
            outdata[:] = b"\x00" * need
            return
    if len(_play_buf) >= need:
        outdata[:] = bytes(_play_buf[:need])
        del _play_buf[:need]
    else:
        n = len(_play_buf)
        outdata[:n] = bytes(_play_buf)
        outdata[n:] = b"\x00" * (need - n)
        del _play_buf[:]
        _primed = False


def _process_frame(
    frame: bytes,
    audio_f32,
    whisper: WhisperModel,
    oww: OWWModel,
    state: dict,
) -> None:
    """Process one 80ms frame through OWW (listening) or energy-VAD (awake)."""
    if _speaking:
        state["cmd_buf"].clear()
        state["awake"] = False
        state["silence_ms"] = 0
        state["pre_ring"].clear()
        oww.reset()
        return

    scores = oww.predict(audio_f32)
    best = max(scores.values()) if scores else 0.0

    if not state["awake"]:
        state["pre_ring"].append(frame)
        if len(state["pre_ring"]) > state["pre_frames"]:
            state["pre_ring"].pop(0)
        if best >= OWW_THRESHOLD:
            state["awake"] = True
            state["silence_ms"] = 0
            state["cmd_buf"].clear()
            state["cmd_buf"].extend(b"".join(state["pre_ring"]))
            print('\n🟢 wake word detected — say your command', flush=True)
            _log(f"OWW score={best:.3f}")
    else:
        state["cmd_buf"].extend(frame)
        energy = np.abs(audio_f32).mean()
        if energy < 0.005:
            state["silence_ms"] += OWW_FRAME_MS
        else:
            state["silence_ms"] = 0
        if state["silence_ms"] >= SILENCE_TAIL_MS or len(state["cmd_buf"]) >= state["max_bytes"]:
            _finalize(whisper, bytes(state["cmd_buf"]))
            state["cmd_buf"].clear()
            state["awake"] = False
            state["silence_ms"] = 0
            state["pre_ring"].clear()
            oww.reset()


def recognizer_thread(whisper: WhisperModel, oww: OWWModel):
    """Two-stage wake pipeline:
      Stage 1 — OWW scores every 80ms frame; fires when score > OWW_THRESHOLD.
      Stage 2 — once fired, collect audio until silence, then Whisper extracts cmd.
    """
    buf = bytearray()
    state = {
        "cmd_buf": bytearray(),
        "awake": False,
        "silence_ms": 0,
        "max_bytes": MAX_UTTER_MS * CAP_RATE // 1000 * 2,
        "pre_frames": 3,   # 3 × 80ms = 240ms pre-roll context
        "pre_ring": [],
    }

    while True:
        buf.extend(cap_q.get())
        while len(buf) >= OWW_FRAME * 2:
            frame = bytes(buf[: OWW_FRAME * 2])
            del buf[: OWW_FRAME * 2]
            audio_f32 = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
            _process_frame(frame, audio_f32, whisper, oww, state)


def _finalize(model: WhisperModel, pcm: bytes):
    """Transcribe post-wake audio with Whisper and push command to cmd_q."""
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size < CAP_RATE // 4:  # < ~0.25s -> nothing useful was said
        print('\n·  (no command heard after wake word)', flush=True)
        return
    segments, _ = model.transcribe(audio, language="en", beam_size=5)
    text = " ".join(s.text for s in segments).strip()
    if not text:
        print('\n·  (whisper returned empty after wake)', flush=True)
        return
    print(f"\n🗣  COMMAND → {text!r}", flush=True)
    _log(f"CMD {text!r}")
    cmd_q.put(text)


def session_config() -> dict:
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": MODEL,
            "instructions": INSTRUCTIONS,
            "output_modalities": ["audio"],
            "audio": {
                "input": {"format": {"type": "audio/pcm", "rate": OUT_RATE}},
                "output": {
                    "format": {"type": "audio/pcm", "rate": OUT_RATE},
                    "voice": VOICE,
                },
            },
            "tools": TOOLS,
            "tool_choice": "auto",
        },
    }


async def ws_session(ws):
    """One connection: send locally-detected commands as text, handle replies."""
    global _speaking
    await ws.send(json.dumps(session_config()))
    loop = asyncio.get_event_loop()

    async def feed():
        while True:
            cmd = await loop.run_in_executor(None, cmd_q.get)
            # send the command as text (we already transcribed it locally for free)
            await ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": cmd}]},
            }))
            await ws.send(json.dumps({"type": "response.create"}))

    async def recv():
        global _speaking
        async for raw in ws:
            ev = json.loads(raw)
            t = ev.get("type", "")
            if t == "response.created":
                _speaking = True
            elif t == "response.output_audio.delta":
                _speaking = True
                play_q.put(base64.b64decode(ev["delta"]))
            elif t == "response.output_audio_transcript.delta":
                print(ev.get("delta", ""), end="", flush=True)
            elif t == "response.output_audio_transcript.done":
                print()
            elif t in ("response.done", "response.output_audio.done"):
                _speaking = False
            elif t == "response.function_call_arguments.done":
                name, call_id = ev["name"], ev["call_id"]
                try:
                    args = json.loads(ev.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                print(f"\n  ⚙  {name}({args})", flush=True)
                _log(f"TOOL {name}({args})")
                result = await dispatch_tool(name, args)
                print(f"  ✓  {result.get('status')}", flush=True)
                await ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "function_call_output",
                             "call_id": call_id, "output": json.dumps(result)},
                }))
                await ws.send(json.dumps({"type": "response.create"}))
            elif t == "error":
                print("\n[realtime error]", json.dumps(ev.get("error", ev)), flush=True)

    await asyncio.gather(feed(), recv())


def resolve_input_device():
    if MIC_NAME:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and MIC_NAME.lower() in d["name"].lower():
                return i, d["name"]
        print(f"⚠  no input device matches {MIC_NAME!r}; using default.")
    try:
        return None, sd.query_devices(sd.default.device[0])["name"]
    except Exception:  # noqa: BLE001
        return None, "default input"


async def main():
    global _speaking, _primed
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY not set. (Only used AFTER the local wake word fires.)")
    headers = {"Authorization": f"Bearer {key}"}
    in_dev, mic_name = resolve_input_device()

    print("loading Whisper STT model… (first run downloads ~75MB)", flush=True)
    whisper = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")

    print(f"loading OpenWakeWord model ({OWW_MODEL!r})…", flush=True)
    openwakeword.utils.download_models()
    oww = OWWModel(wakeword_models=[OWW_MODEL], enable_speex_noise_suppression=False)

    print("=" * 62)
    print('  🎙  VOICE OS — LOCAL WAKE WORD ($0 idle): say "hey jarvis, …"')
    print(f"  mic: {mic_name}   ·   wake: {OWW_MODEL}   ·   STT: {WHISPER_SIZE}   ·   brain: {MODEL}")
    print("  wake detection runs FREE on your Mac; cloud only called on a match.")
    print(f"  OWW threshold: {OWW_THRESHOLD}  (tune via VOICEOS_OWW_THRESHOLD env var)")
    print("  Ctrl-C to quit.")
    print("=" * 62, flush=True)
    _log("--- start (OWW WAKE) ---")

    threading.Thread(target=recognizer_thread, args=(whisper, oww), daemon=True).start()

    with sd.RawInputStream(
        samplerate=CAP_RATE, channels=1, dtype="int16",
        blocksize=FRAME, callback=_mic_cb, device=in_dev,
    ), sd.RawOutputStream(
        samplerate=OUT_RATE, channels=1, dtype="int16",
        blocksize=OUT_BLOCK, callback=_spk_cb,
    ):
        while True:  # reconnect across the 60-min cap / drops (idle = $0, no audio sent)
            try:
                async with websockets.connect(
                    URL, additional_headers=headers, max_size=None
                ) as ws:
                    await ws_session(ws)
            except (websockets.ConnectionClosed, OSError) as e:
                _speaking = False
                _primed = False
                _play_buf.clear()
                _log(f"RECONNECT {getattr(e, 'code', '')}")
                await asyncio.sleep(0.5)
                continue


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nbye.")
