#!/usr/bin/env python3
"""
wake_listener.py — LOCAL wake word, $0 idle.

The mic is processed entirely on your Mac: webrtcvad finds when you're speaking,
faster-whisper (tiny.en) transcribes it locally — all FREE, no network. We only
open a connection to OpenAI's gpt-realtime-2 AFTER we hear "hey chat", to run the
command and speak the reply. So leaving this on all day costs nothing until you
actually summon it (~1–2¢ per command).

Pipeline:
  mic ─▶ webrtcvad (local) ─▶ faster-whisper tiny.en (local) ─▶ "hey chat …"?
        └ no  → ignored, $0                                       │ yes
                                                                  ▼
                          open gpt-realtime-2 ─▶ tool call ─▶ agent-desktop ─▶ speak

Run:  ./run.sh --local     (or)   python wake_listener.py [--mic Scarlett]
Ctrl-C to quit. First run downloads the ~75MB tiny.en model once.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
import re
import sys
import threading
import time

try:
    import numpy as np
    import sounddevice as sd
    import webrtcvad
    import websockets
    from faster_whisper import WhisperModel
except ImportError as e:
    sys.exit(f"Missing dep ({e}). Run: ./run.sh --local  (installs local wake engine).")

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


MIC_NAME = _arg_value("--mic", os.environ.get("VOICEOS_MIC"))
WHISPER_SIZE = os.environ.get("VOICEOS_WHISPER", "base.en")  # base = sturdier than tiny

import difflib

# tolerant local wake matcher. base/tiny whisper mangles the short phrase "hey
# chat" badly ("happy chat", "he chat", "hey chad"...), so we (1) match a broad
# explicit list anywhere in the utterance AND (2) fall back to fuzzy matching on
# the first few word-pairs. Dedicated wake engines exist for exactly this reason;
# this gets us reliable enough without Picovoice.
_WAKE_PREFIX = r"(hey|hay|hi|he|ay|ey|ok|okay|happy|hi+|a)"
_WAKE_NAME = r"(chat|chats|chad|chap|chatt|chett|chet|jack|chent|chap|shot)"
_WAKE_RE = re.compile(rf"\b{_WAKE_PREFIX}\s+{_WAKE_NAME}\b")
_WAKE_ONEWORD = re.compile(r"\b(heychat|haychat|heychad|happychat)\b")
ARM_WINDOW = 6.0  # after a bare "hey chat", treat the next utterance as the command
_armed_until = 0.0


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def _wake_split(text: str):
    """Return the command after the wake word, or None if no wake word present.
    '' means the wake word was heard with no command after it (-> arm)."""
    norm = _norm(text)
    m = _WAKE_RE.search(norm) or _WAKE_ONEWORD.search(norm)
    if m:
        return norm[m.end():].strip(" ,.-")
    # fuzzy fallback: any of the first word-pairs close to "hey chat"
    toks = norm.split()
    for i in range(min(3, max(0, len(toks) - 1))):
        pair = toks[i] + " " + toks[i + 1]
        if difflib.SequenceMatcher(None, pair, "hey chat").ratio() >= 0.82:
            return " ".join(toks[i + 2:]).strip(" ,.-")
    return None

CAP_RATE = 16000        # capture/whisper/vad rate
FRAME_MS = 30
FRAME = CAP_RATE * FRAME_MS // 1000          # 480 samples
SILENCE_TAIL_MS = 450                         # end utterance after this much silence
MAX_UTTER_MS = 6000
OUT_RATE = 24000        # gpt-realtime-2 audio output
OUT_BLOCK = 4800
PRIME_BYTES = OUT_RATE * 2 * 300 // 1000
VOICE = "marin"
EVENT_LOG = "/tmp/voiceos-events.log"

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


def recognizer_thread(model: WhisperModel):
    """Local VAD + Whisper. Runs in a plain thread; pushes wake commands to cmd_q."""
    vad = webrtcvad.Vad(3)  # most aggressive: cleaner speech/silence segmentation
    buf = bytearray()
    voiced = bytearray()
    in_speech = False
    silence_ms = 0
    max_bytes = MAX_UTTER_MS * CAP_RATE // 1000 * 2

    while True:
        buf.extend(cap_q.get())
        # process whole 30ms frames
        while len(buf) >= FRAME * 2:
            frame = bytes(buf[: FRAME * 2])
            del buf[: FRAME * 2]
            if _speaking:  # ignore everything while the model is talking
                voiced.clear()
                in_speech = False
                silence_ms = 0
                continue
            speech = vad.is_speech(frame, CAP_RATE)
            if speech:
                if not in_speech:
                    in_speech = True
                    silence_ms = 0
                voiced.extend(frame)
                silence_ms = 0
            elif in_speech:
                voiced.extend(frame)
                silence_ms += FRAME_MS
                if silence_ms >= SILENCE_TAIL_MS or len(voiced) >= max_bytes:
                    _finalize(model, bytes(voiced))
                    voiced.clear()
                    in_speech = False
                    silence_ms = 0


def _finalize(model: WhisperModel, pcm: bytes):
    if len(pcm) < CAP_RATE:  # < ~0.03s? skip tiny blips (guard)
        pass
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size < CAP_RATE // 3:  # < ~0.33s -> too short to be a command
        return
    segments, _ = model.transcribe(audio, language="en", beam_size=1)
    text = " ".join(s.text for s in segments).strip()
    if not text:
        return

    global _armed_until
    now = time.monotonic()
    cmd = _wake_split(text)

    if cmd is not None:  # wake word heard somewhere in the utterance
        if cmd:  # "hey chat, open spotify" — wake + command together
            print(f"\n🗣  WAKE ✓ → {cmd!r}", flush=True)
            _log(f"WAKE {text!r} -> {cmd!r}")
            cmd_q.put(cmd)
            _armed_until = 0.0
        else:    # just "hey chat" — arm for the next utterance
            print('\n🟢 armed — say your command (e.g. "open Spotify")', flush=True)
            _log(f"ARMED {text!r}")
            _armed_until = now + ARM_WINDOW
    elif now < _armed_until:  # the command after a bare "hey chat"
        print(f"\n🗣  COMMAND (armed) → {text!r}", flush=True)
        _log(f"ARMED-CMD {text!r}")
        cmd_q.put(text)
        _armed_until = 0.0
    else:
        print(f"\n·  ignored locally (no wake, $0): {text!r}", flush=True)
        _log(f"LOCAL-IGNORED {text!r}")


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

    print("loading local speech model… (first run downloads ~75MB)", flush=True)
    model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")

    print("=" * 62)
    print('  🎙  VOICE OS — LOCAL WAKE WORD ($0 idle): say "hey chat, …"')
    print(f"  mic: {mic_name}   ·   local STT: {WHISPER_SIZE}   ·   brain: {MODEL}")
    print("  wake word runs FREE on your Mac; cloud is called only on a match.")
    print("  Ctrl-C to quit.")
    print("=" * 62, flush=True)
    _log("--- start (LOCAL WAKE) ---")

    threading.Thread(target=recognizer_thread, args=(model,), daemon=True).start()

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
