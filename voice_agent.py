#!/usr/bin/env python3
"""
voice_agent.py — gpt-realtime-2 speech-to-speech control of the Mac.

DEFAULT = WAKE WORD ("hey chat"). The mic streams continuously, but the model
ONLY acts on a turn whose transcript starts with the wake word. You talking to
someone else, room noise, or the model's own voice are transcribed, fail the
wake check, and are ignored — so it never does things you didn't ask for, and it
can't loop on itself.

How it works (no Picovoice / no local wake engine needed):
  - server VAD detects your turns and transcribes them, but `create_response` is
    OFF, so the model never auto-replies.
  - we read each transcript; if it starts with "hey chat", we fire a response
    (the model then runs the matching tool and speaks back). Otherwise: ignored.

Modes:
  (default)         wake word "hey chat"
  --push-to-talk    press ENTER to talk (no wake word)

Requires OPENAI_API_KEY with Realtime access. Run via ./run.sh (or ./talk.sh).
Ctrl-C to quit. Live transcript of what it heard prints as "HEARD: ...".
"""
from __future__ import annotations

import array
import asyncio
import base64
import json
import math
import os
import queue
import re
import sys

try:
    import sounddevice as sd
    import websockets
except ImportError:
    sys.exit("Missing deps. Run ./run.sh (installs sounddevice + websockets).")

import actions

def _arg_value(flag, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


PTT = "--push-to-talk" in sys.argv
# hold a global key anywhere to talk (e.g. --hotkey right_option). Default key if
# --hotkey is passed with no value = right_option.
HOTKEY_NAME = _arg_value("--hotkey", "right_option") if "--hotkey" in sys.argv else None
HOTKEY_MODE = HOTKEY_NAME is not None
WAKE_MODE = not (PTT or HOTKEY_MODE)

# choose input device by name substring (e.g. --mic Scarlett or VOICEOS_MIC=Scarlett)
MIC_NAME = _arg_value("--mic", os.environ.get("VOICEOS_MIC"))

MODEL = "gpt-realtime-2"
URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"
SAMPLE_RATE = 24000
CHANNELS = 1
BLOCK = 2400
OUT_BLOCK = 4800
PRIME_BYTES = SAMPLE_RATE * 2 * 300 // 1000
EVENT_LOG = "/tmp/voiceos-events.log"
HUD_FILE = "/tmp/voiceos-hud.json"  # waveform overlay reads {active, level} from here

VOICE = "marin"
WAKE_WORD = "hey chat"
# tolerate common mishears of "hey chat"
_WAKE_RE = re.compile(r"^\s*(hey|hay|a|hi)\s+(chat|chad|chap|chats|chatt|chett|chet|jack)\b")

INSTRUCTIONS = (
    "You are the voice operating system for Pat's Mac. The user addresses you as "
    "'hey chat'. Act on the command that follows the wake word (open an app, play "
    "music, run a terminal command, read the screen, start recording). Call exactly "
    "one matching tool, then give a short, natural spoken confirmation.\n"
    "CRITICAL: Only call a tool when the command is COMPLETE and UNAMBIGUOUS. If the "
    "transcript sounds cut off, trails off, or is just a fragment (e.g. 'can you turn "
    "on...', 'open...'), DO NOT guess and DO NOT call any tool — instead ask Pat to "
    "say the full command again. 'Turn on a song' / 'put on a song' means play_music, "
    "NOT recording. Never map a vague phrase onto a destructive or unrelated action. "
    "Keep replies brief."
)

TOOLS = [
    {
        "type": "function",
        "name": "open_app",
        "description": "Launch or focus a macOS app by name (e.g. Spotify, OBS, Google Chrome, Premiere Pro).",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "type": "function",
        "name": "play_music",
        "description": "Open Spotify and play music. Pass a search query like 'Tchaikovsky', or empty to resume.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "run_terminal",
        "description": "Open Terminal and start a Claude Code session with the given prompt.",
        "parameters": {
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
        },
    },
    {
        "type": "function",
        "name": "read_screen_aloud",
        "description": "Read back the text currently visible in an app (default Terminal).",
        "parameters": {
            "type": "object",
            "properties": {"app": {"type": "string"}},
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "start_obs_recording",
        "description": "Open OBS and start recording.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "premiere_control",
        "description": "Control Premiere Pro: 'pause'/'play'/'stop' toggle playback; 'left' steps the playhead back one frame, 'right' steps it forward one frame (e.g. 'move it left a frame').",
        "parameters": {
            "type": "object",
            "properties": {"action": {"type": "string", "enum": ["pause", "play", "stop", "left", "right"]}},
            "required": [],
        },
    },
]


def is_wake(transcript: str) -> bool:
    norm = re.sub(r"[^a-z0-9\s]", " ", (transcript or "").lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    return bool(_WAKE_RE.match(norm))


def session_config() -> dict:
    audio_in = {
        "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
        "transcription": {"model": "whisper-1"},
    }
    if WAKE_MODE:
        # detect + transcribe turns, but DON'T auto-respond — we gate on the wake
        # word and fire response.create ourselves only when "hey chat" is heard.
        # semantic_vad ends the turn when you've semantically FINISHED a command,
        # not on a fixed silence timer — snappier than server_vad AND fewer
        # mid-sentence cutoffs. We still gate on the wake word + fire the response
        # ourselves (create_response off).
        audio_in["turn_detection"] = {
            "type": "semantic_vad",
            "eagerness": "high",
            "create_response": False,
            "interrupt_response": False,
        }
    elif HOTKEY_MODE:
        # hold-to-talk: WE decide when the turn starts/ends (key down/up), so no
        # automatic VAD — we commit the buffer manually on key release.
        audio_in["turn_detection"] = None
    else:  # push-to-talk (Enter)
        audio_in["turn_detection"] = {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 1500,
        }
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": MODEL,
            "instructions": INSTRUCTIONS,
            "output_modalities": ["audio"],
            "audio": {
                "input": audio_in,
                "output": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    "voice": VOICE,
                },
            },
            "tools": TOOLS,
            "tool_choice": "auto",
        },
    }


mic_q: "queue.Queue[bytes]" = queue.Queue()
play_q: "queue.Queue[bytes]" = queue.Queue()
_play_buf = bytearray()
_primed = False
_speaking = False
_listening = WAKE_MODE  # wake mode streams always; PTT streams only after Enter


def _log(msg: str):
    try:
        with open(EVENT_LOG, "a") as f:
            f.write(msg + "\n")
    except OSError:
        pass


def _write_hud(active: bool, level: float):
    """Publish mic level + listening state for the waveform overlay (overlay.py)."""
    try:
        with open(HUD_FILE, "w") as f:
            json.dump({"active": bool(active), "level": float(level)}, f)
    except OSError:
        pass


def _frame_level(data: bytes) -> float:
    """Normalized RMS amplitude (0..1) of a PCM16 frame, for the waveform."""
    a = array.array("h")
    a.frombytes(data)
    if not a:
        return 0.0
    rms = math.sqrt(sum(x * x for x in a) / len(a))
    return min(1.0, rms / 8000.0)


def _mic_cb(indata, frames, t, status):
    mic_q.put(bytes(indata))


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


async def dispatch_tool(name: str, args: dict) -> dict:
    fn = actions.TOOLS.get(name)
    if not fn:
        return {"status": "error", "error": f"unknown tool {name}"}
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, lambda: fn(**args))
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}


async def mic_pump(ws):
    loop = asyncio.get_event_loop()
    while True:
        data = await loop.run_in_executor(None, mic_q.get)
        # publish level + state for the waveform overlay (active only when the
        # mic is actually going to chat)
        active = _listening and not _speaking and not _play_buf
        _write_hud(active, _frame_level(data) if active else 0.0)
        # don't forward the mic while the model is speaking / draining (anti-echo),
        # and in PTT mode only after Enter.
        if not _listening or _speaking or _play_buf:
            continue
        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(data).decode(),
                }
            )
        )


async def ptt_console(ws):
    global _listening
    loop = asyncio.get_event_loop()
    while True:
        await loop.run_in_executor(None, sys.stdin.readline)
        if _speaking or _play_buf:
            continue
        await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
        _listening = True
        print("🎙  listening… (speak, then pause)", flush=True)


# global hotkey (hold-to-talk anywhere) -------------------------------------
_HOTKEY_MAP = {}  # filled lazily so pynput import isn't required in other modes
key_events: "queue.Queue[str]" = queue.Queue()


def _start_hotkey_listener():
    from pynput import keyboard

    global _HOTKEY_MAP
    _HOTKEY_MAP = {
        "right_option": keyboard.Key.alt_r,
        "left_option": keyboard.Key.alt_l,
        "right_cmd": keyboard.Key.cmd_r,
        "right_shift": keyboard.Key.shift_r,
        "right_ctrl": keyboard.Key.ctrl_r,
        "f8": keyboard.Key.f8,
        "f9": keyboard.Key.f9,
    }
    target = _HOTKEY_MAP.get(HOTKEY_NAME, keyboard.Key.alt_r)
    held = {"v": False}

    def on_press(k):
        if k == target and not held["v"]:
            held["v"] = True
            key_events.put("down")

    def on_release(k):
        if k == target and held["v"]:
            held["v"] = False
            key_events.put("up")

    keyboard.Listener(on_press=on_press, on_release=on_release).start()


async def hotkey_console(ws):
    """Hold the global key to talk; release to send. Manual buffer commit."""
    global _listening
    loop = asyncio.get_event_loop()
    while True:
        ev = await loop.run_in_executor(None, key_events.get)
        if ev == "down":
            if _speaking or _play_buf:
                continue
            await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
            _listening = True
            print("🎙  listening… (hold the key)", flush=True)
        elif ev == "up":
            if not _listening:
                continue
            _listening = False
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            await ws.send(json.dumps({"type": "response.create"}))
            print("⏳ thinking…", flush=True)


async def receive(ws):
    global _speaking, _listening
    async for raw in ws:
        ev = json.loads(raw)
        t = ev.get("type", "")
        if t not in ("response.output_audio.delta", "response.output_audio_transcript.delta"):
            _log(t)

        if t == "response.created":
            _speaking = True
            if PTT:
                _listening = False  # PTT: turn handed to the model until next Enter

        elif t == "response.output_audio.delta":
            _speaking = True
            play_q.put(base64.b64decode(ev["delta"]))

        elif t == "response.output_audio_transcript.delta":
            print(ev.get("delta", ""), end="", flush=True)

        elif t == "response.output_audio_transcript.done":
            print()

        elif t in ("response.done", "response.output_audio.done"):
            _speaking = False
            if PTT and t == "response.done":
                print("\n— press ENTER to talk —", flush=True)

        elif t == "conversation.item.input_audio_transcription.completed":
            heard = (ev.get("transcript") or "").strip()
            if WAKE_MODE:
                if is_wake(heard):
                    print(f"\n🗣  HEARD (wake ✓): {heard!r}", flush=True)
                    _log(f"WAKE {heard!r}")
                    await ws.send(json.dumps({"type": "response.create"}))
                else:
                    print(f"\n·  ignored (no wake word): {heard!r}", flush=True)
                    _log(f"IGNORED {heard!r}")
            else:
                print(f"\n🗣  HEARD: {heard!r}", flush=True)
                _log(f"HEARD {heard!r}")

        elif t == "input_audio_buffer.speech_started":
            while not play_q.empty():
                try:
                    play_q.get_nowait()
                except queue.Empty:
                    break
            _play_buf.clear()

        elif t == "response.function_call_arguments.done":
            name = ev["name"]
            call_id = ev["call_id"]
            try:
                args = json.loads(ev.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            print(f"\n  ⚙  {name}({args})", flush=True)
            _log(f"TOOL {name}({args})")
            result = await dispatch_tool(name, args)
            print(f"  ✓  {result.get('status')}", flush=True)
            await ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps(result),
                        },
                    }
                )
            )
            await ws.send(json.dumps({"type": "response.create"}))

        elif t == "error":
            print("\n[realtime error]", json.dumps(ev.get("error", ev)), flush=True)
            _log("ERROR " + json.dumps(ev.get("error", ev)))


def resolve_input_device():
    """Return (index, name) for the chosen mic, or (None, default name)."""
    if MIC_NAME:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and MIC_NAME.lower() in d["name"].lower():
                return i, d["name"]
        print(f"⚠  no input device matches {MIC_NAME!r}; using system default.")
    try:
        return None, sd.query_devices(sd.default.device[0])["name"]
    except Exception:  # noqa: BLE001
        return None, "default input"


async def main():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY not set. Export a valid Realtime-capable key first.")
    headers = {"Authorization": f"Bearer {key}"}
    in_dev, mic_name = resolve_input_device()
    print("=" * 60)
    if WAKE_MODE:
        print(f"  🎙  VOICE OS — WAKE WORD: say “{WAKE_WORD}, …”")
        print("  e.g. “hey chat, open Spotify”   ·   anything without the")
        print("  wake word is ignored. Ctrl-C to quit.")
    elif HOTKEY_MODE:
        print(f"  🎙  VOICE OS — HOLD-TO-TALK: hold [{HOTKEY_NAME}] anywhere")
        print("  hold the key, speak, release to send. Ctrl-C to quit.")
    else:
        print("  🎙  VOICE OS — PUSH-TO-TALK (press ENTER to talk)")
    print(f"  mic: {mic_name}   ·   brain: {MODEL}   ·   log: {EVENT_LOG}")
    print("=" * 60, flush=True)
    _log(f"--- start ({'WAKE' if WAKE_MODE else 'HOTKEY' if HOTKEY_MODE else 'PTT'}) ---")
    if HOTKEY_MODE:
        _start_hotkey_listener()

    global _speaking, _listening, _primed
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16",
        blocksize=BLOCK, callback=_mic_cb, device=in_dev,
    ), sd.RawOutputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16",
        blocksize=OUT_BLOCK, callback=_spk_cb,
    ):
        # The Realtime API caps a session at 60 minutes, so reconnect forever and
        # re-init the session — the listener stays alive across the cap and any
        # transient network drop. Audio streams stay open across reconnects.
        while True:
            try:
                async with websockets.connect(
                    URL, additional_headers=headers, max_size=None
                ) as ws:
                    await ws.send(json.dumps(session_config()))
                    tasks = [mic_pump(ws), receive(ws)]
                    if PTT:
                        print("\n— press ENTER to talk —", flush=True)
                        tasks.append(ptt_console(ws))
                    elif HOTKEY_MODE:
                        print(f"\n— hold [{HOTKEY_NAME}] to talk —", flush=True)
                        tasks.append(hotkey_console(ws))
                    await asyncio.gather(*tasks)
            except (websockets.ConnectionClosed, OSError) as e:
                # reset per-connection state, drain stale audio, reconnect
                _speaking = False
                _listening = WAKE_MODE
                _primed = False
                _play_buf.clear()
                while not mic_q.empty():
                    try:
                        mic_q.get_nowait()
                    except queue.Empty:
                        break
                print(f"\n↻ session reset ({getattr(e, 'code', '')}) — reconnecting…", flush=True)
                _log(f"RECONNECT {getattr(e, 'code', '')}")
                await asyncio.sleep(0.5)
                continue


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nbye.")
