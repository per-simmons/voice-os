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
import time

_t_release = 0.0  # monotonic time of last key release, for latency timing

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
    "You are the voice operating system for Nathan's Mac. Nathan speaks a command to "
    "control the computer. There is no wake word — just act on what he says. Call "
    "exactly one matching tool, then give a short, natural spoken confirmation.\n"
    "ROUTING RULES (follow exactly):\n"
    "- To LAUNCH or open an app (including the Claude app): open_app. Just opening "
    "the Claude app with no question = open_app('Claude').\n"
    "- 'Open the YouTube Script project' (a Claude PROJECT, not an app): call "
    "ask_claude with an EMPTY question — it just navigates into the project.\n"
    "- If Nathan wants Claude to WRITE, REWRITE, SUGGEST, or answer anything (e.g. "
    "'ask Claude to rewrite the intro', 'have Claude suggest a script', 'open my "
    "YouTube script project and ask Claude to rewrite the intro'): call ask_claude "
    "with the request as the question. It opens the project and asks Claude itself. "
    "NEVER answer on Claude's behalf, and never type a command in as the question.\n"
    "- play music: play_music. control Premiere: premiere_control. read the screen: "
    "read_screen_aloud. start recording: start_obs_recording. switch OBS scene: obs_scene.\n"
    "- search the web / 'open the X docs' / look something up: web_search.\n"
    "- 'click the first link' / 'open the first result' / 'click the second one': click_link.\n"
    "- 'take a note: ...' / 'note that ...' / 'write this down': take_note.\n"
    "Only call a tool when the command is clear; if it's just a fragment, ask Nathan "
    "to repeat it. 'Throw on / put on a song' means play_music. Keep replies brief."
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
        "name": "web_search",
        "description": "Open the browser and search the web for a query. Use for 'search for X', 'look up X', 'google X', 'open the X docs', 'find the docs for X'.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "click_link",
        "description": "Click a search result in the browser — e.g. 'click the first link', 'open the first result', 'click the second one'. Opens the Nth organic result (below the AI overview).",
        "parameters": {
            "type": "object",
            "properties": {"position": {"type": "string", "description": "first / second / third (default first)"}},
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "take_note",
        "description": "Save a note in Apple Notes. Use for 'take a note: ...', 'note that ...', 'remind me ...', 'write this down: ...'. The text after the phrase is the note.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "the note content"}},
            "required": ["text"],
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
        "name": "stop_obs_recording",
        "description": "Stop the OBS recording.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "obs_scene",
        "description": "Switch OBS to a scene by name (e.g. 'YouTube Talking Head', 'Screen Only'). Use for 'switch to my talking head scene', 'change the scene to X'.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "scene name (fuzzy matched)"}},
            "required": ["name"],
        },
    },
    {
        "type": "function",
        "name": "ask_claude",
        "description": "Ask Claude (in Claude Desktop's YouTube Script project) to write/rewrite/suggest something and get its reply. This tool OPENS that project itself, then asks. Use it whenever the request includes asking Claude to write/rewrite/suggest/answer — INCLUDING 'open my YouTube script project and ask Claude to rewrite the intro', 'ask Claude to rewrite the intro', 'have Claude suggest a script'. The signal is a question/request FOR Claude to produce something. Do NOT use it merely to launch the Claude app when there is no question (that is open_app('Claude')). Don't answer on Claude's behalf; the returned 'response' is Claude's actual reply — read it back VERBATIM when Nathan asks what Claude said.",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string", "description": "what to ask Claude (e.g. 'we need a rewrite for this script, can you suggest any')"}},
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "premiere_control",
        "description": "Control Premiere Pro. Transport: 'pause'/'play'/'stop' toggle playback; 'left'/'right' step the playhead by frames (for 'go back 2 frames' use action='left', count=2). Editing: 'cut' (razor/add-edit at the playhead), 'cut_all_tracks', 'undo', 'redo', 'save', 'mark_in', 'mark_out', 'add_marker', 'ripple_delete', 'delete', 'zoom_in', 'zoom_out'. Default count is 1.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["pause", "play", "stop", "left", "right", "cut", "cut_all_tracks",
                             "undo", "redo", "save", "mark_in", "mark_out", "add_marker",
                             "ripple_delete", "delete", "zoom_in", "zoom_out"],
                },
                "count": {"type": "integer", "description": "number of frames to step for left/right (default 1)"},
            },
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
_in_stream = None       # input stream handle (on-demand in hotkey mode)
_in_dev = None          # chosen input device index


def _open_mic():
    """Open + start the mic input stream (idempotent). In hotkey mode we only
    hold the mic WHILE the key is pressed, so other apps (e.g. a dictation tool)
    aren't starved of the microphone the rest of the time."""
    global _in_stream
    if _in_stream is not None:
        return
    while not mic_q.empty():
        try:
            mic_q.get_nowait()
        except queue.Empty:
            break
    _in_stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16",
        blocksize=BLOCK, callback=_mic_cb, device=_in_dev,
    )
    _in_stream.start()


def _close_mic():
    """Stop + release the mic input stream so other apps can use the mic."""
    global _in_stream
    if _in_stream is not None:
        try:
            _in_stream.stop()
            _in_stream.close()
        except Exception:  # noqa: BLE001
            pass
        _in_stream = None


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
        try:
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(data).decode(),
                    }
                )
            )
        except websockets.exceptions.ConnectionClosed:
            break


async def ptt_console(ws):
    global _listening
    loop = asyncio.get_event_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if line == "":
            # Avoid a tight loop when launched without an interactive stdin
            # (for example, smoke tests or process supervisors).
            await asyncio.sleep(0.25)
            continue
        if _speaking or _play_buf:
            continue
        await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
        _listening = True
        print("🎙  listening… (speak, then pause)", flush=True)


# global hotkey (hold-to-talk anywhere) -------------------------------------
_HOTKEY_MAP = {}  # filled lazily so pynput import isn't required in other modes
key_events: "queue.Queue[str]" = queue.Queue()


def _start_hotkey_listener():
    """No-op. The global hotkey is provided by voice_app.py via the macOS
    RegisterEventHotKey API (safe: no event tap, not in the input path). It feeds
    'down'/'up' into key_events. We intentionally DO NOT use pynput here — its
    global listener installs a system-wide event tap that can freeze all input."""
    return


async def hotkey_console(ws):
    """Hold the global key to talk; release to send. Manual buffer commit.
    Pressing the key WHILE the model is talking interrupts it (barge-in)."""
    global _listening, _speaking
    loop = asyncio.get_event_loop()
    while True:
        ev = await loop.run_in_executor(None, key_events.get)
        global _t_release
        if ev == "down":
            if _speaking or _play_buf:
                # BARGE-IN: stop the model talking, then start listening so you
                # can cut it off with "got it" / your next command.
                await ws.send(json.dumps({"type": "response.cancel"}))
                _play_buf.clear()
                while not play_q.empty():
                    try:
                        play_q.get_nowait()
                    except queue.Empty:
                        break
                _speaking = False
                print("⏹  interrupted", flush=True)
            t0 = time.monotonic()
            _open_mic()  # grab the mic ONLY now (freed again on release)
            await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
            _listening = True
            print(f"🎙  listening… (mic open {(time.monotonic()-t0)*1000:.0f}ms)", flush=True)
        elif ev == "up":
            if not _listening:
                continue
            _close_mic()                  # release the mic immediately on release
            await asyncio.sleep(0.15)      # let mic_pump flush already-queued audio
            _listening = False
            _write_hud(False, 0.0)         # clear the waveform (mic_pump won't, it's now idle)
            _t_release = time.monotonic()
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
            lat = (time.monotonic() - _t_release) if _t_release else 0.0
            # clean, legible output for screen recording: ⚙ name(args)  →  ✓ ok
            arg_str = json.dumps(args, ensure_ascii=False)
            if len(arg_str) > 70:
                arg_str = arg_str[:67] + "…}"
            print(f"\n⚙  {name}({arg_str})", flush=True)
            _log(f"TOOL {name}({args}) latency={lat:.2f}s")
            t_exec = time.monotonic()
            result = await dispatch_tool(name, args)
            status = result.get("status", "?")
            print(f"✓  {status}" if status == "ok" else f"✗  {status}", flush=True)
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

    global _speaking, _listening, _primed, _in_dev
    _in_dev = in_dev
    # The speaker stream stays open. The MIC stream is on-demand in hotkey mode
    # (only while the key is held) so we never starve other apps of the mic;
    # wake/PTT modes need it always, so open it up front.
    out_stream = sd.RawOutputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16",
        blocksize=OUT_BLOCK, callback=_spk_cb,
    )
    out_stream.start()
    if not HOTKEY_MODE:
        _open_mic()
    try:
        # The Realtime API caps a session at 60 minutes, so reconnect forever and
        # re-init the session — the listener stays alive across the cap and any
        # transient network drop.
        while True:
            try:
                async with websockets.connect(
                    URL, additional_headers=headers, max_size=None
                ) as ws:
                    await ws.send(json.dumps(session_config()))
                    tasks = [asyncio.ensure_future(mic_pump(ws)),
                             asyncio.ensure_future(receive(ws))]
                    if PTT:
                        print("\n— press ENTER to talk —", flush=True)
                        tasks.append(asyncio.ensure_future(ptt_console(ws)))
                    elif HOTKEY_MODE:
                        print(f"\n— hold [{HOTKEY_NAME}] to talk —", flush=True)
                        tasks.append(asyncio.ensure_future(hotkey_console(ws)))
                    # reconnect as soon as ANY task ends (e.g. receive() returns
                    # when the 60-min session closes) — don't wait on idle tasks.
                    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for t in tasks:
                        t.cancel()
            except (websockets.ConnectionClosed, OSError) as e:
                _log(f"conn err {getattr(e, 'code', '')}")
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
            print("\n↻ reconnecting…", flush=True)
            _log("RECONNECT")
            await asyncio.sleep(0.5)
    finally:
        # always release the mic + speaker so we never leave a device wedged
        _close_mic()
        try:
            out_stream.stop()
            out_stream.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nbye.")
