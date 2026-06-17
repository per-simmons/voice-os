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
import config

try:
    import retrieval as _retrieval
    from session_log import SessionLog
    import retrospective as _retrospective
    _MEMORY_ENABLED = True
except ImportError:
    _MEMORY_ENABLED = False

_session: "SessionLog | None" = None
_cap_index: "_retrieval.CapabilityIndex | None" = None

def _arg_value(flag, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


PTT = "--push-to-talk" in sys.argv
HOTKEY_NAME = _arg_value("--hotkey", "right_option") if "--hotkey" in sys.argv else None
HOTKEY_MODE = HOTKEY_NAME is not None
WAKE_MODE = not (PTT or HOTKEY_MODE)

MIC_NAME = _arg_value("--mic", config.MIC_NAME)

MODEL = config.MODEL
URL = config.URL
SAMPLE_RATE = config.SAMPLE_RATE
CHANNELS = config.OUT_CHANNELS
BLOCK = config.BLOCK
OUT_BLOCK = config.OUT_BLOCK
PRIME_BYTES = config.PRIME_BYTES
EVENT_LOG = config.EVENT_LOG
HUD_FILE = config.HUD_FILE

VOICE = config.VOICE
WAKE_WORD = config.WAKE_WORD
_EVT_RESPONSE_CREATE = json.dumps({"type": "response.create"})
# Whisper mishears of "hey chat" — includes NZ-accent variants
# ("chut", "chit", "jet", "jat", "ject" for "chat"; "a" / "eh" for "hey")
_WAKE_RE = re.compile(
    r"^\s*(hey|hay|hi|he|ay|ey|ok|okay|happy|a|eh|aye)\s+"
    r"(chat|chats|chad|chap|chatt|chett|chet|jack|chent|shot"
    r"|chit|chut|jet|jat|ject|chot|chart|chant|shat|char)\b"
    r"|^\s*(heychat|haychat|heychad|happychat|achat|eychat)\b"
)

def _build_instructions() -> str:
    _name = config.USER_NAME
    _hints = (" " + config.USER_HINTS.strip()) if config.USER_HINTS.strip() else ""
    _browser = config.WEB_BROWSER
    return (
        f"You are the voice operating system for {_name}'s Mac.{_hints}\n"
        f"The default browser is {_browser}.\n"
        f"{_name} speaks a computer control command and you execute it.\n\n"
        "TOOLS:\n"
        "  run_applescript(script)              - run any AppleScript\n"
        "  press_key(combo, app, repeat?)       - send a key to an app\n"
        "  read_screen(app?)                    - read visible text from an app\n"
        f"  open_url(url)                        - open a URL in {_browser}\n"
        "  obs_call(request_type, request_data) - control OBS via WebSocket\n\n"
        "CONTEXT: Before each response you receive RETRIEVED CAPABILITIES showing\n"
        "the most relevant known recipes. Use them as your exact template.\n\n"
        "RULES (follow strictly):\n"
        "- This is a voice OS, not a chatbot. ONLY respond to computer control commands.\n"
        "- If the command matches a STRONG retrieved capability: execute it immediately.\n"
        "- If grounding is WEAK or the command is ambiguous: say 'I can only run Mac\n"
        f"  commands — what would you like me to do?' and stop.\n"
        "- NEVER answer general questions, chat, or act as an assistant.\n"
        "- After a tool returns status ok: one short confirmation sentence, then stop.\n"
        "- Never call the same tool twice in a row.\n"
        "- For 'click the first/second/third link': use the click-link capability\n"
        f"  with the correct template for {_browser} (index 0=first, 1=second, 2=third)."
    )


INSTRUCTIONS = _build_instructions()

TOOLS = [
    {
        "type": "function",
        "name": "run_applescript",
        "description": "Execute AppleScript on the Mac. Use for launching/focusing apps, controlling Spotify, Notes, Terminal, Finder, System Events, or any app with a scripting dictionary.",
        "parameters": {
            "type": "object",
            "properties": {"script": {"type": "string", "description": "The full AppleScript to run"}},
            "required": ["script"],
        },
    },
    {
        "type": "function",
        "name": "press_key",
        "description": "Send a keyboard shortcut to a running macOS app. Use for Premiere Pro editing, or any app that responds to hotkeys.",
        "parameters": {
            "type": "object",
            "properties": {
                "combo": {"type": "string", "description": "Key combo e.g. 'space', 'cmd+k', 'cmd+shift+z', 'left'"},
                "app": {"type": "string", "description": "App name substring e.g. 'Premiere', 'Final Cut'"},
                "repeat": {"type": "integer", "description": "How many times to press (default 1)"},
            },
            "required": ["combo", "app"],
        },
    },
    {
        "type": "function",
        "name": "read_screen",
        "description": "Read the text currently visible in a macOS app via the accessibility tree. Use for 'what does the screen say', 'read that back', 'what did Claude say'.",
        "parameters": {
            "type": "object",
            "properties": {"app": {"type": "string", "description": "App name to read from (default: Terminal)"}},
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "open_url",
        "description": "Open a URL in the browser. For web searches use https://www.google.com/search?q=<url-encoded-query>. Also works for spotify: and other URL schemes.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Full URL to open"}},
            "required": ["url"],
        },
    },
    {
        "type": "function",
        "name": "obs_call",
        "description": "Control OBS via its WebSocket API. request_type examples: StartRecord, StopRecord, SetCurrentProgramScene (with request_data {sceneName: '...'}), GetSceneList.",
        "parameters": {
            "type": "object",
            "properties": {
                "request_type": {"type": "string", "description": "OBS WebSocket request type"},
                "request_data": {"type": "object", "description": "Optional request payload"},
            },
            "required": ["request_type"],
        },
    },
]


def is_wake(transcript: str) -> bool:
    norm = re.sub(r"[^a-z0-9\s]", " ", (transcript or "").lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    return bool(_WAKE_RE.match(norm))


def session_config() -> dict:
    transcription = {"model": config.TRANSCRIBE_MODEL}
    if config.TRANSCRIBE_LANGUAGE:
        transcription["language"] = config.TRANSCRIBE_LANGUAGE
    if config.TRANSCRIBE_PROMPT:
        transcription["prompt"] = config.TRANSCRIBE_PROMPT
    audio_in = {
        "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
        "transcription": transcription,
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
    if not data:
        return 0.0
    try:
        import numpy as _np
        a = _np.frombuffer(data, dtype=_np.int16).astype(_np.float32)
        rms = float(_np.sqrt(_np.mean(a * a)))
    except Exception:  # noqa: BLE001
        a = array.array("h")
        a.frombytes(data)
        rms = math.sqrt(sum(x * x for x in a) / len(a)) if a else 0.0
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


def _drain_play_queue() -> None:
    """Discard all queued playback audio (used on barge-in and speech-start)."""
    while not play_q.empty():
        try:
            play_q.get_nowait()
        except queue.Empty:
            break
    _play_buf.clear()


def _drain_mic_queue() -> None:
    """Discard stale mic frames (used after a reconnect)."""
    while not mic_q.empty():
        try:
            mic_q.get_nowait()
        except queue.Empty:
            break


async def dispatch_tool(name: str, args: dict) -> dict:
    fn = actions.TOOLS.get(name)
    if not fn:
        return {"status": "error", "error": f"unknown tool {name}"}
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, lambda: fn(**args))
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}


async def mic_pump(ws):
    loop = asyncio.get_running_loop()
    while True:
        data = await loop.run_in_executor(None, mic_q.get)
        active = _listening and not _speaking and not _play_buf
        _write_hud(active, _frame_level(data) if active else 0.0)
        if not active:
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
        except websockets.ConnectionClosed:
            return  # socket closed (shutdown / 60-min cap) — exit cleanly


async def ptt_console(ws):
    global _listening
    loop = asyncio.get_running_loop()
    while True:
        await loop.run_in_executor(None, sys.stdin.readline)
        if _speaking or _play_buf:
            continue
        await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
        _listening = True
        print("🎙  listening… (speak, then pause)", flush=True)


# global hotkey (hold-to-talk anywhere) -------------------------------------
key_events: "queue.Queue[str]" = queue.Queue()


def _start_hotkey_listener():
    """No-op. The global hotkey is provided by voice_app.py via the macOS
    RegisterEventHotKey API (safe: no event tap, not in the input path). It feeds
    'down'/'up' into key_events. We intentionally DO NOT use pynput here — its
    global listener installs a system-wide event tap that can freeze all input."""


async def hotkey_console(ws):
    """Hold the global key to talk; release to send. Manual buffer commit.
    Pressing the key WHILE the model is talking interrupts it (barge-in)."""
    global _listening, _speaking
    loop = asyncio.get_running_loop()
    while True:
        ev = await loop.run_in_executor(None, key_events.get)
        global _t_release
        if ev == "down":
            if _speaking or _play_buf:
                await ws.send(json.dumps({"type": "response.cancel"}))
                _drain_play_queue()
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
            await ws.send(_EVT_RESPONSE_CREATE)
            print("⏳ thinking…", flush=True)


async def _inject_capability_context(ws, transcript: str) -> None:
    """Retrieve relevant capabilities and inject as a system context item
    into the conversation before response.create fires."""
    if not _MEMORY_ENABLED or _cap_index is None:
        return
    try:
        results = _cap_index.search(transcript, top_k=3)
        grounding = _cap_index.grounding(results)
        context = _cap_index.format_context(results, grounding)
        if not context:
            return
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": context}],
            },
        }))
    except Exception:  # noqa: BLE001
        pass


async def _handle_transcription(ws, ev: dict) -> None:
    heard = (ev.get("transcript") or "").strip()
    if _session:
        _session.heard(heard)
    if WAKE_MODE:
        if is_wake(heard):
            print(f"\n🗣  HEARD (wake ✓): {heard!r}", flush=True)
            _log(f"WAKE {heard!r}")
            if _session:
                _session.wake(heard)
            await _inject_capability_context(ws, heard)
            await ws.send(_EVT_RESPONSE_CREATE)
        else:
            print(f"\n·  ignored (no wake word): {heard!r}", flush=True)
            _log(f"IGNORED {heard!r}")
            if _session:
                _session.ignored(heard)
    else:
        print(f"\n🗣  HEARD: {heard!r}", flush=True)
        _log(f"HEARD {heard!r}")
        await _inject_capability_context(ws, heard)


async def _handle_tool_call(ws, ev: dict) -> None:
    name = ev["name"]
    call_id = ev["call_id"]
    try:
        args = json.loads(ev.get("arguments") or "{}")
    except json.JSONDecodeError:
        args = {}
    t0 = time.monotonic()
    lat = (t0 - _t_release) if _t_release else 0.0
    arg_str = json.dumps(args, ensure_ascii=False)
    if len(arg_str) > 70:
        arg_str = arg_str[:67] + "…}"
    print(f"\n⚙  {name}({arg_str})", flush=True)
    _log(f"TOOL {name}({args}) latency={lat:.2f}s")
    result = await dispatch_tool(name, args)
    exec_time = time.monotonic() - t0
    status = result.get("status", "?")
    print(f"✓  {status}" if status == "ok" else f"✗  {status}", flush=True)
    if _session:
        _session.tool_call(name, args, result, exec_time)
    await ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {"type": "function_call_output", "call_id": call_id,
                 "output": json.dumps(result)},
    }))
    await ws.send(_EVT_RESPONSE_CREATE)


async def receive(ws):
    global _speaking, _listening
    _NOISY = {"response.output_audio.delta", "response.output_audio_transcript.delta"}
    async for raw in ws:
        ev = json.loads(raw)
        t = ev.get("type", "")
        if t not in _NOISY:
            _log(t)

        if t == "response.created":
            _speaking = True
            if PTT:
                _listening = False
        elif t == "response.output_audio.delta":
            _speaking = True
            play_q.put(base64.b64decode(ev["delta"]))
        elif t == "response.output_audio_transcript.delta":
            print(ev.get("delta", ""), end="", flush=True)
        elif t == "response.output_audio_transcript.done":
            spoken_text = ev.get("transcript", "").strip()
            print()
            if _session and spoken_text:
                _session.spoken(spoken_text)
        elif t in ("response.done", "response.output_audio.done"):
            _speaking = False
            if PTT and t == "response.done":
                print("\n— press ENTER to talk —", flush=True)
        elif t == "conversation.item.input_audio_transcription.completed":
            await _handle_transcription(ws, ev)
        elif t == "input_audio_buffer.speech_started":
            _drain_play_queue()
        elif t == "response.function_call_arguments.done":
            await _handle_tool_call(ws, ev)
        elif t == "error":
            err = ev.get("error", ev)
            print("\n[realtime error]", json.dumps(err), flush=True)
            _log("ERROR " + json.dumps(err))
            if _session:
                _session.error(err)


def _print_banner(mic_name: str) -> None:
    print("=" * 60)
    if WAKE_MODE:
        print(f"  🎙  VOICE OS — WAKE WORD: say \u201c{WAKE_WORD}, \u2026\u201d")
        print("  e.g. \u201chey chat, open Spotify\u201d   \u00b7   anything without the")
        print("  wake word is ignored. Ctrl-C to quit.")
    elif HOTKEY_MODE:
        print(f"  🎙  VOICE OS — HOLD-TO-TALK: hold [{HOTKEY_NAME}] anywhere")
        print("  hold the key, speak, release to send. Ctrl-C to quit.")
    else:
        print("  🎙  VOICE OS — PUSH-TO-TALK (press ENTER to talk)")
    print(f"  mic: {mic_name}   \u00b7   brain: {MODEL}   \u00b7   log: {EVENT_LOG}")
    print("=" * 60, flush=True)
    if WAKE_MODE:
        mode_label = "WAKE"
    elif HOTKEY_MODE:
        mode_label = "HOTKEY"
    else:
        mode_label = "PTT"
    _log(f"--- start ({mode_label}) ---")


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
    global _session, _cap_index
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY not set. Export a valid Realtime-capable key first.")
    headers = {"Authorization": f"Bearer {key}"}
    in_dev, mic_name = resolve_input_device()
    _print_banner(mic_name)
    if HOTKEY_MODE:
        _start_hotkey_listener()

    if _MEMORY_ENABLED:
        _session = SessionLog(user=config.USER_NAME)
        try:
            _cap_index = _retrieval.get_index(verbose=True)
        except Exception as e:  # noqa: BLE001
            print(f"[memory] retrieval index unavailable: {e}", flush=True)
            _cap_index = None

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
                    tasks = [asyncio.create_task(mic_pump(ws)),
                             asyncio.create_task(receive(ws))]
                    if PTT:
                        print("\n— press ENTER to talk —", flush=True)
                        tasks.append(asyncio.create_task(ptt_console(ws)))
                    elif HOTKEY_MODE:
                        print(f"\n— hold [{HOTKEY_NAME}] to talk —", flush=True)
                        tasks.append(asyncio.create_task(hotkey_console(ws)))
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
            _drain_play_queue()
            _drain_mic_queue()
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
        if _session:
            _session.close()
            # run the dreaming loop — reflect on this session and learn from it
            print("\n💭 running retrospective…", flush=True)
            loop = asyncio.get_event_loop()
            try:
                added = await loop.run_in_executor(
                    None,
                    lambda: _retrospective.run_retrospective(
                        session_log_path=_session.path, verbose=True
                    )
                )
                if added:
                    print(f"💡 learned {added} memory update(s) this session", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[retrospective] failed: {e}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nbye.")
