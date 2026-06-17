#!/usr/bin/env python3
"""
voice_app.py — SAFE global hotkey front-end for the voice OS.

Registers a real macOS global hotkey via Carbon `RegisterEventHotKey`. This is
the safe way to do a system-wide push-to-talk:
  - it does NOT require Accessibility / Input Monitoring permission, and
  - it is NOT an event tap — it does not sit in the path of all your keystrokes,
    so it physically cannot freeze your input (unlike pynput, which did).

Hold the combo to talk, release to send. Default combo: Left Option + Z (⌥Z) —
one-handed, bottom-left, and it avoids Right Option (which Murmur owns).

Change the combo with --combo, e.g.:
    python voice_app.py --combo opt+z
    python voice_app.py --combo ctrl+opt+z
    python voice_app.py --combo opt+grave

The mic is grabbed ONLY while you hold the key (on-demand), so other apps keep
the mic the rest of the time.
"""
import ctypes
import ctypes.util
import sys
import threading
from ctypes import CFUNCTYPE, POINTER, Structure, byref, c_int32, c_uint32, c_void_p


def _arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


COMBO = _arg("--combo", "opt+z").lower()
MIC = _arg("--mic", "")  # leave empty to use the system default (avoids Scarlett/Murmur)

# Carbon virtual key codes (US layout) for the keys we support as the trigger key
KEYCODES = {
    "z": 6, "x": 7, "c": 8, "v": 9, "a": 0, "s": 1, "d": 2, "q": 12, "w": 13,
    "space": 49, "grave": 50, "`": 50, "tab": 48,
}
# Carbon modifier masks
MODS = {"cmd": 0x100, "shift": 0x200, "opt": 0x800, "option": 0x800, "ctrl": 0x1000, "control": 0x1000}


def parse_combo(s):
    parts = s.replace("-", "+").split("+")
    mods = 0
    key = None
    for p in parts:
        p = p.strip()
        if p in MODS:
            mods |= MODS[p]
        elif p in KEYCODES:
            key = KEYCODES[p]
    if key is None:
        key = KEYCODES["z"]
        mods = MODS["opt"]
    return key, mods


KEYCODE, MODIFIERS = parse_combo(COMBO)

# ---- bring up the voice agent in hotkey mode, driven by THIS file's hotkey ----
sys.argv = ["voice_agent.py", "--hotkey", "external"] + (["--mic", MIC] if MIC else [])
import asyncio  # noqa: E402

import voice_agent as va  # noqa: E402


def _start_voice_loop():
    asyncio.run(va.main())


# ---- Carbon global hotkey ----
carbon = ctypes.CDLL(ctypes.util.find_library("Carbon"))


class EventTypeSpec(Structure):
    _fields_ = [("eventClass", c_uint32), ("eventKind", c_uint32)]


class EventHotKeyID(Structure):
    _fields_ = [("signature", c_uint32), ("id", c_uint32)]


def _fourcc(s):
    return (ord(s[0]) << 24) | (ord(s[1]) << 16) | (ord(s[2]) << 8) | ord(s[3])


kEventClassKeyboard = _fourcc("keyb")
kEventHotKeyPressed = 5
kEventHotKeyReleased = 6

carbon.GetApplicationEventTarget.restype = c_void_p
carbon.GetApplicationEventTarget.argtypes = []
carbon.GetEventKind.restype = c_uint32
carbon.GetEventKind.argtypes = [c_void_p]

HandlerProc = CFUNCTYPE(c_int32, c_void_p, c_void_p, c_void_p)

# CRITICAL: declare argtypes so 64-bit pointers aren't truncated to int (segfault).
carbon.InstallEventHandler.restype = c_int32
carbon.InstallEventHandler.argtypes = [
    c_void_p, HandlerProc, c_uint32, POINTER(EventTypeSpec), c_void_p, POINTER(c_void_p),
]
carbon.RegisterEventHotKey.restype = c_int32
carbon.RegisterEventHotKey.argtypes = [
    c_uint32, c_uint32, EventHotKeyID, c_void_p, c_uint32, POINTER(c_void_p),
]


def _handler(next_handler, event, user_data):
    kind = carbon.GetEventKind(event)
    if kind == kEventHotKeyPressed:
        va.key_events.put("down")
    elif kind == kEventHotKeyReleased:
        va.key_events.put("up")
    return 0


_handler_cb = HandlerProc(_handler)  # keep a global ref so it isn't GC'd


def _register_hotkey():
    target = carbon.GetApplicationEventTarget()
    types = (EventTypeSpec * 2)(
        EventTypeSpec(kEventClassKeyboard, kEventHotKeyPressed),
        EventTypeSpec(kEventClassKeyboard, kEventHotKeyReleased),
    )
    handler_ref = c_void_p()
    carbon.InstallEventHandler(
        target, _handler_cb, 2, types, None, byref(handler_ref)
    )
    hk_id = EventHotKeyID(_fourcc("vchk"), 1)
    hk_ref = c_void_p()
    status = carbon.RegisterEventHotKey(
        c_uint32(KEYCODE), c_uint32(MODIFIERS), hk_id, target, 0, byref(hk_ref)
    )
    return status


def main():
    import os
    import subprocess

    from AppKit import NSApplication, NSApplicationActivationPolicyAccessory

    # voice loop (websocket + audio) runs in a daemon thread
    threading.Thread(target=_start_voice_loop, daemon=True).start()

    # waveform HUD (separate process; click-through, can't steal focus). Optional.
    if "--no-hud" not in sys.argv:
        here = os.path.dirname(os.path.abspath(__file__))
        try:
            subprocess.Popen([sys.executable, os.path.join(here, "overlay.py")],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:  # noqa: BLE001
            pass

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    status = _register_hotkey()
    pretty = COMBO.replace("opt", "⌥").replace("ctrl", "⌃").replace("cmd", "⌘").upper()
    if status == 0:
        print(f"✅ global hotkey registered: hold [{pretty}] to talk (release to send)")
    else:
        print(f"⚠ RegisterEventHotKey returned {status} (combo may be taken — try --combo)")
    print("   safe: no event tap, no permission, mic grabbed only while held.")
    app.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye.")
