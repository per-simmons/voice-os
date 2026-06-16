#!/usr/bin/env python3
"""
overlay.py — native click-through waveform HUD (black & white) at the top of the
screen while you talk to chat.

Uses a borderless, transparent, always-on-top NSWindow with
setIgnoresMouseEvents_(True), so it physically CANNOT intercept clicks or steal
focus (the earlier tkinter version did — this one can't). Reads
/tmp/voiceos-hud.json ({active, level}) which voice_agent.py writes.

Run:  python overlay.py        (needs pyobjc-framework-Cocoa)
"""
import json
import math

import objc
from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSScreen,
    NSScreenSaverWindowLevel,
    NSTimer,
    NSView,
    NSWindow,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorIgnoresCycle,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskBorderless,
)
from Foundation import NSMakeRect

HUD_FILE = "/tmp/voiceos-hud.json"
W, H = 760, 110
BARS = 60


def read_hud():
    try:
        with open(HUD_FILE) as f:
            d = json.load(f)
        return bool(d.get("active")), float(d.get("level", 0.0))
    except (OSError, ValueError):
        return False, 0.0


class WaveView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(WaveView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.levels = [0.0] * BARS
        self.disp = 0.0
        self.alpha = 0.0
        self.pulse = 0.0
        return self

    def isFlipped(self):
        return False

    def drawRect_(self, rect):
        active, level = read_hud()
        target = 0.94 if active else 0.0
        self.alpha += (target - self.alpha) * 0.25
        self.disp += (level - self.disp) * 0.35
        self.levels.append(self.disp if active else 0.0)
        del self.levels[0]
        self.pulse = (self.pulse + 0.12) % (2 * math.pi)

        b = self.bounds()
        NSColor.clearColor().set()
        NSBezierPath.fillRect_(b)
        if self.alpha <= 0.02:
            return

        a = self.alpha
        # rounded dark container
        bg = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(2, 2, b.size.width - 4, b.size.height - 4), 26, 26
        )
        NSColor.colorWithCalibratedWhite_alpha_(0.04, 0.9 * a).set()
        bg.fill()
        NSColor.colorWithCalibratedWhite_alpha_(0.18, a).set()
        bg.setLineWidth_(1.0)
        bg.stroke()

        cy = b.size.height / 2
        x0 = 116
        span = b.size.width - x0 - 30
        bw = span / BARS
        for i, lv in enumerate(self.levels):
            env = 0.30 + 0.70 * (1 - abs(i - BARS / 2) / (BARS / 2))
            h = max(3, lv * env * (b.size.height * 0.60))
            x = x0 + i * bw + bw * 0.5
            white = 1.0 if lv > 0.04 else 0.30
            NSColor.colorWithCalibratedWhite_alpha_(white, a).set()
            bar = NSBezierPath.bezierPath()
            bar.setLineWidth_(max(2.0, bw * 0.55))
            bar.setLineCapStyle_(1)  # round
            bar.moveToPoint_((x, cy - h / 2))
            bar.lineToPoint_((x, cy + h / 2))
            bar.stroke()

        # pulsing dot + "chat" label
        glow = 0.5 + 0.5 * math.sin(self.pulse)
        rr = 5 + 2 * glow
        NSColor.colorWithCalibratedWhite_alpha_(1.0, a).set()
        dot = NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(40 - rr, cy - rr, 2 * rr, 2 * rr)
        )
        dot.fill()
        self._draw_label("chat", 60, cy, a)

    def _draw_label(self, text, x, cy, a):
        from AppKit import (
            NSFont,
            NSColor as C,
            NSFontAttributeName,
            NSForegroundColorAttributeName,
        )
        from Foundation import NSString
        attrs = {
            NSFontAttributeName: NSFont.boldSystemFontOfSize_(16),
            NSForegroundColorAttributeName: C.colorWithCalibratedWhite_alpha_(1.0, a),
        }
        NSString.stringWithString_(text).drawAtPoint_withAttributes_((x, cy - 11), attrs)


class Controller(objc.lookUpClass("NSObject")):
    def tick_(self, timer):
        self.view.setNeedsDisplay_(True)


def main():
    import os

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    # Pick which monitor to show the waveform on. VOICEOS_DISPLAY is 1-based
    # (1 = primary, 2 = second monitor). Defaults to display 2 if it exists.
    screens = NSScreen.screens()
    want = int(os.environ.get("VOICEOS_DISPLAY", "2"))
    idx = want - 1
    if idx < 0 or idx >= len(screens):
        idx = len(screens) - 1  # fall back to the last available screen
    fr = screens[idx].frame()
    x = fr.origin.x + (fr.size.width - W) / 2
    y = fr.origin.y + fr.size.height - H - 24  # top of THAT screen
    rect = NSMakeRect(x, y, W, H)

    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        rect, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False
    )
    win.setOpaque_(False)
    win.setBackgroundColor_(NSColor.clearColor())
    win.setLevel_(NSScreenSaverWindowLevel)
    win.setIgnoresMouseEvents_(True)   # <-- click-through: can't steal clicks/focus
    win.setCollectionBehavior_(
        NSWindowCollectionBehaviorCanJoinAllSpaces
        | NSWindowCollectionBehaviorStationary
        | NSWindowCollectionBehaviorIgnoresCycle
    )
    win.setHasShadow_(False)

    view = WaveView.alloc().initWithFrame_(NSMakeRect(0, 0, W, H))
    win.setContentView_(view)
    win.orderFrontRegardless()

    ctrl = Controller.alloc().init()
    ctrl.view = view
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        1.0 / 30, ctrl, "tick:", None, True
    )

    app.run()


if __name__ == "__main__":
    main()
