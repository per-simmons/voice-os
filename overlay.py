#!/usr/bin/env python3
"""
overlay.py — a cool black & white waveform HUD that appears at the top of the
screen while you're talking to chat. Frameless, always-on-top, click-through-ish
(no focus steal). Driven by /tmp/voiceos-hud.json which voice_agent.py writes
({active, level}). Fades in while the mic is going to chat, fades out otherwise.

Run alongside the agent (run.sh does this for you), or standalone:
    python overlay.py
"""
import json
import tkinter as tk

HUD_FILE = "/tmp/voiceos-hud.json"
W, H = 760, 96
BARS = 60
BG = "#000000"
FG = "#ffffff"
DIM = "#3a3a3a"

root = tk.Tk()
root.overrideredirect(True)
root.attributes("-topmost", True)
try:
    root.attributes("-alpha", 0.0)  # start hidden
except tk.TclError:
    pass
sw = root.winfo_screenwidth()
root.geometry(f"{W}x{H}+{(sw - W) // 2}+24")
cv = tk.Canvas(root, width=W, height=H, bg=BG, highlightthickness=0)
cv.pack()

levels = [0.0] * BARS
disp = 0.0
cur_alpha = 0.0
pulse = 0.0


def round_rect(c, x1, y1, x2, y2, r, **kw):
    pts = [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
        x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]
    return c.create_polygon(pts, smooth=True, **kw)


def read_hud():
    try:
        with open(HUD_FILE) as f:
            d = json.load(f)
        return bool(d.get("active")), float(d.get("level", 0.0))
    except (OSError, ValueError):
        return False, 0.0


def tick():
    global disp, cur_alpha, pulse
    active, level = read_hud()

    # smooth target alpha (fade in/out)
    target_alpha = 0.94 if active else 0.0
    cur_alpha += (target_alpha - cur_alpha) * 0.25
    try:
        root.attributes("-alpha", max(0.0, min(0.94, cur_alpha)))
    except tk.TclError:
        pass

    disp += (level - disp) * 0.35
    levels.append(disp if active else 0.0)
    del levels[0]
    pulse = (pulse + 0.12) % 6.2831853

    cv.delete("all")
    if cur_alpha > 0.02:
        round_rect(cv, 2, 2, W - 2, H - 2, 26, fill="#0b0b0b", outline="#222222")
        cx = W / 2
        cy = H / 2
        bw = (W - 150) / BARS
        x0 = 120
        for i, lv in enumerate(levels):
            # envelope: taller toward the center, tapered at the edges
            env = 0.30 + 0.70 * (1 - abs(i - BARS / 2) / (BARS / 2))
            h = max(3, lv * env * (H * 0.62))
            x = x0 + i * bw + bw * 0.5
            shade = FG if lv > 0.04 else DIM
            cv.create_line(
                x, cy - h / 2, x, cy + h / 2,
                fill=shade, width=max(2, int(bw * 0.55)), capstyle="round",
            )
        # left: pulsing dot + label
        import math
        glow = 0.5 + 0.5 * math.sin(pulse)
        rr = 5 + 2 * glow
        cv.create_oval(40 - rr, cy - rr, 40 + rr, cy + rr, fill=FG, outline="")
        cv.create_text(60, cy, text="chat", fill=FG, anchor="w",
                       font=("Helvetica Neue", 16, "bold"))

    root.after(30, tick)


root.after(60, tick)
root.mainloop()
