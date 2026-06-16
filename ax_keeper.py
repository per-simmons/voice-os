"""Keeps Electron apps' (Claude) accessibility tree forced ON. Must run in a
trusted context. Re-forces every few seconds so the tree survives app restarts."""
import subprocess, time
from ApplicationServices import AXUIElementCreateApplication, AXUIElementSetAttributeValue
while True:
    for app in ("Claude",):
        pids = subprocess.run(["pgrep","-x",app],capture_output=True,text=True).stdout.split()
        for p in pids:
            try:
                AXUIElementSetAttributeValue(AXUIElementCreateApplication(int(p)),"AXManualAccessibility",True)
            except Exception:
                pass
    time.sleep(4)
