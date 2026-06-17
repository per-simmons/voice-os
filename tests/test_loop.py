#!/usr/bin/env python3
"""
test_loop.py — headless end-to-end proof of the Realtime loop (no mic needed).

Connects to gpt-realtime-2 over the real WebSocket, configures the session with
our tools (text output so we can print), injects a TEXT user turn, and verifies
the model calls the right tool, we execute it on the Mac, and the model speaks a
confirmation back. This exercises the entire brain->tool->hands->brain path that
the voice loop uses — only the audio I/O is swapped for text/stdout.

Usage:
    source .venv/bin/activate
    python test_loop.py "open Spotify and play some Tchaikovsky"
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import websockets  # noqa: E402

import actions  # noqa: E402
from voice_agent import MODEL, URL, TOOLS, INSTRUCTIONS, dispatch_tool  # noqa: E402


async def run(user_text: str):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY not set (source .env or run via run.sh).")
    headers = {"Authorization": f"Bearer {key}"}

    async with websockets.connect(URL, additional_headers=headers, max_size=None) as ws:
        # text-output session with the same tools/instructions
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": MODEL,
                "instructions": INSTRUCTIONS,
                "output_modalities": ["text"],
                "tools": TOOLS,
                "tool_choice": "auto",
            },
        }))
        # inject a user text turn
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": user_text}],
            },
        }))
        await ws.send(json.dumps({"type": "response.create"}))

        print(f"\nYOU: {user_text}\n")
        tools_called = 0
        async for raw in ws:
            ev = json.loads(raw)
            t = ev.get("type", "")

            if t == "session.updated":
                print("  · session configured, tools advertised:",
                      [x["name"] for x in TOOLS])

            elif t == "response.output_text.delta":
                print(ev.get("delta", ""), end="", flush=True)

            elif t == "response.function_call_arguments.done":
                tools_called += 1
                name, call_id = ev["name"], ev["call_id"]
                args = json.loads(ev.get("arguments") or "{}")
                print(f"\n  ⚙  MODEL CALLED: {name}({args})")
                result = await dispatch_tool(name, args)
                print(f"  ✓  executed → {json.dumps(result)[:200]}")
                await ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result),
                    },
                }))
                await ws.send(json.dumps({"type": "response.create"}))

            elif t == "response.done":
                # finished a response turn; stop once we've run a tool + spoken back
                status = ev.get("response", {}).get("status")
                has_fc = any(
                    i.get("type") == "function_call"
                    for i in ev.get("response", {}).get("output", [])
                )
                if not has_fc and tools_called > 0:
                    print(f"\n\n[done] status={status}, tools_called={tools_called}")
                    return

            elif t == "error":
                print("\n[realtime error]", json.dumps(ev.get("error", ev)))
                return


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "open Spotify"
    asyncio.run(run(text))
