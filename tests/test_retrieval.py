#!/usr/bin/env python3
"""
test_retrieval.py — the heart of the matter: spoken VARIATIONS must retrieve the
right capability template.

This is what makes the voice OS forgiving: a user never says the exact example
phrase, so each command is a paraphrase. These tests assert that realistic
paraphrases land on the correct capability as the top hit with STRONG grounding,
and that off-topic chatter stays WEAK (so the OS doesn't fire on noise).

Needs the local embedding model (sentence-transformers, cached after first run).

    pytest test_retrieval.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402

pytest.importorskip("sentence_transformers")

import retrieval  # noqa: E402


@pytest.fixture(scope="module")
def index():
    return retrieval.CapabilityIndex(verbose=False)


# realistic paraphrases a user might actually speak -> the capability that should win.
# (Verified against the current builtin capability set; each is the top hit.)
VARIATIONS = [
    ("launch Google Chrome",                "app-open"),
    ("pause the track",                     "spotify-pause"),
    ("resume the music",                    "spotify-pause"),
    ("google the weather in Auckland",      "web-search"),
    ("start screen recording in OBS",       "obs-start-recording"),
    ("stop the recording",                  "obs-stop-recording"),
    ("switch OBS to my talking head scene", "obs-switch-scene"),
    ("chop the clip here",                  "premiere-cut"),
    ("save my project",                     "premiere-save"),
    ("step back one frame",                 "premiere-step-back"),
    ("read out what's on screen",           "read-screen"),
    ("jot this down as a note",             "take-note"),
]


@pytest.mark.parametrize("query,expected_id", VARIATIONS)
def test_variation_retrieves_correct_capability(index, query, expected_id):
    results = index.search(query, top_k=3)
    assert results, f"no results for {query!r}"
    top = results[0]
    assert top.capability.id == expected_id, (
        f"{query!r} -> {[(r.capability.id, round(r.score, 2)) for r in results]}"
    )
    assert index.grounding(results) == "STRONG", f"{query!r} matched but grounding was weak"


def test_click_link_variation_maps_to_a_click_link_capability(index):
    # there are per-browser click-link caps (safari/chrome); either is correct.
    results = index.search("open the first search result", top_k=3)
    assert results[0].capability.id.startswith("click-link")
    assert index.grounding(results) == "STRONG"


def test_off_topic_chatter_is_weak(index):
    # a voice OS must NOT confidently fire on conversation that isn't a command.
    results = index.search("how are you feeling today", top_k=3)
    assert index.grounding(results) == "WEAK"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
