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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(score: float, cap_id: str = "test") -> retrieval.SearchResult:
    cap = retrieval.Capability(
        id=cap_id, description="d", examples=["e"], primitive="p", template="t"
    )
    return retrieval.SearchResult(capability=cap, score=score, matched_example="e")


# ---------------------------------------------------------------------------
# Paraphrase coverage — every row is a realistic spoken variation
# ---------------------------------------------------------------------------

# Core capabilities verified against the current builtin set.
VARIATIONS = [
    # app control
    ("launch Google Chrome",                "app-open"),
    ("switch over to Finder",               "app-open"),
    ("bring up Terminal",                   "app-open"),
    # Spotify
    ("pause the track",                     "spotify-pause"),
    ("resume the music",                    "spotify-pause"),
    ("unpause Spotify",                     "spotify-pause"),
    ("throw on some jazz",                  "spotify-play-search"),
    ("search Spotify for ambient music",    "spotify-play-search"),
    # web
    ("google the weather in Auckland",      "web-search"),
    ("look up the Python docs",             "web-search"),
    ("search online for a plumber",         "web-search"),
    # OBS
    ("start screen recording in OBS",       "obs-start-recording"),
    ("begin recording",                      "obs-start-recording"),
    ("stop the recording",                  "obs-stop-recording"),
    ("finish recording",                    "obs-stop-recording"),
    ("switch OBS to my talking head scene", "obs-switch-scene"),
    # Premiere — editing
    ("chop the clip here",                  "premiere-cut"),
    ("split here",                          "premiere-cut"),
    ("cut all tracks at the playhead",      "premiere-cut-all"),
    ("razor all tracks",                    "premiere-cut-all"),
    ("save my project",                     "premiere-save"),
    ("save Premiere",                       "premiere-save"),
    ("step back one frame",                 "premiere-step-back"),
    ("rewind a frame",                      "premiere-step-back"),
    ("move forward a frame",               "premiere-step-forward"),
    ("next frame",                          "premiere-step-forward"),
    ("pause Premiere",                      "premiere-play-pause"),
    ("play from here",                      "premiere-play-pause"),
    ("undo the last cut",                   "premiere-undo"),
    ("undo that",                           "premiere-undo"),
    ("set the in point here",               "premiere-mark-in"),
    ("mark the out point",                  "premiere-mark-out"),
    ("ripple delete this",                  "premiere-ripple-delete"),
    ("delete and close the gap",            "premiere-ripple-delete"),
    # screen reading
    ("read out what's on screen",           "read-screen"),
    ("what did Claude just say",            "read-screen"),
    # notes
    ("jot this down as a note",             "take-note"),
    ("write this down",                     "take-note"),
    # terminal
    ("run a command in the terminal",       "terminal-run"),
    ("start a Claude Code session",         "terminal-run"),
]


@pytest.mark.parametrize("query,expected_id", VARIATIONS)
def test_variation_retrieves_correct_capability(index, query, expected_id):
    results = index.search(query, top_k=3)
    assert results, f"no results for {query!r}"
    top = results[0]
    assert top.capability.id == expected_id, (
        f"{query!r} -> {[(r.capability.id, round(r.score, 2)) for r in results]}"
    )
    assert index.grounding(results) == "STRONG", f"{query!r} matched but grounding was WEAK"


def test_click_link_variation_maps_to_a_click_link_capability(index):
    # there are per-browser click-link caps (safari/chrome); either is correct.
    results = index.search("open the first search result", top_k=3)
    assert results[0].capability.id.startswith("click-link")
    assert index.grounding(results) == "STRONG"


# ---------------------------------------------------------------------------
# Off-topic chatter must stay WEAK (the OS must not fire on noise)
# ---------------------------------------------------------------------------

OFF_TOPIC = [
    "how are you feeling today",
    "what time is it",
    "tell me a joke",
    "that sounds really interesting",
    "thanks for your help",
    "the quick brown fox",
]


@pytest.mark.parametrize("query", OFF_TOPIC)
def test_off_topic_chatter_is_weak(index, query):
    results = index.search(query, top_k=3)
    assert index.grounding(results) == "WEAK", (
        f"{query!r} should be WEAK but got STRONG "
        f"(top: {results[0].capability.id!r} @ {results[0].score:.3f})"
    )


# ---------------------------------------------------------------------------
# Grounding unit tests — verify threshold logic with known scores
# These don't depend on the embedding model; they test the decision rule.
# ---------------------------------------------------------------------------

def test_grounding_strong_by_absolute_threshold(index):
    # any top score >= 0.52 is STRONG regardless of runner-up
    assert index.grounding([_make_result(0.52)]) == "STRONG"
    assert index.grounding([_make_result(0.60)]) == "STRONG"
    assert index.grounding([_make_result(0.52), _make_result(0.51)]) == "STRONG"


def test_grounding_strong_by_dominance(index):
    # top < 0.52 but >= 0.40 and clearly dominates runner-up (>= runner-up * 1.35)
    # 0.45 >= 0.30 * 1.35 = 0.405 → STRONG
    assert index.grounding([_make_result(0.45), _make_result(0.30)]) == "STRONG"
    # single result with 0.50 — second defaults to 0.0, so 0.50 >= 0.0 * 1.35 → STRONG
    assert index.grounding([_make_result(0.50)]) == "STRONG"


def test_grounding_weak_when_scores_are_too_close(index):
    # top < 0.52 and runner-up is close: 0.45 < 0.40 * 1.35 = 0.54 → WEAK
    assert index.grounding([_make_result(0.45), _make_result(0.40)]) == "WEAK"


def test_grounding_weak_when_top_score_is_low(index):
    # top below 0.40 can never be STRONG
    assert index.grounding([_make_result(0.35), _make_result(0.10)]) == "WEAK"
    assert index.grounding([_make_result(0.39)]) == "WEAK"


def test_grounding_weak_for_empty_results(index):
    assert index.grounding([]) == "WEAK"


# ---------------------------------------------------------------------------
# format_context output shape
# ---------------------------------------------------------------------------

def test_format_context_strong_contains_expected_sections(index):
    results = index.search("pause the music", top_k=3)
    ctx = index.format_context(results, "STRONG")
    assert "RETRIEVED CAPABILITIES" in ctx
    assert "grounding: STRONG" in ctx
    assert "spotify-pause" in ctx
    # strong hint tells the model to use the retrieved capability
    assert "use them" in ctx.lower()


def test_format_context_weak_contains_clarification_hint(index):
    results = index.search("pause the music", top_k=3)
    ctx = index.format_context(results, "WEAK")
    assert "WEAK" in ctx
    assert "clarification" in ctx.lower()


def test_format_context_includes_all_top_k_capabilities(index):
    results = index.search("cut the clip", top_k=3)
    ctx = index.format_context(results, "STRONG")
    # each result should appear as a block
    for r in results:
        assert r.capability.id in ctx


def test_format_context_empty_results_returns_empty_string(index):
    assert index.format_context([], "WEAK") == ""
    assert index.format_context([], "STRONG") == ""


# ---------------------------------------------------------------------------
# Search mechanics
# ---------------------------------------------------------------------------

def test_search_top_k_limits_results(index):
    for k in (1, 2, 3):
        results = index.search("pause the music", top_k=k)
        assert len(results) <= k


def test_search_results_are_ordered_by_score_descending(index):
    results = index.search("open Spotify", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_matched_example_is_a_string(index):
    results = index.search("pause the music", top_k=3)
    for r in results:
        assert isinstance(r.matched_example, str)
        assert len(r.matched_example) > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
