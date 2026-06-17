"""
retrieval.py — local capability retrieval for voice-os.

Loads capabilities.json (+ optional user overlay), embeds every example query
using a local sentence-transformers model, and exposes cosine search so the
voice agent can inject the most relevant capabilities into the model's context
on every turn — instead of handing the model a flat list of 12 named tools.

The embedding model (all-MiniLM-L6-v2, ~22 MB) runs fully locally. No API
calls, no cost at idle.

Usage:
    index = CapabilityIndex()       # loads + embeds at startup (~0.5 s first run)
    results = index.search("cut the clip in Premiere", top_k=3)
    context = index.format_context(results)  # inject this into the model turn
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import config

# src/ lives one level under the project root; memory/ sits at the root.
_ROOT = Path(__file__).resolve().parent.parent
_CAPABILITIES_PATH = _ROOT / "memory" / "capabilities.json"
_USER_CAPABILITIES_PATH = _ROOT / "memory" / "capabilities.user.json"
_EMBED_CACHE = _ROOT / "memory" / "embeddings.npy"
_EMBED_IDS_CACHE = _ROOT / "memory" / "embedding_ids.json"

_MODEL_NAME = os.environ.get("VOICEOS_EMBED_MODEL", "all-MiniLM-L6-v2")


@dataclass
class Capability:
    id: str
    description: str
    examples: list[str]
    primitive: str
    template: Any
    source: str = "builtin"  # "builtin" | "learned" | "user"

    def to_context_block(self) -> str:
        """Format this capability as a context block for the model."""
        tmpl = (
            json.dumps(self.template, ensure_ascii=False)
            if isinstance(self.template, dict)
            else self.template
        )
        return (
            f"[{self.id}] {self.description}\n"
            f"  primitive: {self.primitive}\n"
            f"  template: {tmpl}"
        )


@dataclass
class SearchResult:
    capability: Capability
    score: float
    matched_example: str


class CapabilityIndex:
    """Loads capabilities, embeds examples, supports cosine search."""

    def __init__(self, verbose: bool = False):
        self._verbose = verbose
        self._caps: list[Capability] = []
        # parallel arrays: one row per (capability, example) pair
        self._embeddings: np.ndarray | None = None
        self._index: list[tuple[int, str]] = []  # (cap_idx, example_text)
        self._model = None
        self._load_capabilities()
        self._build_index()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_capabilities(self) -> None:
        caps: list[Capability] = []

        def _load_file(path: Path, source: str) -> None:
            if not path.exists():
                return
            try:
                with open(path) as f:
                    data = json.load(f)
                for item in data:
                    caps.append(Capability(
                        id=item["id"],
                        description=item["description"],
                        examples=item.get("examples", []),
                        primitive=item["primitive"],
                        template=item.get("template", ""),
                        source=source,
                    ))
            except Exception as e:  # noqa: BLE001
                print(f"[retrieval] warning: could not load {path}: {e}", flush=True)

        _load_file(_CAPABILITIES_PATH, "builtin")
        _load_file(_USER_CAPABILITIES_PATH, "user")

        # user entries override builtin ones with the same id
        seen: dict[str, Capability] = {}
        for cap in caps:
            seen[cap.id] = cap  # last write wins (user overrides builtin)
        self._caps = list(seen.values())

        if self._verbose:
            learned = sum(1 for c in self._caps if c.source in ("learned", "user"))
            print(f"[retrieval] loaded {len(self._caps)} capabilities "
                  f"({learned} learned/user)", flush=True)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
            t0 = time.monotonic()
            self._model = SentenceTransformer(_MODEL_NAME)
            if self._verbose:
                print(f"[retrieval] model loaded in {time.monotonic()-t0:.1f}s", flush=True)
        return self._model

    def _build_index(self) -> None:
        """Build the (example → embedding) index, using cache if valid."""
        # flatten all (cap_idx, example) pairs
        pairs: list[tuple[int, str]] = []
        for i, cap in enumerate(self._caps):
            for ex in cap.examples:
                pairs.append((i, ex))

        if not pairs:
            self._embeddings = np.empty((0, 384), dtype=np.float32)
            self._index = []
            return

        # check cache validity: same ids + same examples
        ids_snapshot = {c.id: c.examples for c in self._caps}
        cache_valid = False
        if _EMBED_CACHE.exists() and _EMBED_IDS_CACHE.exists():
            try:
                with open(_EMBED_IDS_CACHE) as f:
                    cached_ids = json.load(f)
                cache_valid = cached_ids == ids_snapshot
            except Exception:  # noqa: BLE001
                pass

        if cache_valid:
            self._embeddings = np.load(str(_EMBED_CACHE))
            self._index = pairs
            if self._verbose:
                print(f"[retrieval] loaded {len(pairs)} embeddings from cache", flush=True)
            # Eagerly warm the model so first-command latency is ~0ms not ~7s
            self._get_model()
            return

        # embed all examples
        t0 = time.monotonic()
        model = self._get_model()
        texts = [ex for _, ex in pairs]
        embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self._embeddings = np.array(embs, dtype=np.float32)
        self._index = pairs

        # save cache
        try:
            _EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(_EMBED_CACHE), self._embeddings)
            with open(_EMBED_IDS_CACHE, "w") as f:
                json.dump(ids_snapshot, f)
        except Exception as e:  # noqa: BLE001
            print(f"[retrieval] warning: could not save embedding cache: {e}", flush=True)

        if self._verbose:
            print(f"[retrieval] embedded {len(texts)} examples in "
                  f"{time.monotonic()-t0:.2f}s", flush=True)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """Return the top_k most relevant capabilities for a spoken query."""
        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        model = self._get_model()
        q_emb = model.encode([query], normalize_embeddings=True)[0]  # (D,)

        # cosine similarity (embeddings are L2-normalised so this is just dot product)
        scores = self._embeddings @ q_emb  # (N,)

        # aggregate per capability: best example score
        cap_best: dict[int, tuple[float, str]] = {}
        for idx, (cap_idx, example) in enumerate(self._index):
            s = float(scores[idx])
            if cap_idx not in cap_best or s > cap_best[cap_idx][0]:
                cap_best[cap_idx] = (s, example)

        ranked = sorted(cap_best.items(), key=lambda x: x[1][0], reverse=True)
        results = []
        for cap_idx, (score, matched_ex) in ranked[:top_k]:
            results.append(SearchResult(
                capability=self._caps[cap_idx],
                score=score,
                matched_example=matched_ex,
            ))
        return results

    def grounding(self, results: list[SearchResult]) -> str:
        """STRONG if top result is clearly relevant, WEAK if vague match."""
        if not results:
            return "WEAK"
        top = results[0].score
        second = results[1].score if len(results) > 1 else 0.0
        # STRONG: high absolute score OR clear dominance over runner-up
        if top >= 0.52 or (top >= 0.40 and top >= second * 1.35):
            return "STRONG"
        return "WEAK"

    def format_context(self, results: list[SearchResult], grounding: str) -> str:
        """Format retrieved capabilities as a context block to inject per-turn."""
        if not results:
            return ""
        blocks = "\n\n".join(r.capability.to_context_block() for r in results)
        hint = (
            "These capabilities directly match the command — use them."
            if grounding == "STRONG"
            else "These are loosely related — ask for clarification if the command is ambiguous."
        )
        return (
            f"RETRIEVED CAPABILITIES (grounding: {grounding})\n"
            f"{hint}\n\n"
            f"{blocks}"
        )

    # ------------------------------------------------------------------
    # Index refresh (called after retrospective adds new capabilities)
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Reload capabilities and rebuild the embedding index."""
        self._caps = []
        self._embeddings = None
        self._index = []
        self._load_capabilities()
        # invalidate cache so new entries get embedded
        _EMBED_CACHE.unlink(missing_ok=True)
        _EMBED_IDS_CACHE.unlink(missing_ok=True)
        self._build_index()
        if self._verbose:
            print("[retrieval] index refreshed", flush=True)


# Module-level singleton — loaded once at startup
_index: CapabilityIndex | None = None


def get_index(verbose: bool = True) -> CapabilityIndex:
    global _index
    if _index is None:
        _index = CapabilityIndex(verbose=verbose)
    return _index
