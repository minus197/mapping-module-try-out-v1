"""
pathfinding/node_resolver.py  —  Sprint 5
------------------------------------------
Resolves a free-text destination query to a NavigationNode in the FloorGraph.

Three-tier resolution strategy (run in order, return on first confident match):
  Tier 1 — Sentence embedding similarity  (MiniLM, cosine ≥ 0.65)
  Tier 2 — Fuzzy string match             (rapidfuzz, score ≥ 75)
  Tier 3 — Category keyword match         (exact substring)

The sentence embedding model is loaded lazily on first use and cached.
In environments without sentence-transformers (e.g. CI, Edge device testing),
the resolver falls back automatically to Tiers 2 and 3.

Usage
-----
    from pathfinding.node_resolver import NodeResolver
    from shared.types import FloorGraph

    resolver = NodeResolver(graph)
    node = resolver.resolve("I want to go to the food court")
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from shared.types import FloorGraph, NavigationNode

# ── Optional heavy imports ────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    _EMBEDDINGS_AVAILABLE = False

try:
    from rapidfuzz import fuzz, process as rfprocess
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    import difflib
    _RAPIDFUZZ_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────
_EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
_EMBED_THRESHOLD   = 0.65   # cosine similarity minimum
_FUZZY_THRESHOLD   = 75     # rapidfuzz WRatio minimum (0–100)

_CATEGORY_ALIASES: Dict[str, List[str]] = {
    "shop":       ["shop", "store", "retail", "brand"],
    "food_court": ["food", "eat", "lunch", "dinner", "cafe", "restaurant"],
    "restroom":   ["toilet", "wc", "bathroom", "restroom", "washroom"],
    "entrance":   ["entrance", "entry", "lobby", "start"],
    "exit":       ["exit", "out", "leave", "emergency"],
    "corridor":   ["corridor", "hallway", "passage"],
}

_NODE_TYPE_PRIORITY = {
    "elevator":    3,
    "escalator":   3,
    "stair":       3,
    "door":        2,
    "landmark":    2,
    "zone_centroid": 1,
    "junction":    0,
}


class NodeResolver:
    """
    Resolves free-text destination queries to FloorGraph nodes.

    Parameters
    ----------
    graph : FloorGraph
    """

    def __init__(self, graph: FloorGraph):
        self.graph   = graph
        self._model  = None           # lazy-loaded SentenceTransformer
        self._node_labels: List[str]          = []
        self._node_list:   List[NavigationNode] = []
        self._label_embeddings = None
        self._build_label_index()

    # ── Public ────────────────────────────────────────────────────────────────

    def resolve(self, query: str,
                exclude_types: Optional[List[str]] = None) -> Optional[NavigationNode]:
        """
        Resolve a free-text query to the best-matching NavigationNode.

        Parameters
        ----------
        query         : str  — e.g. "I want to go to Nike"
        exclude_types : list — node_type values to exclude from results

        Returns
        -------
        Best matching NavigationNode, or None if no match exceeds thresholds.
        """
        candidates = [
            n for n in self.graph.nodes
            if (exclude_types is None or n.node_type not in exclude_types)
        ]
        if not candidates:
            return None

        clean_query = _extract_location_phrase(query)

        # Tier 1: embedding similarity
        if _EMBEDDINGS_AVAILABLE:
            result = self._resolve_by_embedding(clean_query, candidates)
            if result is not None:
                return result

        # Tier 2: fuzzy string match
        result = self._resolve_by_fuzzy(clean_query, candidates)
        if result is not None:
            return result

        # Tier 3: category keyword match
        return self._resolve_by_category(clean_query, candidates)

    # ── Tier 1 ────────────────────────────────────────────────────────────────

    def _resolve_by_embedding(self,
                               query: str,
                               candidates: List[NavigationNode]
                               ) -> Optional[NavigationNode]:
        """[Sprint 5 — TODO: implement full embedding resolution]"""
        # Stub: return None so Tier 2 always runs in this sprint
        return None

    # ── Tier 2 ────────────────────────────────────────────────────────────────

    def _resolve_by_fuzzy(self,
                          query: str,
                          candidates: List[NavigationNode]
                          ) -> Optional[NavigationNode]:
        labels = [n.label for n in candidates]
        q_low  = query.lower()

        if _RAPIDFUZZ_AVAILABLE:
            result = rfprocess.extractOne(
                q_low,
                [l.lower() for l in labels],
                scorer=fuzz.WRatio
            )
            if result and result[1] >= _FUZZY_THRESHOLD:
                return candidates[result[2]]
        else:
            # difflib fallback
            matches = difflib.get_close_matches(q_low,
                                                [l.lower() for l in labels],
                                                n=1, cutoff=0.5)
            if matches:
                idx = [l.lower() for l in labels].index(matches[0])
                return candidates[idx]
        return None

    # ── Tier 3 ────────────────────────────────────────────────────────────────

    def _resolve_by_category(self,
                              query: str,
                              candidates: List[NavigationNode]
                              ) -> Optional[NavigationNode]:
        q_low = query.lower()
        for category, aliases in _CATEGORY_ALIASES.items():
            if any(alias in q_low for alias in aliases):
                # Return highest-priority node matching this category
                matches = [
                    n for n in candidates
                    if n.tags.get("category") == category
                ]
                if matches:
                    return max(matches,
                               key=lambda n: _NODE_TYPE_PRIORITY.get(n.node_type, 0))
        return None

    # ── Index ─────────────────────────────────────────────────────────────────

    def _build_label_index(self) -> None:
        self._node_list   = list(self.graph.nodes)
        self._node_labels = [n.label for n in self._node_list]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_location_phrase(query: str) -> str:
    """
    Strip navigation intent preamble to extract the core location name.
    e.g. "I want to go to the food court" → "food court"
         "Navigate me to Nike"            → "Nike"
         "Food court"                     → "Food court"
    """
    patterns = [
        r"(?:i want to go to|take me to|navigate (?:me )?to|"
        r"go to|find|where is|i need to get to|lead me to)\s+(?:the\s+)?(.+)",
    ]
    q = query.strip()
    for pat in patterns:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return q
