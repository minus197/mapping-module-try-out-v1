"""
pathfinding/scorer.py  —  Sprint 6
-------------------------------------
Composite path quality scoring for visually impaired navigation.

Three scoring dimensions (all normalised 0.0–1.0):
  safety_score    — weighted mean of edge safety_scores
                    (corridor width + obstacle clearance, set by graph_builder)
  shore_score     — fraction of edges that are shore_linable
                    (path keeps user in contact with a wall surface)
  landmark_score  — mean of edge landmark_scores
                    (density of navigation reference points along the route)

Selection strategy (Sprint 6):
  1. Score all K candidate paths.
  2. Compute a composite score = w_s·safety + w_sh·shore + w_l·landmark.
  3. Paths with composite ≥ TOP_TIER_THRESHOLD are "top tier".
  4. Among top-tier paths, select the one with minimum total distance.
  5. Return top path + up to MAX_ALTERNATIVES runner-ups.

Weights and thresholds are tunable — set them during Sprint 8 calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from shared.types import NavigationEdge

# ── Tunable weights (Sprint 8 calibration) ───────────────────────────────────
W_SAFETY    = 0.40
W_SHORE     = 0.35
W_LANDMARK  = 0.25

TOP_TIER_THRESHOLD = 0.55   # minimum composite score to be "top tier"
MAX_ALTERNATIVES   = 3


@dataclass
class ScoredPath:
    edges:          List[NavigationEdge]
    total_distance: float
    safety_score:   float
    shore_score:    float
    landmark_score: float
    composite:      float


def score_path(edges: List[NavigationEdge]) -> ScoredPath:
    """Compute all three quality scores for a single path."""
    if not edges:
        return ScoredPath([], 0.0, 0.0, 0.0, 0.0, 0.0)

    safety   = _weighted_mean([e.safety_score          for e in edges], [e.distance for e in edges])
    shore    = _weighted_mean([float(e.shore_linable)  for e in edges], [e.distance for e in edges])
    landmark = _weighted_mean([e.landmark_score        for e in edges], [e.distance for e in edges])
    composite = W_SAFETY * safety + W_SHORE * shore + W_LANDMARK * landmark
    total_d   = sum(e.distance for e in edges)

    return ScoredPath(
        edges          = edges,
        total_distance = round(total_d, 3),
        safety_score   = round(safety,   3),
        shore_score    = round(shore,    3),
        landmark_score = round(landmark, 3),
        composite      = round(composite, 3),
    )


def select_best(scored_paths: List[ScoredPath],
                ) -> Tuple[ScoredPath, List[ScoredPath]]:
    """
    Select the best path and up to MAX_ALTERNATIVES runner-ups.

    Returns
    -------
    (best, alternatives)
    """
    if not scored_paths:
        raise ValueError("No scored paths to select from.")

    # Sort by composite descending, then distance ascending
    ranked = sorted(scored_paths,
                    key=lambda p: (-p.composite, p.total_distance))

    top_composite = ranked[0].composite
    threshold     = max(TOP_TIER_THRESHOLD, top_composite * 0.85)

    top_tier = [p for p in ranked if p.composite >= threshold]
    # Among top tier: pick shortest
    best = min(top_tier, key=lambda p: p.total_distance)
    alternatives = [p for p in ranked if p is not best][:MAX_ALTERNATIVES]

    return best, alternatives


# ── Helpers ───────────────────────────────────────────────────────────────────

def _weighted_mean(values: List[float], weights: List[float]) -> float:
    total_w = sum(weights)
    if total_w == 0:
        return sum(values) / len(values) if values else 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_w
