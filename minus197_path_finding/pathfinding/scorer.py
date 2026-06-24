"""
pathfinding/scorer.py  —  WP2  (Stage 5)
-----------------------------------------
Computes distance-weighted quality scores for a single path (list of edges).

Three dimensions — all in [0, 1], reporting only, never used for selection:
    safety_score   — distance-weighted mean of edge safety_scores
    shore_score    — distance-weighted fraction of shore_linable edges
    landmark_score — distance-weighted mean of edge landmark_scores

Returns (0.0, 0.0, 0.0) on an empty or zero-total-distance path (no ZeroDivision).
"""

from __future__ import annotations

from typing import List, Tuple

from shared.types import NavigationEdge


def score_path(edges: List[NavigationEdge]) -> Tuple[float, float, float]:
    """
    Distance-weighted quality scores for a path.

    Returns
    -------
    (safety_score, shore_score, landmark_score)  each in [0, 1]
    """
    if not edges:
        return (0.0, 0.0, 0.0)

    total_d = sum(e.distance for e in edges)
    if total_d == 0.0:
        return (0.0, 0.0, 0.0)

    safety   = sum(e.distance * e.safety_score          for e in edges) / total_d
    shore    = sum(e.distance * float(e.shore_linable)  for e in edges) / total_d
    landmark = sum(e.distance * e.landmark_score        for e in edges) / total_d

    return (safety, shore, landmark)
