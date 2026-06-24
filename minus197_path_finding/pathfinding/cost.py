"""
pathfinding/cost.py  —  WP1  (Stage 0 + Stage 2)
--------------------------------------------------
Stage 0  normalise_landmark_scores()
    Clamps every edge's raw landmark_score into [0, 1] using the 99th-percentile
    of all raw scores as the ceiling.  Mutates graph in place.

Stage 2  CostWeights / compute_edge_costs()
    Populates NavigationEdge.combined_cost using the proportional penalty form:

        combined_cost = distance × (1 + λ(1−safety) + μ(1−shore) + ν(1−landmark))

    where λ+μ+ν = 1 and each quality dimension is already in [0,1].

    A perfect edge  (safety=1, shore=1, landmark=1)  →  cost = distance  (×1)
    A worst edge    (safety=0, shore=0, landmark=0)  →  cost = 2×distance (×2)
    Combined cost is always ≥ distance > 0, so Dijkstra's non-negative-weight
    precondition is guaranteed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from shared.types import FloorGraph


# ── Stage 0 ───────────────────────────────────────────────────────────────────

def normalise_landmark_scores(
    graph: FloorGraph,
    landmark_max: Optional[float] = None,
) -> float:
    """
    Clamp landmark_score on every edge into [0, 1].

    Parameters
    ----------
    graph        : FloorGraph  — mutated in place
    landmark_max : float | None
        The value to treat as 1.0.  If None, uses the 99th percentile of all
        raw landmark_scores in the graph (robust to outliers).

    Returns
    -------
    float  — the landmark_max that was used (record this for the write-up /
              model definition; it is part of the calibration artefact).
    """
    if not graph.edges:
        return 1.0

    raw = np.array([e.landmark_score for e in graph.edges], dtype=float)

    if landmark_max is None:
        landmark_max = float(np.percentile(raw, 99))

    # Avoid division by zero when all edges share the same score
    if landmark_max <= 0.0:
        landmark_max = 1.0

    for edge in graph.edges:
        edge.landmark_score = float(np.clip(edge.landmark_score / landmark_max, 0.0, 1.0))

    return landmark_max


# ── Stage 2 ───────────────────────────────────────────────────────────────────

@dataclass
class CostWeights:
    """
    λ (lam)  — weight on safety penalty   (corridor safety)
    μ (mu)   — weight on shore penalty    (wall-linable fraction)
    ν (nu)   — weight on landmark penalty (landmark density)

    Constraint: lam + mu + nu == 1.0  (enforced by validate()).
    Default values are the calibrated starting point from the build plan.
    """
    lam: float = 0.40   # λ
    mu:  float = 0.35   # μ
    nu:  float = 0.25   # ν

    def validate(self) -> None:
        total = self.lam + self.mu + self.nu
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"CostWeights must sum to 1.0, got {total:.6f} "
                f"(lam={self.lam}, mu={self.mu}, nu={self.nu})"
            )


def compute_edge_costs(graph: FloorGraph, weights: CostWeights) -> None:
    """
    Populate NavigationEdge.combined_cost for every edge in graph.

    Proportional penalty form:
        combined_cost = d × (1 + λ(1−s) + μ(1−sh) + ν(1−lm))

    Preconditions
    -------------
    • normalise_landmark_scores() should have been called first so that
      landmark_score is already in [0, 1].
    • weights.validate() should pass (caller's responsibility; engine calls
      both in its constructor).

    Mutates graph in place; returns nothing.
    """
    lam, mu, nu = weights.lam, weights.mu, weights.nu
    shore_val = {True: 1.0, False: 0.0}

    for edge in graph.edges:
        sh = shore_val[edge.shore_linable]
        penalty = lam * (1.0 - edge.safety_score) \
                + mu  * (1.0 - sh) \
                + nu  * (1.0 - edge.landmark_score)
        edge.combined_cost = edge.distance * (1.0 + penalty)
