"""
tests/conftest.py  —  Shared test fixtures (WP0 infrastructure)
---------------------------------------------------------------
Fixture A  — tiny_graph:  hand-built FloorGraph with two start→dest routes
             and a decoy node across a wall, so every cost/score/path test
             has known numbers that can be verified by hand.

Fixture B  — fake_wall_checker: callable crosses_wall(x1,y1,x2,y2)->bool
             that returns True only for segments crossing a hard-coded wall
             at x=5.0 (vertical line), letting WP5 tests run without the
             real perception_map.

Fixture C  — golden: dict of hand-computed expected values for Route X and
             Route Y under default weights λ=0.40 μ=0.35 ν=0.25.

Graph layout (all coordinates in metres):
                        WALL at x=5.0
  (0,0) START ──────── (3,0) MID-X ────────── (8,0) DEST
                  ╲
                   (3,3) MID-Y ──────────────── (8,0) DEST
  (4.9,0) DECOY   [across wall from START at (0,0)]

Route X  (START→MID-X→DEST):  shorter distance, low quality
Route Y  (START→MID-Y→DEST):  longer distance, high quality
DECOY    (4.9, 0):  close to START geometrically but wall-blocked

Node IDs
  SKE-START   junction  (0, 0)
  MID-X       junction  (3, 0)     Route X intermediate
  MID-Y       junction  (3, 3)     Route Y intermediate
  ZONE-DEST   zone_centroid (8, 0) destination
  DECOY       junction  (4.9, 0)   across wall — should never be chosen

Edge IDs and properties
  EX1  START→MID-X   dist=3.0  shore=False  safety=0.3  landmark=0.2
  EX2  MID-X→DEST    dist=5.0  shore=False  safety=0.3  landmark=0.2
  EY1  START→MID-Y   dist=4.24 shore=True   safety=0.9  landmark=0.8
  EY2  MID-Y→DEST    dist=6.40 shore=True   safety=0.9  landmark=0.8
  ED1  DECOY→MID-X   dist=1.9  shore=False  safety=0.5  landmark=0.3
"""

import math
import pytest
import sys
from pathlib import Path

# Make sure the package root is on the path regardless of how pytest is invoked
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.types import (
    FloorGraph, NavigationEdge, NavigationNode,
)

# ── Geometry helpers ──────────────────────────────────────────────────────────
def _dist(ax, ay, bx, by):
    return round(math.sqrt((bx - ax) ** 2 + (by - ay) ** 2), 4)


# ── Fixture A — tiny_graph ────────────────────────────────────────────────────
@pytest.fixture
def tiny_graph() -> FloorGraph:
    nodes = [
        NavigationNode("SKE-START", "Start Junction", (0.0, 0.0), "junction", None),
        NavigationNode("MID-X",     "Mid X",          (3.0, 0.0), "junction", None),
        NavigationNode("MID-Y",     "Mid Y",          (3.0, 3.0), "junction", None),
        NavigationNode("ZONE-DEST", "Destination",    (8.0, 0.0), "zone_centroid", "Z1"),
        NavigationNode("DECOY",     "Decoy Node",     (4.9, 0.0), "junction", None),
    ]

    # Route X — short, low quality
    ex1_dist = _dist(0, 0, 3, 0)   # 3.0
    ex2_dist = _dist(3, 0, 8, 0)   # 5.0

    # Route Y — longer, high quality
    ey1_dist = _dist(0, 0, 3, 3)   # ~4.2426
    ey2_dist = _dist(3, 3, 8, 0)   # ~5.831  (sqrt(25+9)=sqrt(34)≈5.831)

    # Decoy edge
    ed1_dist = _dist(4.9, 0, 3, 0)  # ~1.9

    edges = [
        NavigationEdge("EX1", "SKE-START", "MID-X",   ex1_dist, False, 0.3, 0.2),
        NavigationEdge("EX2", "MID-X",     "ZONE-DEST", ex2_dist, False, 0.3, 0.2),
        NavigationEdge("EY1", "SKE-START", "MID-Y",   ey1_dist, True,  0.9, 0.8),
        NavigationEdge("EY2", "MID-Y",     "ZONE-DEST", ey2_dist, True,  0.9, 0.8),
        NavigationEdge("ED1", "DECOY",     "MID-X",   ed1_dist, False, 0.5, 0.3),
    ]

    fg = FloorGraph(floor_label="L1", source_file="test_fixture")
    fg.nodes = nodes
    fg.edges = edges
    fg.rebuild_index()
    return fg


# ── Fixture B — fake_wall_checker ─────────────────────────────────────────────
@pytest.fixture
def fake_wall_checker():
    """
    Simulates a wall running vertically at x = 5.0.
    Returns True if a segment (x1,y1)→(x2,y2) straddles x=5.0.
    """
    WALL_X = 5.0

    def crosses_wall(x1: float, y1: float, x2: float, y2: float) -> bool:
        lo, hi = min(x1, x2), max(x1, x2)
        return lo < WALL_X < hi

    return crosses_wall


# ── Fixture C — golden hand-computed values ───────────────────────────────────
# Weights: λ=0.40 (safety), μ=0.35 (shore), ν=0.25 (landmark)
# Penalty form: proportional  →  combined_cost = d * (1 + λ(1-s) + μ(1-sh) + ν(1-lm))
# landmark_score already in [0,1] for this fixture (no normalisation needed)
#
# Route X  edges EX1 (d=3, s=0.3, sh=0, lm=0.2) + EX2 (d=5, s=0.3, sh=0, lm=0.2)
#   penalty per unit = λ(1-0.3)+μ(1-0)+ν(1-0.2) = 0.40*0.7+0.35*1.0+0.25*0.8
#                    = 0.28 + 0.35 + 0.20 = 0.83
#   cost EX1 = 3.0*(1+0.83) = 5.49
#   cost EX2 = 5.0*(1+0.83) = 9.15
#   total Route X combined_cost = 14.64,  total distance = 8.0
#
# Route Y  edges EY1 (d≈4.2426, s=0.9, sh=1, lm=0.8) + EY2 (d≈5.831, s=0.9, sh=1, lm=0.8)
#   penalty per unit = 0.40*(1-0.9)+0.35*(1-1)+0.25*(1-0.8) = 0.04+0+0.05 = 0.09
#   cost EY1 = 4.2426*(1+0.09) ≈ 4.6244
#   cost EY2 = 5.831*(1+0.09)  ≈ 6.3558
#   total Route Y combined_cost ≈ 10.9802, total distance ≈ 10.0736
#
# Route Y wins on combined_cost despite being physically longer.

import math as _math

_EX1_D = 3.0
_EX2_D = 5.0
_EY1_D = round(_math.sqrt(3**2 + 3**2), 4)   # 4.2426
_EY2_D = round(_math.sqrt(5**2 + 3**2), 4)   # 5.831

_LAM, _MU, _NU = 0.40, 0.35, 0.25

def _cost(d, s, sh, lm):
    return round(d * (1 + _LAM*(1-s) + _MU*(1-sh) + _NU*(1-lm)), 6)

GOLDEN = {
    # per-edge combined costs
    "EX1_cost": _cost(_EX1_D, 0.3, 0.0, 0.2),
    "EX2_cost": _cost(_EX2_D, 0.3, 0.0, 0.2),
    "EY1_cost": _cost(_EY1_D, 0.9, 1.0, 0.8),
    "EY2_cost": _cost(_EY2_D, 0.9, 1.0, 0.8),
    # route totals
    "route_X_combined_cost": _cost(_EX1_D,0.3,0.0,0.2) + _cost(_EX2_D,0.3,0.0,0.2),
    "route_Y_combined_cost": _cost(_EY1_D,0.9,1.0,0.8) + _cost(_EY2_D,0.9,1.0,0.8),
    "route_X_distance": _EX1_D + _EX2_D,
    "route_Y_distance": _EY1_D + _EY2_D,
    # distance-weighted quality scores for Route X
    "route_X_safety":   (_EX1_D*0.3 + _EX2_D*0.3) / (_EX1_D + _EX2_D),   # 0.3
    "route_X_shore":    0.0,
    "route_X_landmark": (_EX1_D*0.2 + _EX2_D*0.2) / (_EX1_D + _EX2_D),   # 0.2
    # distance-weighted quality scores for Route Y
    "route_Y_safety":   (_EY1_D*0.9 + _EY2_D*0.9) / (_EY1_D + _EY2_D),   # 0.9
    "route_Y_shore":    1.0,
    "route_Y_landmark": (_EY1_D*0.8 + _EY2_D*0.8) / (_EY1_D + _EY2_D),   # 0.8
}


@pytest.fixture
def golden():
    return GOLDEN
