"""
pathfinding/instructions.py  —  WP3  (Stage 6)
-----------------------------------------------
Converts a sequence of NavigationNodes + edges into human-readable PathSteps
suitable for text-to-speech output for a visually impaired user.

Public API
----------
bearing(p1, p2)                     → float   degrees clockwise from north
turn_phrase(delta)                  → str     e.g. "Turn left"
landmark_hint(node)                 → str     e.g. "You will reach the lift."
build_steps(user_xy, start_node,
            path_nodes, edge_lookup) → list[PathStep]

Turn-phrase table (delta = new_bearing − prev_bearing, wrapped to (−180, +180])
    |delta| ≤  20   →  "Continue straight"
    |delta| ≤ 100   →  "Bear right" / "Bear left"
    |delta| ≤ 170   →  "Turn right" / "Turn left"
    otherwise       →  "Turn around"

First segment (true user position → start node) is emitted only when the user
is more than 0.5 m from the start node.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Protocol, Tuple

from shared.types import NavigationEdge, NavigationNode, PathStep, Point2D

# ── Constants ─────────────────────────────────────────────────────────────────
_FIRST_STEP_MIN_DIST = 0.5   # metres — skip first step if user is on the node
_LANDMARK_TYPES = {"door", "stair", "lift", "elevator", "escalator", "landmark"}


# ── Edge-lookup protocol (declared here; concrete impl in engine.py WP6) ──────
class EdgeLookup(Protocol):
    def __call__(self, a_id: str, b_id: str) -> Optional[NavigationEdge]: ...
    def node(self, node_id: str) -> Optional[NavigationNode]: ...


# ── Public functions ───────────────────────────────────────────────────────────

def bearing(p1: Point2D, p2: Point2D) -> float:
    """
    Bearing from p1 to p2, degrees clockwise from north (0–360).
    North is the positive-Y direction.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dx, dy)) % 360


def turn_phrase(delta: float) -> str:
    """
    Convert a bearing change (degrees, any range) to an instruction phrase.
    delta is normalised to (−180, +180] before comparison.
    """
    # normalise to (-180, +180]
    d = ((delta + 180) % 360) - 180

    abs_d = abs(d)
    if abs_d <= 20:
        return "Continue straight"
    if abs_d <= 100:
        return "Bear right" if d > 0 else "Bear left"
    if abs_d <= 170:
        return "Turn right" if d > 0 else "Turn left"
    return "Turn around"


def landmark_hint(node: NavigationNode) -> str:
    """
    Return a spoken hint if the node is a notable landmark.
    Fires for door/stair/lift/elevator/escalator/landmark types,
    or any node with a non-empty admin_label tag.
    """
    admin = node.tags.get("admin_label", "").strip()
    if node.node_type in _LANDMARK_TYPES or admin:
        name = admin or node.label
        return f" You will reach {name}."
    return ""


def build_steps(
    user_xy:    Point2D,
    start_node: NavigationNode,
    path_nodes: List[NavigationNode],
    edge_lookup: EdgeLookup,
) -> List[PathStep]:
    """
    Build PathStep list for a resolved path.

    Parameters
    ----------
    user_xy     : true user position (from GPS / BLE / IMU)
    start_node  : first graph node on the path
    path_nodes  : ordered list of NavigationNodes (start → destination)
    edge_lookup : callable(a_id, b_id) → NavigationEdge | None
                  must also support .node(id) → NavigationNode | None

    Returns
    -------
    List[PathStep], possibly starting with a "walk to start node" step.
    """
    steps: List[PathStep] = []
    prev_bearing: Optional[float] = None

    # ── Optional first step: true position → start node ──────────────────────
    dist_to_start = math.sqrt(
        (start_node.position[0] - user_xy[0]) ** 2 +
        (start_node.position[1] - user_xy[1]) ** 2
    )
    if dist_to_start > _FIRST_STEP_MIN_DIST:
        b = bearing(user_xy, start_node.position)
        hint = landmark_hint(start_node)
        dist_str = _format_dist(dist_to_start)
        instruction = f"Start walking. Walk {dist_str}.{hint}"
        # Synthesise a minimal edge for this virtual step
        virtual_edge = NavigationEdge(
            edge_id="VIRTUAL-START",
            source_id="USER",
            target_id=start_node.node_id,
            distance=dist_to_start,
            shore_linable=False,
            safety_score=1.0,
            landmark_score=0.0,
        )
        # Create a virtual "user" node
        user_node = NavigationNode(
            node_id="USER", label="Your position",
            position=user_xy, node_type="junction", zone_id=None,
        )
        steps.append(PathStep(
            from_node=user_node,
            to_node=start_node,
            edge=virtual_edge,
            bearing=b,
            distance=dist_to_start,
            instruction=instruction,
        ))
        prev_bearing = b

    # ── Main path steps ───────────────────────────────────────────────────────
    for i in range(len(path_nodes) - 1):
        src = path_nodes[i]
        tgt = path_nodes[i + 1]
        edge = edge_lookup(src.node_id, tgt.node_id)
        if edge is None:
            continue

        b = bearing(src.position, tgt.position)
        delta = (b - prev_bearing) if prev_bearing is not None else 0.0
        turn = turn_phrase(delta) if prev_bearing is not None else "Start walking"
        hint = landmark_hint(tgt)
        dist_str = _format_dist(edge.distance)
        instruction = f"{turn}. Walk {dist_str}.{hint}"

        steps.append(PathStep(
            from_node=src,
            to_node=tgt,
            edge=edge,
            bearing=b,
            distance=edge.distance,
            instruction=instruction.strip(),
        ))
        prev_bearing = b

    return steps


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_dist(d: float) -> str:
    if d >= 1.0:
        return f"{d:.0f} m"
    return f"{d * 100:.0f} cm"
