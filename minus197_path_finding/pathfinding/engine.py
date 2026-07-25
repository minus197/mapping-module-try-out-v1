"""
pathfinding/engine.py  —  WP6  (Section 6 integrator)
-------------------------------------------------------
PathfindingEngine wires WP0–WP5 into a single public call:

    engine = PathfindingEngine(graph, wall_checker)
    result = engine.find_path(start_node_id, destination_node_id)

Construction (runs once)
    Stage 0  normalise_landmark_scores()
    Stage 2  compute_edge_costs()
    Stage 3  build_nx_graph()
    Stage 1  StartNodeResolver (KD-tree)

find_path(start_node_id, dest_id)
    Stage 1  look up start node by id
    Stage 3  Dijkstra to confirm dest reachable; bail early if not
    Stage 4  k_best_paths() — winner + alternatives
    Stage 5  score_path()   — quality scores for each path
    Stage 6  build_steps()  — turn-by-turn PathStep list
             _assemble()    — pack into PathResult
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from shared.types import (
    FloorGraph, NavigationEdge, NavigationNode, PathResult, PathStep, Point2D,
)
from pathfinding.cost import CostWeights, compute_edge_costs, normalise_landmark_scores
from pathfinding.instructions import build_steps
from pathfinding.node_resolver import StartNodeResolver
from pathfinding.path_node_adjuster import PathNode
from pathfinding.scorer import score_path
from pathfinding.search import build_nx_graph, k_best_paths, optimal_path

WallChecker = Callable[[float, float, float, float], bool]

_MAX_ALTERNATIVES = 3
_K_PATHS = 4        # k passed to Yen's; winner + up to 3 alternatives


class PathfindingEngine:
    """
    End-to-end pathfinding for visually impaired indoor navigation.

    Parameters
    ----------
    graph        : FloorGraph    — from map_extraction (nodes/edges loaded)
    wall_checker : WallChecker   — crosses_wall(x1,y1,x2,y2)->bool
    weights      : CostWeights   — λ/μ/ν penalty weights (default calibrated)
    landmark_max : float | None  — 99th-pct ceiling for landmark normalisation;
                                   None = auto-detect from graph
    path_nodes   : list[PathNode] | None — optional path-node layer; when given,
                                   find_path() can accept a path-node id as the
                                   current (start) node. Adds no graph nodes/edges.
    """

    def __init__(
        self,
        graph:        FloorGraph,
        wall_checker: WallChecker,
        weights:      CostWeights = None,
        landmark_max: Optional[float] = None,
        path_nodes:   Optional[List[PathNode]] = None,
    ) -> None:
        if weights is None:
            weights = CostWeights()
        weights.validate()

        self.graph   = graph
        self.weights = weights

        # Stage 0 + Stage 2 — run once at construction
        self._landmark_max = normalise_landmark_scores(graph, landmark_max)
        compute_edge_costs(graph, weights)

        # Stage 3 — build networkx graph once
        self._G = build_nx_graph(graph)

        # Stage 1 — KD-tree resolver
        self._resolver = StartNodeResolver(graph, wall_checker)

        # Edge lookup used by build_steps (WP3)
        self._edge_lookup = _EdgeLookup(graph)

        # Read-only lookup so find_path can resolve a path-node id passed as the
        # current node. Path nodes are NOT added to any graph — this adds no
        # nodes/edges and leaves all non-path-node routes byte-for-byte identical.
        self._path_node_index: Dict[str, PathNode] = {
            pn.node_id: pn for pn in (path_nodes or [])
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def find_path(
        self,
        start_node_id:       str,
        destination_node_id: str,
    ) -> PathResult:
        """
        Find the optimal path from a start node to a destination node.

        Parameters
        ----------
        start_node_id        : node_id string of the current location
        destination_node_id  : node_id string of the destination

        Returns
        -------
        PathResult — found=False with empty lists if either node is unknown/unreachable.
        """
        dest_node = self.graph.node(destination_node_id)
        if dest_node is None:
            return _empty_result()

        start_node = self.graph.node(start_node_id)
        if start_node is not None:
            # Non-path (graph) node — unchanged behaviour.
            user_xy: Point2D = start_node.position
        else:
            # Not a graph node — try to treat it as a path node.
            pn = self._path_node_index.get(start_node_id)
            if pn is None:
                return _empty_result()               # genuinely unknown id
            # Resolve the path node's world position to a real graph start node,
            # exactly as we resolve a raw user position. user_xy stays the path
            # node's true position so build_steps' first step starts on it.
            start_node, user_xy = self._resolver.resolve(pn.position[0], pn.position[1])

        # Quick reachability check (also handles start == dest)
        if start_node.node_id == dest_node.node_id:
            return self._assemble(
                user_xy=user_xy,
                start_node=start_node,
                dest_node=dest_node,
                winner=[start_node.node_id],
                alternatives=[],
            )

        if optimal_path(self._G, start_node.node_id, dest_node.node_id) is None:
            return _empty_result()

        # Stage 4 — k best paths
        paths = k_best_paths(self._G, start_node.node_id, dest_node.node_id, k=_K_PATHS)
        if not paths:
            return _empty_result()

        winner    = paths[0]
        alt_paths = paths[1:_MAX_ALTERNATIVES + 1]

        # Stage 5 + 6 + assemble winner
        result = self._assemble(user_xy, start_node, dest_node, winner, [])

        # Assemble alternatives (no sub-alternatives)
        alt_results: List[PathResult] = []
        for alt in alt_paths:
            alt_results.append(
                self._assemble(user_xy, start_node, dest_node, alt, [])
            )
        result.alternatives = alt_results

        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _assemble(
        self,
        user_xy:    Point2D,
        start_node: NavigationNode,
        dest_node:  NavigationNode,
        winner:     List[str],
        alternatives: List[PathResult],
    ) -> PathResult:
        """Convert a node-id path into a fully populated PathResult."""
        edges  = _path_edges(winner, self._edge_lookup)
        safety, shore, landmark = score_path(edges)

        path_nodes = [
            self.graph.node(nid) for nid in winner
            if self.graph.node(nid) is not None
        ]

        steps = build_steps(user_xy, start_node, path_nodes, self._edge_lookup)

        total_cost     = sum(e.combined_cost for e in edges)
        total_distance = sum(e.distance      for e in edges)

        return PathResult(
            found            = True,
            start_node       = start_node,
            destination_node = dest_node,
            steps            = steps,
            alternatives     = alternatives,
            total_cost       = round(total_cost,     4),
            total_distance   = round(total_distance, 4),
            safety_score     = round(safety,   3),
            shore_score      = round(shore,    3),
            landmark_score   = round(landmark, 3),
        )


# ── Edge lookup ───────────────────────────────────────────────────────────────

class _EdgeLookup:
    """
    Satisfies the EdgeLookup protocol declared in instructions.py.
    Supports both directed (a→b) and undirected (b→a) lookups.
    """

    def __init__(self, graph: FloorGraph) -> None:
        self._fwd: Dict[Tuple[str, str], NavigationEdge] = {}
        for e in graph.edges:
            self._fwd[(e.source_id, e.target_id)] = e
            self._fwd[(e.target_id, e.source_id)] = e   # undirected
        self._node_index = graph._node_index

    def __call__(
        self, a_id: str, b_id: str
    ) -> Optional[NavigationEdge]:
        return self._fwd.get((a_id, b_id))

    def node(self, node_id: str) -> Optional[NavigationNode]:
        return self._node_index.get(node_id)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _path_edges(
    node_ids: List[str],
    lookup: _EdgeLookup,
) -> List[NavigationEdge]:
    edges = []
    for i in range(len(node_ids) - 1):
        e = lookup(node_ids[i], node_ids[i + 1])
        if e is not None:
            edges.append(e)
    return edges


def _empty_result() -> PathResult:
    return PathResult(found=False)


# ── Feedback-module serialiser ────────────────────────────────────────────────

def to_feedback_json(result: PathResult,
                     terminal: bool = True,
                     as_list: bool = False):
    """
    Serialise a PathResult into the feedback-module action format.

    Every PathStep becomes exactly one movement action, anchored at the node
    the user is standing on (`from_node`). When `terminal` is set, a separate
    "stop" action is *appended* at the destination rather than consuming the
    final movement row — so a route ending `… → PATH-0126 → DOOR → CENTROID`
    emits the door approach as its own step instead of hiding it.

    Movement actions:
      - "continue"  — straight or start-walking segments
      - "turn"      — any turn, carrying both `direction` and `band`

    Each action carries the full PathStep contract (shared/types.PathStep):
    `bearing`, float `distance`, `instruction`, and the edge attributes
    needed to compute the route-quality metrics (`edge_id`, `shore_linable`,
    `safety_score`, `landmark_score`). Distances are emitted as floats;
    round at the speech layer, and prefer rounding a cumulative figure over
    rounding each hop independently.

    terminal : when False, no "stop" is appended — used for a non-final leg
               of a multi-floor route that ends at an elevator.
    as_list  : when True, return the Python list of action dicts instead of a
               JSON string (so callers can concatenate legs).
    """
    if not result.found or not result.steps:
        return [] if as_list else json.dumps([])

    actions: List[Dict[str, Any]] = []
    steps = result.steps

    for step in steps:
        phrase = _first_sentence(step.instruction)   # e.g. "Turn left"

        if _is_turn(phrase):
            action: Dict[str, Any] = {
                "action":    "turn",
                "direction": _turn_direction(phrase),
                "band":      _turn_band(phrase),
                "distance":  round(step.distance, 2),
            }
        else:
            action = {"action": "continue", "distance": round(step.distance, 2)}

        # Movement actions are performed at the node the user stands on.
        action["node_id"] = step.from_node.node_id
        # World coordinates in metres — see NavigationNode.position.
        action["position"] = [round(step.from_node.position[0], 4),
                              round(step.from_node.position[1], 4)]
        action["to_node_id"] = step.to_node.node_id
        action["bearing"]    = round(step.bearing, 2)
        action["instruction"] = step.instruction

        edge = step.edge
        if edge is not None:
            action["edge_id"]        = edge.edge_id
            action["shore_linable"]  = bool(edge.shore_linable)
            action["safety_score"]   = round(edge.safety_score, 4)
            action["landmark_score"] = round(edge.landmark_score, 4)

        actions.append(action)

    if terminal:
        prev = steps[-2] if len(steps) >= 2 else None
        actions.append(_stop_action(steps[-1], prev))

    return actions if as_list else json.dumps(actions, indent=2)


def _stop_action(final_step: PathStep,
                 prev_step: Optional[PathStep]) -> Dict[str, Any]:
    """
    Build the terminal "stop", anchored at the destination node reached.

    Emitted as an extra action so the final movement step keeps its own row —
    on a route ending `… → PATH-0126 → DOOR → CENTROID` the door approach is
    a movement step of its own and is no longer consumed as this anchor.

    `side` names which side the destination lies on relative to the heading
    the user was already travelling on.
    """
    dest = final_step.to_node
    landmark = dest.tags.get("admin_label", "").strip() or dest.label
    side = _approach_side(final_step, prev_step)
    dist = round(final_step.distance, 2)

    if side in ("left", "right"):
        instruction = f"The entrance to {landmark} is on your {side} in {dist:.0f} metres."
    else:
        instruction = f"You have arrived at {landmark}."

    return {
        "action":      "stop",
        "landmark":    landmark,
        "side":        side,
        "node_id":     dest.node_id,
        "position":    [round(dest.position[0], 4), round(dest.position[1], 4)],
        "instruction": instruction,
    }


def _approach_side(final_step: PathStep,
                   prev_step: Optional[PathStep]) -> str:
    """
    Which side of the user the destination sits on: left | right | ahead.

    Sign of the cross product between the heading the user is already
    travelling (prev_step) and the vector to the destination. Positive cross
    (x1*y2 − y1*x2) means the destination lies counter-clockwise of the
    approach heading — which, with north as +Y and bearings measured
    clockwise, is the user's left.

    Returns "ahead" when the two are collinear within the dead band, or when
    there is no preceding step to establish a heading.
    """
    if prev_step is None:
        return "ahead"

    # Heading the user is already travelling on.
    hx = prev_step.to_node.position[0] - prev_step.from_node.position[0]
    hy = prev_step.to_node.position[1] - prev_step.from_node.position[1]

    # Vector from the user's current position to the destination.
    dx = final_step.to_node.position[0] - final_step.from_node.position[0]
    dy = final_step.to_node.position[1] - final_step.from_node.position[1]

    if (hx == 0.0 and hy == 0.0) or (dx == 0.0 and dy == 0.0):
        return "ahead"

    cross = hx * dy - hy * dx
    scale = math.hypot(hx, hy) * math.hypot(dx, dy)
    if scale == 0.0 or abs(cross) / scale < _SIDE_DEAD_BAND:
        return "ahead"
    return "left" if cross > 0 else "right"


# sin of the half-angle below which we call the destination "straight ahead"
_SIDE_DEAD_BAND = 0.087   # ≈ 5°


def _first_sentence(instruction: str) -> str:
    """Extract the first sentence (up to the first period)."""
    return instruction.split(".")[0].strip()


_TURN_KEYWORDS = {"turn", "bear", "around"}

def _is_turn(phrase: str) -> bool:
    return any(kw in phrase.lower() for kw in _TURN_KEYWORDS)


def _turn_direction(phrase: str) -> str:
    lower = phrase.lower()
    if "left" in lower:
        return "left"
    if "right" in lower:
        return "right"
    return "around"


def _turn_band(phrase: str) -> str:
    """
    Preserve the magnitude band that turn_phrase() resolved, which
    `direction` alone discards: a 29° bear and a 141° near-reversal are
    both "left" but are not the same instruction.

    Bands mirror instructions.turn_phrase(): bear | turn | around.
    """
    lower = phrase.lower()
    if "around" in lower:
        return "around"
    if "bear" in lower:
        return "bear"
    return "turn"
