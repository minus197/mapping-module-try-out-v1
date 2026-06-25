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
from typing import Any, Callable, Dict, List, Optional, Tuple

from shared.types import (
    FloorGraph, NavigationEdge, NavigationNode, PathResult, PathStep, Point2D,
)
from pathfinding.cost import CostWeights, compute_edge_costs, normalise_landmark_scores
from pathfinding.instructions import build_steps
from pathfinding.node_resolver import StartNodeResolver
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
    """

    def __init__(
        self,
        graph:        FloorGraph,
        wall_checker: WallChecker,
        weights:      CostWeights = None,
        landmark_max: Optional[float] = None,
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
        if start_node is None:
            return _empty_result()

        user_xy: Point2D = start_node.position

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

def to_feedback_json(result: PathResult) -> str:
    """
    Serialise a PathResult into the feedback-module action format.

    Each PathStep becomes one action object:
      - "continue"  — straight or start-walking segments
      - "turn"      — any turn (left/right/bear/around), with direction
      - "stop"      — final step only, with landmark name

    Distance is rounded to the nearest metre and omitted on "stop".
    """
    if not result.found or not result.steps:
        return json.dumps([])

    actions: List[Dict[str, Any]] = []
    steps = result.steps

    for i, step in enumerate(steps):
        is_last = (i == len(steps) - 1)
        phrase = _first_sentence(step.instruction)   # e.g. "Turn left"
        dist_m = round(step.distance)

        if is_last:
            landmark = (
                step.to_node.tags.get("admin_label", "").strip()
                or step.to_node.label
            )
            action: Dict[str, Any] = {"action": "stop", "landmark": landmark}
        elif _is_turn(phrase):
            direction = _turn_direction(phrase)
            action = {"action": "turn", "direction": direction, "distance": dist_m}
        else:
            action = {"action": "continue", "distance": dist_m}

        actions.append(action)

    return json.dumps(actions, indent=2)


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
