"""
pathfinding/path_node_adjuster.py  —  post-processing stage (not part of WP0-6)
---------------------------------------------------------------------------------
Rewrites an already-computed PathResult so that the walking line hugs the
floor's path-node layer (cane-trailing waypoints, see map_extraction/
path_nodes.py) wherever possible, and only leaves that layer to cross open
corridor space when a flat "leave the wall" cost is worth paying.

The original graph route (junctions/doors/zone_centroids from
PathfindingEngine.find_path()) is used only to decide WHICH corridors/zones
the walk passes through and to seed the start/destination/doorway anchors.
The actual walking geometry is re-derived by a single shortest-path search
over a combined graph of path nodes + those anchors:

  * same-face hops (consecutive/corner-linked path nodes on the same wall
    side) cost their real Euclidean distance — cheap, as many as needed;
  * crossing hops (leaving one wall face for another, or for a mandatory
    door/start/goal anchor) cost a flat BETA — a real, fixed penalty per
    crossing event, not proportional to distance;
  * the original route's own edges are always seeded as a fallback, so a
    leg with no viable path-node alternative still produces a route.

This module does not change PathfindingEngine, the graph, or the cost/safety
scoring model — combined_cost / safety_score / shore_score / landmark_score on
the returned PathResult continue to describe the graph-level route. Only
`steps` is replaced with the expanded, path-node-hugging turn-by-turn list.

Algorithm
---------
1.  Derive a `face_id` for every path node: `"{wall_id}#{sign}"`, where
    `sign` is the side of the wall (+1/-1) recovered from the node's own
    position relative to the wall geometry — a free-standing wall with
    corridor on both sides shares one wall_id but must not chain its two
    sides together as if adjacent.
2.  Build a proximity graph over path nodes: two nodes on the SAME face are
    linked to their immediate same-face neighbour (strict axis ordering, as
    before); corner bridges link mutual-nearest-neighbour endpoints of
    DIFFERENT faces. Both kinds of link cost real Euclidean distance.
3.  Add crossing edges between path nodes on DIFFERENT faces: candidate when
    the connecting segment does not cross any wall, is within
    MAX_CROSSING_M, and is within PERP_ANGLE_MAX_DEG of the origin face's
    normal (a genuine crossing, not a diagonal drift down the corridor).
    These cost the flat BETA, not distance.
4.  Connect mandatory anchors (start node, destination node, every door/
    feature node on the original route) to their nearest path node within
    SNAP_RADIUS at real distance — but ONLY when one exists that close; no
    widening to "nearest on the whole floor", so a genuinely isolated
    crossing leg still prefers the original direct edge.
5.  Seed the original route's own edges into the graph (real distance) so a
    leg with no viable path-node alternative still has a route.
6.  Restrict the search to path nodes within ROUTE_BUFFER_M of the original
    route's polyline (optionally widened to any path node whose nearest wall
    borders a circulation zone the route touches, when `zones` is given).
7.  Run one Dijkstra from the true start to the true destination over this
    mixed graph. Synthetic NavigationEdges are created for the winning
    sequence's hops (mirroring the existing VIRTUAL-START edge pattern in
    instructions.py), and build_steps() is called unchanged on the expanded
    node sequence to regenerate turn-by-turn instructions with one step per
    hop.

Public API
----------
load_path_nodes(path)                                    -> List[PathNode]
load_corridor_walls(sfm_path)                             -> List[Wall]
load_zones(sfm_path)                                       -> List[Zone]
adjust_with_path_nodes(result, path_nodes, walls,
                       zones=None, user_xy=None)          -> PathResult
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import KDTree

from shared.types import (
    FloorGraph, NavigationEdge, NavigationNode, PathResult, PathStep, Point2D,
)
from pathfinding.instructions import build_steps

# ── Tunables ──────────────────────────────────────────────────────────────────
# Mirrors map_extraction/path_nodes.py SPACING_MAX (6.0 m); 1.5x tolerates one
# missing node in a run without bridging an unrelated stretch of corridor.
MAX_CHAIN_GAP     = 9.0    # m — max straight-line gap between chained path nodes
SNAP_RADIUS       = 6.0    # m — max distance from an anchor to its path-node link
# How far (relative to the two nodes' own gap_m) the midpoint of a straight
# segment between them may drift from the nearest wall before it is judged to
# be cutting across open corridor space rather than hugging a wall.
DRIFT_TOLERANCE_FACTOR = 1.75

MAX_CROSSING_M    = 15.0   # m — max corridor width + margin for a crossing edge
BETA              = 8.0    # flat cost charged per corridor crossing
PERP_ANGLE_MAX_DEG = 35.0  # max angle from the origin face's normal for a crossing
ROUTE_BUFFER_M    = 12.0   # m — eligibility buffer around the original route's polyline
# Multiplier applied to an original-route edge's real distance when it is
# seeded as a fallback candidate. The point of this stage is to PREFER
# path-node hugging even when it is geometrically a little longer than the
# direct graph edge — a fallback edge must only win the search when there is
# truly no viable path-node alternative (isolated crossing, coverage gap),
# not merely because hugging adds a few metres over a straight line. This
# keeps it usable (search still terminates, defensive fallback for gaps)
# without letting it out-compete a genuine hugging route on cost alone.
ORIGINAL_EDGE_PENALTY_FACTOR = 5.0

CIRCULATION_CATEGORIES = {"corridor", "entrance", "exit"}


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class PathNode:
    """One cane-trailing waypoint, as loaded from a *_path_nodes.json file."""
    node_id:  str
    position: Point2D
    wall_id:  str
    gap_m:    float

    def to_navigation_node(self) -> NavigationNode:
        """Adapt to a NavigationNode so it can flow through build_steps() and
        the visualizer unchanged (mirrors the virtual USER-node pattern in
        pathfinding/instructions.py)."""
        return NavigationNode(
            node_id=self.node_id,
            label=f"Path node ({self.position[0]:.1f},{self.position[1]:.1f})",
            position=self.position,
            node_type="path_node",
            zone_id=None,
            tags={"wall_id": self.wall_id},
        )


@dataclass
class Wall:
    """Minimal wall geometry needed for face derivation and crossing checks."""
    wall_id: str
    start:   Point2D
    end:     Point2D


@dataclass
class Zone:
    """Minimal zone geometry needed for corridor-sequence eligibility scoping."""
    zone_id:          str
    category:         str
    boundary_polygon: List[Point2D]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_path_nodes(path: str | Path) -> List[PathNode]:
    """Parse a *_path_nodes.json file (see map_extraction/path_nodes.py)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        PathNode(
            node_id=n["node_id"],
            position=(float(n["position"][0]), float(n["position"][1])),
            wall_id=n["wall_id"],
            gap_m=float(n.get("gap_m", 2.0)),
        )
        for n in data.get("path_nodes", [])
    ]


def load_corridor_walls(sfm_path: str | Path) -> List[Wall]:
    """Parse the `walls` array out of a *_sfm.json file."""
    with open(sfm_path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        Wall(
            wall_id=w["wall_id"],
            start=(float(w["start"][0]), float(w["start"][1])),
            end=(float(w["end"][0]), float(w["end"][1])),
        )
        for w in data.get("walls", [])
    ]


def load_zones(sfm_path: str | Path) -> List[Zone]:
    """Parse the `zones` array out of a *_sfm.json file (same file
    load_corridor_walls already reads — zones is a sibling array). Optional
    input to adjust_with_path_nodes(); omitting it just falls back to
    route-polyline-buffer-only eligibility scoping."""
    with open(sfm_path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        Zone(
            zone_id=z["zone_id"],
            category=z.get("category", "unknown"),
            boundary_polygon=[
                (float(pt[0]), float(pt[1])) for pt in z.get("boundary_polygon", [])
            ],
        )
        for z in data.get("zones", [])
    ]


# ── Face derivation ───────────────────────────────────────────────────────────

def _face_id(position: Point2D, wall: Optional[Wall]) -> str:
    """
    Stable face key for a path node: "{wall_id}#{sign}", where sign (+1/-1)
    is the side of the wall the node sits on, recovered from geometry (the
    node's own PathNode data never stores which side it was placed on — see
    map_extraction/path_nodes.py _place_along_wall/_corridor_sides). This is
    what stops a free-standing wall with corridor on both sides (one
    wall_id, nodes on both faces) from being treated as a single chainable
    face.
    """
    if wall is None:
        return "UNKNOWN#0"
    s = np.array(wall.start, dtype=float)
    e = np.array(wall.end, dtype=float)
    axis = e - s
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return f"{wall.wall_id}#0"
    axis = axis / n
    normal = np.array([-axis[1], axis[0]])
    sign = 1 if np.dot(np.array(position, dtype=float) - s, normal) >= 0 else -1
    return f"{wall.wall_id}#{sign}"


# ── Global path-node proximity graph (same-face runs + corner bridges) ───────

class _GlobalPathGraph:
    """
    Adjacency structure over path nodes: same-face runs are strictly ordered
    along their axis and linked only to their immediate predecessor/
    successor; corners (where one face's run ends and another begins) are
    bridged via mutual-nearest-neighbour endpoints of DIFFERENT faces. Both
    kinds of edge cost real Euclidean distance and are tagged kind="path".
    """

    def __init__(self, nodes: List[PathNode], walls: List[Wall]) -> None:
        self.nodes = nodes
        self._by_id = {n.node_id: n for n in nodes}
        self._wall_lines = [
            (np.array(w.start, dtype=float), np.array(w.end, dtype=float))
            for w in walls
        ]
        self._wall_by_id = {w.wall_id: w for w in walls}

        self.face_of: Dict[str, str] = {
            n.node_id: _face_id(n.position, self._wall_by_id.get(n.wall_id))
            for n in nodes
        }

        self._positions = np.array([n.position for n in nodes]) if nodes else np.empty((0, 2))
        self._tree = KDTree(self._positions) if len(nodes) else None

        # adjacency: node_id -> list[(neighbour_id, distance, kind)]
        self.adjacency: Dict[str, List[Tuple[str, float, str]]] = {n.node_id: [] for n in nodes}
        self._build_edges()

    def _build_edges(self) -> None:
        if not self.nodes:
            return

        by_face: Dict[str, List[PathNode]] = {}
        for n in self.nodes:
            by_face.setdefault(self.face_of[n.node_id], []).append(n)

        endpoints: List[PathNode] = []
        for face_id, run in by_face.items():
            axis = self._face_axis(face_id, run)
            ordered = sorted(run, key=lambda n: np.dot(n.position, axis))
            for a, b in zip(ordered, ordered[1:]):
                d = _euclidean(a.position, b.position)
                if d < 1e-9 or d > MAX_CHAIN_GAP:
                    continue
                if self._is_wall_hugging(a, b):
                    self._link(a.node_id, b.node_id, d, "path")
            if ordered:
                endpoints.append(ordered[0])
                if len(ordered) > 1:
                    endpoints.append(ordered[-1])

        def _nearest_cross_face(n: PathNode) -> Optional[Tuple[float, PathNode]]:
            """Nearest endpoint on a DIFFERENT wall (a genuine corner), not
            merely a different face of the SAME wall — two faces of one
            free-standing wall are opposite sides of open corridor space and
            must be linked (if at all) by a crossing edge, not a corner
            bridge."""
            best: Optional[Tuple[float, PathNode]] = None
            for other in endpoints:
                if other.node_id == n.node_id or other.wall_id == n.wall_id:
                    continue
                d = _euclidean(n.position, other.position)
                if best is None or d < best[0]:
                    best = (d, other)
            return best

        for node in endpoints:
            nearest = _nearest_cross_face(node)
            if nearest is None:
                continue
            d, other = nearest
            if d < 1e-9 or d > MAX_CHAIN_GAP:
                continue
            mutual = _nearest_cross_face(other)
            if mutual is None or mutual[1].node_id != node.node_id:
                continue
            if self._is_wall_hugging(node, other):
                self._link(node.node_id, other.node_id, d, "path")

    def _link(self, a_id: str, b_id: str, d: float, kind: str) -> None:
        if any(nid == b_id for nid, _, _ in self.adjacency[a_id]):
            return
        self.adjacency[a_id].append((b_id, d, kind))
        self.adjacency[b_id].append((a_id, d, kind))

    def _face_axis(self, face_id: str, run: List[PathNode]) -> np.ndarray:
        """Unit direction to sort a face's path nodes along. Falls back to
        the run's own principal direction when wall geometry is
        unavailable."""
        wall_id = face_id.rsplit("#", 1)[0]
        wall = self._wall_by_id.get(wall_id)
        if wall is not None:
            v = np.array(wall.end, dtype=float) - np.array(wall.start, dtype=float)
            n = np.linalg.norm(v)
            if n > 1e-9:
                return v / n
        if len(run) >= 2:
            pts = np.array([n.position for n in run])
            v = pts[-1] - pts[0]
            n = np.linalg.norm(v)
            if n > 1e-9:
                return v / n
        return np.array([1.0, 0.0])

    def _is_wall_hugging(self, a: PathNode, b: PathNode) -> bool:
        """True if the straight segment a->b never drifts far from the
        nearest wall relative to the nodes' own gap_m — i.e. it keeps
        shore-lining rather than cutting across open corridor width."""
        if not self._wall_lines:
            return True
        tol = DRIFT_TOLERANCE_FACTOR * max(a.gap_m, b.gap_m, 1e-6)
        mx = (a.position[0] + b.position[0]) / 2.0
        my = (a.position[1] + b.position[1]) / 2.0
        return self._nearest_wall_dist((mx, my)) <= tol

    def _nearest_wall_dist(self, p: Point2D) -> float:
        pt = np.array(p, dtype=float)
        best = math.inf
        for s, e in self._wall_lines:
            d = _point_segment_dist(pt, s, e)
            if d < best:
                best = d
        return best

    def face_normal_at(self, node_id: str) -> np.ndarray:
        """Outward face normal (pointing away from the wall, toward the
        corridor side the node occupies) at a path node's own position."""
        n = self._by_id[node_id]
        wall = self._wall_by_id.get(n.wall_id)
        if wall is None:
            return np.array([1.0, 0.0])
        s = np.array(wall.start, dtype=float)
        e = np.array(wall.end, dtype=float)
        axis = e - s
        an = np.linalg.norm(axis)
        if an < 1e-9:
            return np.array([1.0, 0.0])
        axis = axis / an
        normal = np.array([-axis[1], axis[0]])
        sign = 1.0 if np.dot(np.array(n.position, dtype=float) - s, normal) >= 0 else -1.0
        return normal * sign

    def crosses_any_wall(self, p1: Point2D, p2: Point2D) -> bool:
        for s, e in self._wall_lines:
            if _segments_intersect(np.array(p1), np.array(p2), s, e):
                return True
        return False

    def nearest(self, pos: Point2D, radius: float) -> Optional[PathNode]:
        """Nearest path node to `pos` within `radius`, or None."""
        if self._tree is None:
            return None
        d, idx = self._tree.query(pos)
        if d > radius:
            return None
        return self.nodes[int(idx)]


# ── Crossing edges between different faces ───────────────────────────────────

def _build_crossing_edges(pn_graph: _GlobalPathGraph) -> List[Tuple[str, str, float, str]]:
    """
    Candidate crossing edges between path nodes on DIFFERENT faces: the
    connecting segment must not cross any wall, must be within
    MAX_CROSSING_M, and must be within PERP_ANGLE_MAX_DEG of the origin
    face's normal (a genuine crossing, not a diagonal drift down the
    corridor). Cost is the flat BETA, not distance.
    """
    edges: List[Tuple[str, str, float, str]] = []
    nodes = pn_graph.nodes
    n = len(nodes)
    for i in range(n):
        a = nodes[i]
        face_a = pn_graph.face_of[a.node_id]
        normal_a = pn_graph.face_normal_at(a.node_id)
        for j in range(i + 1, n):
            b = nodes[j]
            if pn_graph.face_of[b.node_id] == face_a:
                continue
            d = _euclidean(a.position, b.position)
            if d < 1e-9 or d > MAX_CROSSING_M:
                continue
            if pn_graph.crosses_any_wall(a.position, b.position):
                continue
            seg = np.array(b.position, dtype=float) - np.array(a.position, dtype=float)
            if not _within_perp_angle(seg, normal_a):
                # Also allow it if roughly perpendicular from b's own face
                normal_b = pn_graph.face_normal_at(b.node_id)
                if not _within_perp_angle(-seg, normal_b):
                    continue
            edges.append((a.node_id, b.node_id, BETA, "crossing"))
    return edges


def _within_perp_angle(seg: np.ndarray, face_normal: np.ndarray) -> bool:
    seg_n = np.linalg.norm(seg)
    if seg_n < 1e-9:
        return False
    cos_angle = float(np.dot(seg, face_normal) / seg_n)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle_deg = math.degrees(math.acos(cos_angle))
    return angle_deg <= PERP_ANGLE_MAX_DEG


# ── Public entry point ────────────────────────────────────────────────────────

def adjust_with_path_nodes(
    result:     PathResult,
    path_nodes: List[PathNode],
    walls:      List[Wall],
    zones:      Optional[List[Zone]] = None,
    user_xy:    Optional[Point2D] = None,
) -> PathResult:
    """
    Return a new PathResult whose `steps` walk a re-derived line over the
    path-node layer: the original graph route (`result`) is used only to
    pick the corridor/zone sequence (via its node sequence, for eligibility
    scoping) and to seed mandatory anchors (start, destination, every door/
    feature node on the route). The actual walking line is found by a
    single shortest-path search over path nodes + those anchors, charging a
    flat BETA for every corridor crossing and real distance for every
    same-face hop. All other PathResult fields (scores, cost, distance) are
    copied from `result` unchanged — they describe the graph-level route.

    Parameters
    ----------
    result     : PathResult — output of PathfindingEngine.find_path()
    path_nodes : path nodes for this floor (load_path_nodes())
    walls      : wall geometry for this floor (load_corridor_walls())
    zones      : optional zone geometry (load_zones()) — when given, widens
                 eligibility scoping to path nodes bordering a circulation
                 zone the original route touches; omitting it falls back to
                 route-polyline-buffer-only scoping.
    user_xy    : true user position; defaults to result.start_node.position
    """
    if not result.found or not result.steps:
        return result

    route_nodes = _reconstruct_route_nodes(result)

    pn_graph = _GlobalPathGraph(path_nodes, walls)

    # adjacency over a mixed id space: path-node ids ∪ route/mandatory graph-node ids
    adjacency: Dict[str, List[Tuple[str, float, str]]] = {
        nid: list(edges) for nid, edges in pn_graph.adjacency.items()
    }
    positions: Dict[str, Point2D] = {n.node_id: n.position for n in path_nodes}
    node_lookup: Dict[str, NavigationNode] = {n.node_id: n for n in route_nodes}

    for a_id, b_id, cost, kind in _build_crossing_edges(pn_graph):
        _mixed_link(adjacency, a_id, b_id, cost, kind)

    mandatory_nodes = _mandatory_nodes(route_nodes)
    for nav in mandatory_nodes:
        adjacency.setdefault(nav.node_id, [])
        positions[nav.node_id] = nav.position
        anchor = pn_graph.nearest(nav.position, SNAP_RADIUS)
        if anchor is not None:
            d = _euclidean(nav.position, anchor.position)
            _mixed_link(adjacency, nav.node_id, anchor.node_id, d, "mandatory")

    # Seed the original route's own edges as a fallback so a leg with no
    # viable path-node alternative still produces a route. Penalised so a
    # genuine hugging route (even if geometrically a little longer) is
    # always preferred — this edge should only win when nothing else does.
    for a, b in zip(route_nodes, route_nodes[1:]):
        d = _euclidean(a.position, b.position) * ORIGINAL_EDGE_PENALTY_FACTOR
        adjacency.setdefault(a.node_id, [])
        adjacency.setdefault(b.node_id, [])
        positions[a.node_id] = a.position
        positions[b.node_id] = b.position
        _mixed_link(adjacency, a.node_id, b.node_id, d, "original")

    eligible = _eligible_path_node_ids(route_nodes, path_nodes, zones)
    mandatory_ids = {n.node_id for n in mandatory_nodes} | {n.node_id for n in route_nodes}
    allowed = eligible | mandatory_ids

    start_id = route_nodes[0].node_id
    goal_id = route_nodes[-1].node_id

    winning = _dijkstra_mixed(adjacency, allowed, start_id, goal_id)
    if winning is None:
        return result
    winning_ids, winning_kinds = winning

    expanded_nodes: List[NavigationNode] = []
    synthetic_edges: Dict[Tuple[str, str], NavigationEdge] = {}
    pn_by_id = {n.node_id: n for n in path_nodes}

    for nid in winning_ids:
        if nid in node_lookup:
            expanded_nodes.append(node_lookup[nid])
        elif nid in pn_by_id:
            expanded_nodes.append(pn_by_id[nid].to_navigation_node())
        else:
            continue  # defensive — should not happen

    for a, b in zip(expanded_nodes, expanded_nodes[1:]):
        _add_synthetic_edge(synthetic_edges, a, b)

    lookup = _AdjustedEdgeLookup(result, synthetic_edges)
    start_pos = user_xy if user_xy is not None else expanded_nodes[0].position
    new_steps = build_steps(start_pos, expanded_nodes[0], expanded_nodes, lookup)

    adjusted = PathResult(
        found=result.found,
        start_node=result.start_node,
        destination_node=result.destination_node,
        steps=new_steps,
        alternatives=result.alternatives,
        total_cost=result.total_cost,
        total_distance=result.total_distance,
        safety_score=result.safety_score,
        shore_score=result.shore_score,
        landmark_score=result.landmark_score,
    )
    adjusted.adjustment_stats = _summarise_adjustment(winning_ids, winning_kinds)
    return adjusted


def _summarise_adjustment(
    winning_ids: List[str], winning_kinds: List[str],
) -> Dict[str, object]:
    """Measure how much of the winning route hugs the path-node layer versus
    how much fell back to the penalised original-route ("original") edges —
    the seam where junctions survive into the geometry. `fallback_hops > 0`
    means the shoreline graph could not carry the route the whole way and the
    junction skeleton was used for `fallback_node_ids` of the stretch; it is
    the evaluation metric for how often the all-path-node output degrades."""
    total = len(winning_kinds)
    fallback = sum(1 for k in winning_kinds if k == "original")
    fallback_node_ids: List[str] = []
    for i, kind in enumerate(winning_kinds):
        if kind == "original":
            fallback_node_ids.extend(winning_ids[i:i + 2])
    # de-dup while preserving order
    seen: Set[str] = set()
    fallback_node_ids = [
        nid for nid in fallback_node_ids
        if not (nid in seen or seen.add(nid))
    ]
    return {
        "total_hops": total,
        "fallback_hops": fallback,
        "clean": fallback == 0,
        "fallback_node_ids": fallback_node_ids,
    }


def _reconstruct_route_nodes(result: PathResult) -> List[NavigationNode]:
    """Reconstruct the original node sequence from the steps (steps[0].from_node
    may be the virtual USER node — skip it, build_steps re-synthesises it)."""
    orig_nodes: List[NavigationNode] = [result.start_node]
    for step in result.steps:
        if step.from_node.node_id == "USER":
            continue
        orig_nodes.append(step.to_node)
    nodes: List[NavigationNode] = [orig_nodes[0]]
    for n in orig_nodes[1:]:
        if n.node_id != nodes[-1].node_id:
            nodes.append(n)
    return nodes


def _mandatory_nodes(route_nodes: List[NavigationNode]) -> List[NavigationNode]:
    """Start, destination, and every door/feature node on the route — these
    get a mandatory (if in-range) connection to the path-node layer."""
    out: List[NavigationNode] = [route_nodes[0], route_nodes[-1]]
    door_types = {"door", "stair", "lift", "elevator", "escalator", "landmark"}
    for n in route_nodes[1:-1]:
        if n.node_type in door_types:
            out.append(n)
    # de-dup while preserving order
    seen: Set[str] = set()
    deduped: List[NavigationNode] = []
    for n in out:
        if n.node_id not in seen:
            seen.add(n.node_id)
            deduped.append(n)
    return deduped


def _eligible_path_node_ids(
    route_nodes: List[NavigationNode],
    path_nodes: List[PathNode],
    zones: Optional[List[Zone]],
) -> Set[str]:
    """Path nodes 'in play': within ROUTE_BUFFER_M of the original route's
    polyline, optionally widened (when zones is given) to any path node
    whose nearest wall borders a circulation zone the route touches."""
    eligible: Set[str] = set()
    poly = [n.position for n in route_nodes]
    for pn in path_nodes:
        if _dist_to_polyline(pn.position, poly) <= ROUTE_BUFFER_M:
            eligible.add(pn.node_id)

    if zones:
        touched_zone_ids = {n.zone_id for n in route_nodes if n.zone_id}
        circulation_polys = [
            z.boundary_polygon for z in zones
            if z.category in CIRCULATION_CATEGORIES and len(z.boundary_polygon) >= 3
        ]
        # Widen to any path node inside/near a circulation zone the route's
        # own zone_ids touch, or — when zone_id linkage is unavailable — any
        # circulation zone at all (still far cheaper than the whole floor).
        for pn in path_nodes:
            if pn.node_id in eligible:
                continue
            for poly_pts in circulation_polys:
                if _point_in_polygon(pn.position, poly_pts):
                    eligible.add(pn.node_id)
                    break
        del touched_zone_ids  # zone_id-level filtering left for a future refinement

    return eligible


def _dijkstra_mixed(
    adjacency: Dict[str, List[Tuple[str, float, str]]],
    allowed:   Set[str],
    start_id:  str,
    goal_id:   str,
) -> Optional[Tuple[List[str], List[str]]]:
    """Simple O(V^2) Dijkstra over the mixed graph, restricted to `allowed`
    node ids (path nodes within scoring + all mandatory/route node ids).
    Node counts per floor are small (hundreds), so this stays fast.

    Returns (node_chain, edge_kinds) where edge_kinds[i] is the kind of the
    hop from node_chain[i] to node_chain[i+1] ("path"/"crossing"/"mandatory"/
    "original"), so callers can measure how much of the winning route fell
    back to the penalised original-route ("original") edges. Returns None
    when no route exists."""
    if start_id == goal_id:
        return [start_id], []

    dist: Dict[str, float] = {start_id: 0.0}
    prev: Dict[str, str] = {}
    prev_kind: Dict[str, str] = {}
    visited: Set[str] = set()
    frontier = {start_id}

    while frontier:
        u = min(frontier, key=lambda n: dist.get(n, math.inf))
        frontier.discard(u)
        if u in visited:
            continue
        visited.add(u)
        if u == goal_id:
            break
        for v, w, kind in adjacency.get(u, []):
            if v in visited:
                continue
            if v != goal_id and v != start_id and v not in allowed:
                continue
            nd = dist[u] + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                prev_kind[v] = kind
                frontier.add(v)

    if goal_id not in dist:
        return None

    chain = [goal_id]
    kinds: List[str] = []
    cur = goal_id
    while cur != start_id:
        kinds.append(prev_kind[cur])
        cur = prev[cur]
        chain.append(cur)
    chain.reverse()
    kinds.reverse()
    return chain, kinds


def _mixed_link(
    adjacency: Dict[str, List[Tuple[str, float, str]]],
    a_id: str, b_id: str, cost: float, kind: str,
) -> None:
    adjacency.setdefault(a_id, [])
    adjacency.setdefault(b_id, [])
    if not any(nid == b_id for nid, _, _ in adjacency[a_id]):
        adjacency[a_id].append((b_id, cost, kind))
        adjacency[b_id].append((a_id, cost, kind))


def _add_synthetic_edge(
    store: Dict[Tuple[str, str], NavigationEdge],
    src:   NavigationNode,
    tgt:   NavigationNode,
) -> None:
    d = _euclidean(src.position, tgt.position)
    edge = NavigationEdge(
        edge_id=f"PN-EDGE-{src.node_id}-{tgt.node_id}",
        source_id=src.node_id,
        target_id=tgt.node_id,
        distance=d,
        shore_linable=True,
        safety_score=1.0,
        landmark_score=0.0,
    )
    store[(src.node_id, tgt.node_id)] = edge
    store[(tgt.node_id, src.node_id)] = edge


class _AdjustedEdgeLookup:
    """
    EdgeLookup for build_steps(): synthetic path-node-hop edges take priority,
    falling back to the original graph's edge lookup for untouched legs.
    """

    def __init__(
        self,
        result: PathResult,
        synthetic: Dict[Tuple[str, str], NavigationEdge],
    ) -> None:
        self._synthetic = synthetic
        self._fwd: Dict[Tuple[str, str], NavigationEdge] = {}
        self._nodes: Dict[str, NavigationNode] = {}
        for step in result.steps:
            self._fwd[(step.edge.source_id, step.edge.target_id)] = step.edge
            self._fwd[(step.edge.target_id, step.edge.source_id)] = step.edge
            self._nodes[step.from_node.node_id] = step.from_node
            self._nodes[step.to_node.node_id] = step.to_node

    def __call__(self, a_id: str, b_id: str) -> Optional[NavigationEdge]:
        return self._synthetic.get((a_id, b_id)) or self._fwd.get((a_id, b_id))

    def node(self, node_id: str) -> Optional[NavigationNode]:
        return self._nodes.get(node_id)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _euclidean(a: Point2D, b: Point2D) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _point_segment_dist(p: np.ndarray, s: np.ndarray, e: np.ndarray) -> float:
    seg = e - s
    seg_len_sq = float(np.dot(seg, seg))
    if seg_len_sq < 1e-12:
        return float(np.linalg.norm(p - s))
    t = max(0.0, min(1.0, float(np.dot(p - s, seg)) / seg_len_sq))
    proj = s + t * seg
    return float(np.linalg.norm(p - proj))


def _dist_to_polyline(p: Point2D, polyline: List[Point2D]) -> float:
    if len(polyline) < 2:
        return _euclidean(p, polyline[0]) if polyline else math.inf
    pt = np.array(p, dtype=float)
    best = math.inf
    for a, b in zip(polyline, polyline[1:]):
        d = _point_segment_dist(pt, np.array(a, dtype=float), np.array(b, dtype=float))
        if d < best:
            best = d
    return best


def _point_in_polygon(p: Point2D, poly: List[Point2D]) -> bool:
    """Standard ray-casting point-in-polygon test — no shapely dependency."""
    x, y = p
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def _segments_intersect(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray,
) -> bool:
    """True if segment p1->p2 properly crosses segment p3->p4 (standard
    orientation-based 2D segment intersection test)."""
    def _orient(a, b, c):
        val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(val) < 1e-12:
            return 0
        return 1 if val > 0 else -1

    def _on_segment(a, b, c):
        return (
            min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9
            and min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9
        )

    o1 = _orient(p1, p2, p3)
    o2 = _orient(p1, p2, p4)
    o3 = _orient(p3, p4, p1)
    o4 = _orient(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p2, p3):
        return True
    if o2 == 0 and _on_segment(p1, p2, p4):
        return True
    if o3 == 0 and _on_segment(p3, p4, p1):
        return True
    if o4 == 0 and _on_segment(p3, p4, p2):
        return True
    return False
