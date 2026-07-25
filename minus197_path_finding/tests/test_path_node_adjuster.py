"""
tests/test_path_node_adjuster.py
------------------------------------
Unit tests for pathfinding/path_node_adjuster.py.

Layout (all coordinates in metres) — a straight corridor along the x-axis,
bounded by a wall at y=2.0, with a chain of path nodes hugging that wall
0.3-ish m off it (gap_m used here is 0.3 to keep numbers small), plus a
separate "crossing" leg into a shop with no nearby path nodes at all.

  WALL  y=2.0  ───────────────────────────────────────────────
  PATH-0 (1,1.7) PATH-1 (4,1.7) PATH-2 (7,1.7) PATH-3 (10,1.7)

  JUNC-A (0,1.0) ───────────────── JUNC-B (11,1.0)   [along-corridor leg]
  JUNC-B (11,1.0) ─────────────────── DOOR-C (11,5.0) [corridor-crossing leg,
                                                        far from any path node]

Run:  pytest tests/test_path_node_adjuster.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from shared.types import NavigationEdge, NavigationNode, PathResult, PathStep
from pathfinding.instructions import bearing
from pathfinding.path_node_adjuster import (
    PathNode, Wall, adjust_with_path_nodes,
    BETA, CROSSING_SHORTCUT_FACTOR, SHORE_FRACTION,
    _add_synthetic_edge, _build_crossing_edges, _GlobalPathGraph,
    _path_only_distance,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def corridor_wall():
    return [Wall(wall_id="W1", start=(-5.0, 2.0), end=(15.0, 2.0))]


@pytest.fixture
def hugging_path_nodes():
    return [
        PathNode("PATH-0", (1.0, 1.7), "W1", 0.3),
        PathNode("PATH-1", (4.0, 1.7), "W1", 0.3),
        PathNode("PATH-2", (7.0, 1.7), "W1", 0.3),
        PathNode("PATH-3", (10.0, 1.7), "W1", 0.3),
    ]


def _node(node_id, pos, ntype="junction", zone_id=None):
    return NavigationNode(node_id, node_id, pos, ntype, zone_id)


def _edge(edge_id, a, b, dist, shore=True, safety=0.8, landmark=0.0):
    return NavigationEdge(edge_id, a.node_id, b.node_id, dist, shore, safety, landmark)


def _result_for_leg(a, b, edge, extra_steps=None):
    """Build a minimal found PathResult for a single a->b leg."""
    step = PathStep(
        from_node=a, to_node=b, edge=edge,
        bearing=bearing(a.position, b.position),
        distance=edge.distance,
        instruction=f"Walk to {b.node_id}.",
    )
    steps = [step] if extra_steps is None else extra_steps
    return PathResult(
        found=True,
        start_node=a,
        destination_node=b,
        steps=steps,
        total_distance=edge.distance,
        total_cost=edge.distance,
        safety_score=edge.safety_score,
        shore_score=1.0 if edge.shore_linable else 0.0,
        landmark_score=edge.landmark_score,
    )


# ── Corridor-hugging leg gets expanded ────────────────────────────────────────

class TestCorridorHugging:
    def test_along_corridor_leg_expands_through_path_nodes(
        self, corridor_wall, hugging_path_nodes,
    ):
        a = _node("JUNC-A", (0.0, 1.0))
        b = _node("JUNC-B", (11.0, 1.0))
        edge = _edge("E-AB", a, b, dist=11.0)
        result = _result_for_leg(a, b, edge)

        adjusted = adjust_with_path_nodes(result, hugging_path_nodes, corridor_wall)

        node_ids = [step.to_node.node_id for step in adjusted.steps]
        # Every path node between A and B should appear, in order.
        assert node_ids == ["PATH-0", "PATH-1", "PATH-2", "PATH-3", "JUNC-B"]

    def test_expanded_steps_have_one_hop_per_path_node(
        self, corridor_wall, hugging_path_nodes,
    ):
        a = _node("JUNC-A", (0.0, 1.0))
        b = _node("JUNC-B", (11.0, 1.0))
        edge = _edge("E-AB", a, b, dist=11.0)
        result = _result_for_leg(a, b, edge)

        adjusted = adjust_with_path_nodes(result, hugging_path_nodes, corridor_wall)
        assert len(adjusted.steps) == 5  # A->P0, P0->P1, P1->P2, P2->P3, P3->B

    def test_expanded_steps_have_nonempty_instructions(
        self, corridor_wall, hugging_path_nodes,
    ):
        a = _node("JUNC-A", (0.0, 1.0))
        b = _node("JUNC-B", (11.0, 1.0))
        edge = _edge("E-AB", a, b, dist=11.0)
        result = _result_for_leg(a, b, edge)

        adjusted = adjust_with_path_nodes(result, hugging_path_nodes, corridor_wall)
        for step in adjusted.steps:
            assert step.instruction.strip() != ""

    def test_original_scores_and_totals_preserved(
        self, corridor_wall, hugging_path_nodes,
    ):
        a = _node("JUNC-A", (0.0, 1.0))
        b = _node("JUNC-B", (11.0, 1.0))
        edge = _edge("E-AB", a, b, dist=11.0, safety=0.42, landmark=0.33)
        result = _result_for_leg(a, b, edge)

        adjusted = adjust_with_path_nodes(result, hugging_path_nodes, corridor_wall)
        # Graph-level scores describe the ORIGINAL route and must be untouched.
        assert adjusted.total_distance == result.total_distance
        assert adjusted.total_cost == result.total_cost
        assert adjusted.safety_score == result.safety_score
        assert adjusted.landmark_score == result.landmark_score


# ── Corridor-crossing leg stays direct ────────────────────────────────────────

class TestCorridorCrossing:
    def test_crossing_leg_with_no_nearby_path_nodes_stays_direct(
        self, corridor_wall, hugging_path_nodes,
    ):
        # DOOR-C is far from every path node (>SNAP_RADIUS), simulating a
        # corridor-crossing / doorway leg with no same-side chain available.
        b = _node("JUNC-B", (11.0, 1.0))
        c = _node("DOOR-C", (11.0, 20.0), ntype="door")
        edge = _edge("E-BC", b, c, dist=19.0, shore=False, safety=0.5)
        result = _result_for_leg(b, c, edge)

        adjusted = adjust_with_path_nodes(result, hugging_path_nodes, corridor_wall)

        node_ids = [step.to_node.node_id for step in adjusted.steps]
        assert node_ids == ["DOOR-C"]  # unchanged: single direct step

    def test_crossing_leg_keeps_original_distance(
        self, corridor_wall, hugging_path_nodes,
    ):
        b = _node("JUNC-B", (11.0, 1.0))
        c = _node("DOOR-C", (11.0, 20.0), ntype="door")
        edge = _edge("E-BC", b, c, dist=19.0, shore=False, safety=0.5)
        result = _result_for_leg(b, c, edge)

        adjusted = adjust_with_path_nodes(result, hugging_path_nodes, corridor_wall)
        assert adjusted.steps[0].distance == pytest.approx(19.0)


# ── Coverage gap falls back to direct ─────────────────────────────────────────

class TestCoverageGap:
    def test_gap_larger_than_max_chain_gap_falls_back_to_direct(self, corridor_wall):
        # Only two path nodes, 20 m apart — far beyond MAX_CHAIN_GAP (9 m) —
        # so no proximity edge links them; the leg must stay direct.
        sparse_nodes = [
            PathNode("PATH-0", (1.0, 1.7), "W1", 0.3),
            PathNode("PATH-1", (21.0, 1.7), "W1", 0.3),
        ]
        a = _node("JUNC-A", (0.0, 1.0))
        b = _node("JUNC-B", (22.0, 1.0))
        edge = _edge("E-AB", a, b, dist=22.0)
        result = _result_for_leg(a, b, edge)

        adjusted = adjust_with_path_nodes(result, sparse_nodes, corridor_wall)
        node_ids = [step.to_node.node_id for step in adjusted.steps]
        assert node_ids == ["JUNC-B"]


# ── Adjustment stats: junction-fallback reporting ─────────────────────────────

class TestAdjustmentStats:
    def test_clean_hugging_route_reports_zero_fallback(
        self, corridor_wall, hugging_path_nodes,
    ):
        # A -> path nodes -> B: A and B both snap to the path layer within
        # SNAP_RADIUS, so the whole route rides "path"/"mandatory" edges and
        # no original-edge fallback is used.
        a = _node("JUNC-A", (0.0, 1.0))
        b = _node("JUNC-B", (11.0, 1.0))
        edge = _edge("E-AB", a, b, dist=11.0)
        result = _result_for_leg(a, b, edge)

        adjusted = adjust_with_path_nodes(result, hugging_path_nodes, corridor_wall)
        stats = adjusted.adjustment_stats
        assert stats["clean"] is True
        assert stats["fallback_hops"] == 0
        assert stats["fallback_node_ids"] == []
        assert stats["total_hops"] == len(adjusted.steps)

    def test_coverage_gap_reports_fallback_hops(self, corridor_wall):
        # Same sparse-gap layout as TestCoverageGap: the only route is the
        # penalised original A->B edge, so the fallback must be flagged and
        # both endpoints reported.
        sparse_nodes = [
            PathNode("PATH-0", (1.0, 1.7), "W1", 0.3),
            PathNode("PATH-1", (21.0, 1.7), "W1", 0.3),
        ]
        a = _node("JUNC-A", (0.0, 1.0))
        b = _node("JUNC-B", (22.0, 1.0))
        edge = _edge("E-AB", a, b, dist=22.0)
        result = _result_for_leg(a, b, edge)

        adjusted = adjust_with_path_nodes(result, sparse_nodes, corridor_wall)
        stats = adjusted.adjustment_stats
        assert stats["clean"] is False
        assert stats["fallback_hops"] >= 1
        assert "JUNC-A" in stats["fallback_node_ids"]
        assert "JUNC-B" in stats["fallback_node_ids"]


# ── Corner: chain continues onto a perpendicular wall ─────────────────────────

class TestCorner:
    def test_chain_bridges_a_corner_onto_a_second_wall(self):
        # Two walls meeting at a right-angle corner: W1 along y=2 (x: 0..6),
        # W2 along x=6 (y: 2..8). Path nodes hug each wall on its corridor
        # side; the chain must turn the corner and touch every node.
        walls = [
            Wall(wall_id="W1", start=(0.0, 2.0), end=(6.0, 2.0)),
            Wall(wall_id="W2", start=(6.0, 2.0), end=(6.0, 8.0)),
        ]
        path_nodes = [
            PathNode("PATH-0", (1.0, 1.7), "W1", 0.3),
            PathNode("PATH-1", (4.0, 1.7), "W1", 0.3),
            PathNode("PATH-2", (6.3, 3.0), "W2", 0.3),
            PathNode("PATH-3", (6.3, 6.0), "W2", 0.3),
        ]
        a = _node("JUNC-A", (0.0, 1.0))
        b = _node("JUNC-B", (6.3, 7.0))
        edge = _edge("E-AB", a, b, dist=9.5)
        result = _result_for_leg(a, b, edge)

        adjusted = adjust_with_path_nodes(result, path_nodes, walls)
        node_ids = [step.to_node.node_id for step in adjusted.steps]
        assert node_ids == ["PATH-0", "PATH-1", "PATH-2", "PATH-3", "JUNC-B"]


# ── No path nodes / no walls at all ───────────────────────────────────────────

class TestNoPathNodeData:
    def test_empty_path_nodes_leaves_route_unchanged(self, corridor_wall):
        a = _node("JUNC-A", (0.0, 1.0))
        b = _node("JUNC-B", (11.0, 1.0))
        edge = _edge("E-AB", a, b, dist=11.0)
        result = _result_for_leg(a, b, edge)

        adjusted = adjust_with_path_nodes(result, [], corridor_wall)
        node_ids = [step.to_node.node_id for step in adjusted.steps]
        assert node_ids == ["JUNC-B"]

    def test_not_found_result_returned_unchanged(self, corridor_wall, hugging_path_nodes):
        result = PathResult(found=False)
        adjusted = adjust_with_path_nodes(result, hugging_path_nodes, corridor_wall)
        assert adjusted is result


# ── Crossing edge: genuine perpendicular shortcut across the corridor ────────

class TestCrossingEdge:
    def test_perpendicular_crossing_is_used_between_two_wall_faces(self):
        # Two parallel walls (a NARROW 3 m corridor): bottom wall at y=0,
        # top wall at y=3. Path nodes hug each wall 0.3 m off it. Start is
        # near the bottom-left, goal is near the top-right — a real route
        # MUST cross the corridor at some point. The corridor is narrow
        # enough that only the crossing directly opposite the top node
        # (P-B3 -> P-T3, angle 0 deg from the wall normal) stays within
        # PERP_ANGLE_MAX_DEG=35; every earlier bottom node's angle to P-T3
        # exceeds 35 deg (a shallow diagonal drift, not a genuine crossing),
        # so the route must hug the full bottom face before crossing once.
        walls = [
            Wall(wall_id="BOTTOM", start=(-2.0, 0.0), end=(12.0, 0.0)),
            Wall(wall_id="TOP", start=(-2.0, 3.0), end=(12.0, 3.0)),
        ]
        bottom = [
            PathNode("P-B0", (1.0, 0.3), "BOTTOM", 0.3),
            PathNode("P-B1", (4.0, 0.3), "BOTTOM", 0.3),
            PathNode("P-B2", (7.0, 0.3), "BOTTOM", 0.3),
            PathNode("P-B3", (10.0, 0.3), "BOTTOM", 0.3),
        ]
        top = [
            PathNode("P-T3", (10.0, 2.7), "TOP", 0.3),
        ]
        a = _node("JUNC-A", (0.0, 0.5))
        b = _node("JUNC-B", (11.0, 2.5))
        edge = _edge("E-AB", a, b, dist=11.02)
        result = _result_for_leg(a, b, edge)

        adjusted = adjust_with_path_nodes(result, bottom + top, walls)
        node_ids = [step.to_node.node_id for step in adjusted.steps]

        # Must hug the bottom face out to P-B3, cross once to P-T3, then reach B.
        assert node_ids == ["P-B0", "P-B1", "P-B2", "P-B3", "P-T3", "JUNC-B"]


# ── Crossing edge rejected: intervening wall blocks the straight line ────────

class TestCrossingBlockedByWall:
    def test_crossing_blocked_by_intervening_wall_is_rejected(self):
        # Same narrow two-parallel-wall corridor as TestCrossingEdge, but a
        # wall stub crosses directly between the two nearest opposite-face
        # nodes at x=10, so the straight P-B3 -> P-T3 crossing is not
        # walkable and must not become a crossing-edge candidate — even
        # though it would otherwise pass the distance and perpendicularity
        # checks (it's the same geometry as TestCrossingEdge's accepted
        # case, just with a wall now in the way).
        walls = [
            Wall(wall_id="BOTTOM", start=(-2.0, 0.0), end=(12.0, 0.0)),
            Wall(wall_id="TOP", start=(-2.0, 3.0), end=(12.0, 3.0)),
            Wall(wall_id="BLOCKER", start=(8.0, 1.5), end=(12.0, 1.5)),
        ]
        p_b3 = PathNode("P-B3", (10.0, 0.3), "BOTTOM", 0.3)
        p_t3 = PathNode("P-T3", (10.0, 2.7), "TOP", 0.3)

        graph = _GlobalPathGraph([p_b3, p_t3], walls)
        edges = _build_crossing_edges(graph)

        pairs = {(a, b) for a, b, _, _ in edges} | {(b, a) for a, b, _, _ in edges}
        assert ("P-B3", "P-T3") not in pairs


# ── Both sides of a free-standing wall are different faces ──────────────────

class TestFreeStandingWallFaces:
    def test_opposite_sides_of_same_wall_id_are_not_chained_together(self):
        # A free-standing wall (corridor on both sides) shares one wall_id
        # for nodes on both faces (mirrors _corridor_sides in
        # map_extraction/path_nodes.py placing nodes on both sides). Nodes
        # on opposite sides must NOT be treated as same-face neighbours —
        # only a genuine (walkable, perpendicular, in-range) crossing edge
        # may link them.
        wall = Wall(wall_id="FS1", start=(5.0, 0.0), end=(5.0, 10.0))
        plus_side = PathNode("P-PLUS", (4.7, 5.0), "FS1", 0.3)   # face sign +1
        minus_side = PathNode("P-MINUS", (5.3, 5.0), "FS1", 0.3)  # face sign -1

        graph = _GlobalPathGraph([plus_side, minus_side], [wall])

        assert graph.face_of["P-PLUS"] != graph.face_of["P-MINUS"]
        # No same-face/corner link should have been created between them
        # (they're each other's only same-wall neighbour, but on different
        # faces, so _build_edges must not link them as an in-face run).
        linked_ids = {nid for nid, _, _ in graph.adjacency["P-PLUS"]}
        assert "P-MINUS" not in linked_ids


# ── Perpendicularity boundary: a shallow-angle drift is rejected ────────────

class TestPerpendicularityBoundary:
    def test_shallow_angle_crossing_is_rejected(self):
        # Bottom wall at y=0, path node at (5, 0.3). A candidate "crossing"
        # target sits far to the side — the segment from the path node to
        # it is ~40 degrees off the wall's normal (well past
        # PERP_ANGLE_MAX_DEG=35), i.e. a diagonal drift down the corridor
        # rather than a genuine perpendicular crossing, so it must not
        # become a crossing edge candidate.
        walls = [Wall(wall_id="BOTTOM", start=(-2.0, 0.0), end=(12.0, 0.0))]
        p = PathNode("P-B0", (5.0, 0.3), "BOTTOM", 0.3)
        shallow_target = PathNode("P-SHALLOW", (9.2, 5.3), "OTHER", 0.3)

        graph = _GlobalPathGraph([p, shallow_target], walls)
        edges = _build_crossing_edges(graph)

        pairs = {(a, b) for a, b, _, _ in edges} | {(b, a) for a, b, _, _ in edges}
        assert ("P-B0", "P-SHALLOW") not in pairs


# ── Corner-bridge precedence: real distance beats a same-cost crossing ──────

class TestCornerVsCrossingPrecedence:
    def test_corner_bridge_edge_dominates_over_crossing_edge(self):
        # Same corner fixture as TestCorner: PATH-1 and PATH-2 sit on
        # different wall_ids (a genuine corner, not a free-standing wall),
        # so they qualify as both a corner-bridge (real distance, ~2.7 m)
        # and, geometrically, a crossing-edge candidate (different faces).
        # Dijkstra must prefer the far cheaper real-distance corner edge,
        # not the flat BETA=8.0 crossing cost, confirming corner edges are
        # not shadowed or overridden by crossing edges between the same pair.
        walls = [
            Wall(wall_id="W1", start=(0.0, 2.0), end=(6.0, 2.0)),
            Wall(wall_id="W2", start=(6.0, 2.0), end=(6.0, 8.0)),
        ]
        path_nodes = [
            PathNode("PATH-1", (4.0, 1.7), "W1", 0.3),
            PathNode("PATH-2", (6.3, 3.0), "W2", 0.3),
        ]
        graph = _GlobalPathGraph(path_nodes, walls)
        edge_costs = {
            (nid, cost) for nid, cost, _kind in graph.adjacency["PATH-1"]
        }
        # The corner-bridge link (kind="path") must exist and cost real
        # distance (~2.7 m), strictly cheaper than BETA.
        matching = [cost for nid, cost in edge_costs if nid == "PATH-2"]
        assert matching, "expected a corner-bridge link between PATH-1 and PATH-2"
        assert matching[0] < BETA


# ── Regression: crossing edges must not short-circuit a connected wall route ─
# See LAST_MILE_FIX.md §2. On the L4 ODEL->Popeyes route a flat-BETA crossing
# beat a fully connected 15.62 m wall chain purely because 8.0 < 15.62, putting
# the user on an 11.25 m unreferenced diagonal. A crossing must only be offered
# where the wall does NOT already connect the two nodes.

class TestCrossingShortCircuit:
    @staticmethod
    def _connected_corner():
        """
        An L-shaped corner whose two faces ARE joined by wall-hugging links,
        and whose endpoints are also geometrically eligible as a crossing pair
        (different faces, unobstructed, near-perpendicular, < MAX_CROSSING_M).
        The diagonal is shorter than walking the corner, so without
        suppression the crossing wins.
        """
        walls = [
            Wall(wall_id="W1", start=(0.0, 2.0), end=(6.0, 2.0)),
            Wall(wall_id="W2", start=(6.0, 2.0), end=(6.0, 8.0)),
        ]
        path_nodes = [
            PathNode("P-A", (1.0, 1.7), "W1", 0.3),
            PathNode("P-B", (4.0, 1.7), "W1", 0.3),
            PathNode("P-C", (6.3, 3.0), "W2", 0.3),
            PathNode("P-D", (6.3, 6.0), "W2", 0.3),
        ]
        # Wall route P-A -> P-D is 8.64 m; the straight diagonal is 6.82 m, so
        # an unsuppressed flat-BETA crossing would win. This is the L4 defect
        # in miniature.
        return walls, path_nodes

    def test_crossing_suppressed_when_wall_route_exists(self):
        walls, path_nodes = self._connected_corner()
        graph = _GlobalPathGraph(path_nodes, walls)

        # Precondition: the wall genuinely connects the two faces.
        assert _path_only_distance(
            graph, "P-A", "P-D", limit=CROSSING_SHORTCUT_FACTOR * BETA,
        ) is not None

        crossings = _build_crossing_edges(graph)
        pairs = {frozenset((a, b)) for a, b, _c, _k in crossings}
        assert frozenset(("P-A", "P-D")) not in pairs, (
            "a crossing must not short-circuit an already-connected wall route"
        )

    def test_crossing_kept_when_no_wall_route_exists(self):
        # Two facing walls with no corner joining them: the ONLY way across is
        # the crossing edge, so suppression must leave it in place.
        walls = [
            Wall(wall_id="BOTTOM", start=(-2.0, 0.0), end=(12.0, 0.0)),
            Wall(wall_id="TOP", start=(-2.0, 3.0), end=(12.0, 3.0)),
        ]
        path_nodes = [
            PathNode("P-B3", (10.0, 0.3), "BOTTOM", 0.3),
            PathNode("P-T3", (10.0, 2.7), "TOP", 0.3),
        ]
        graph = _GlobalPathGraph(path_nodes, walls)

        assert _path_only_distance(
            graph, "P-B3", "P-T3", limit=CROSSING_SHORTCUT_FACTOR * BETA,
        ) is None

        pairs = {frozenset((a, b)) for a, b, _c, _k in _build_crossing_edges(graph)}
        assert frozenset(("P-B3", "P-T3")) in pairs, (
            "a crossing bridging a genuinely disconnected pair must be kept"
        )

    def test_bounded_search_returns_none_beyond_limit(self):
        walls, path_nodes = self._connected_corner()
        graph = _GlobalPathGraph(path_nodes, walls)
        # The real wall route is several metres; a 1 m budget cannot reach it.
        assert _path_only_distance(graph, "P-A", "P-D", limit=1.0) is None


# ── Regression: synthetic edges must MEASURE shore_linable, not assert it ────
# See LAST_MILE_FIX.md §3. Every synthetic hop previously claimed
# shore_linable=True, so an open-floor diagonal was indistinguishable from a
# wall run in any downstream metric.

class TestSyntheticEdgeShoreHonesty:
    def test_wall_hugging_hop_is_shore_linable(
        self, corridor_wall, hugging_path_nodes,
    ):
        a = _node("JUNC-A", (0.0, 1.7))
        b = _node("JUNC-B", (11.0, 1.7))
        edge = _edge("E-AB", a, b, dist=11.0)
        adjusted = adjust_with_path_nodes(
            _result_for_leg(a, b, edge), hugging_path_nodes, corridor_wall,
        )
        hops = [s for s in adjusted.steps if s.edge.edge_id.startswith("PN-EDGE-")]
        assert hops, "expected synthetic path-node hops"
        assert all(s.edge.shore_linable for s in hops), (
            "hops running 0.3 m off a wall must report shore_linable=True"
        )

    def test_open_floor_hop_is_not_shore_linable(self):
        """
        A hop whose length runs far from any wall must report False. Two short
        wall stubs sit at the ends of a wide gap, so the connecting hop leaves
        the shoreline entirely in the middle.
        """
        walls = [
            Wall(wall_id="W1", start=(0.0, 0.0), end=(3.0, 0.0)),
            Wall(wall_id="W2", start=(30.0, 0.0), end=(33.0, 0.0)),
        ]
        path_nodes = [
            PathNode("P-A", (1.5, 2.0), "W1", 2.0),
            PathNode("P-B", (31.5, 2.0), "W2", 2.0),
        ]
        graph = _GlobalPathGraph(path_nodes, walls)
        store = {}
        _add_synthetic_edge(
            store,
            _node("P-A", (1.5, 2.0)),
            _node("P-B", (31.5, 2.0)),
            graph,
        )
        edge = store[("P-A", "P-B")]
        assert edge.shore_linable is False, (
            "a hop crossing open floor must not claim to be shore-linable"
        )
        assert float(edge.tags["shore_fraction"]) < SHORE_FRACTION

    def test_shore_fraction_is_recorded_on_every_synthetic_edge(
        self, corridor_wall, hugging_path_nodes,
    ):
        a = _node("JUNC-A", (0.0, 1.7))
        b = _node("JUNC-B", (11.0, 1.7))
        edge = _edge("E-AB", a, b, dist=11.0)
        adjusted = adjust_with_path_nodes(
            _result_for_leg(a, b, edge), hugging_path_nodes, corridor_wall,
        )
        for step in adjusted.steps:
            if step.edge.edge_id.startswith("PN-EDGE-"):
                assert "shore_fraction" in step.edge.tags
