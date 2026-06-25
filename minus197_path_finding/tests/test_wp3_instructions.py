"""
tests/test_wp3_instructions.py  —  M2 gate (WP3)
--------------------------------------------------
Unit tests for pathfinding/instructions.py.

Run:  pytest tests/test_wp3_instructions.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import pytest
from shared.types import NavigationEdge, NavigationNode
from pathfinding.instructions import bearing, build_steps, landmark_hint, turn_phrase


# ── bearing ───────────────────────────────────────────────────────────────────
class TestBearing:
    def test_north(self):
        assert bearing((0, 0), (0, 10)) == pytest.approx(0.0, abs=1e-6)

    def test_east(self):
        assert bearing((0, 0), (10, 0)) == pytest.approx(90.0, abs=1e-6)

    def test_south(self):
        assert bearing((0, 0), (0, -10)) == pytest.approx(180.0, abs=1e-6)

    def test_west(self):
        # atan2(-10, 0) = -90° → wrapped to 270°
        assert bearing((0, 0), (-10, 0)) == pytest.approx(270.0, abs=1e-6)

    def test_northeast_diagonal(self):
        b = bearing((0, 0), (1, 1))
        assert b == pytest.approx(45.0, abs=1e-6)


# ── turn_phrase ───────────────────────────────────────────────────────────────
class TestTurnPhrase:
    # boundary at ±20
    def test_straight_at_zero(self):
        assert turn_phrase(0) == "Continue straight"

    def test_straight_at_positive_20(self):
        assert turn_phrase(20) == "Continue straight"

    def test_straight_at_negative_20(self):
        assert turn_phrase(-20) == "Continue straight"

    def test_bear_right_just_above_20(self):
        assert turn_phrase(21) == "Bear right"

    def test_bear_left_just_above_20_negative(self):
        assert turn_phrase(-21) == "Bear left"

    # boundary at ±100
    def test_bear_right_at_100(self):
        assert turn_phrase(100) == "Bear right"

    def test_turn_right_just_above_100(self):
        assert turn_phrase(101) == "Turn right"

    def test_bear_left_at_negative_100(self):
        assert turn_phrase(-100) == "Bear left"

    def test_turn_left_just_above_negative_100(self):
        assert turn_phrase(-101) == "Turn left"

    # boundary at ±170
    def test_turn_right_at_170(self):
        assert turn_phrase(170) == "Turn right"

    def test_turn_around_just_above_170(self):
        assert turn_phrase(171) == "Turn around"

    def test_turn_left_at_negative_170(self):
        assert turn_phrase(-170) == "Turn left"

    def test_turn_around_just_above_negative_170(self):
        assert turn_phrase(-171) == "Turn around"

    def test_turn_around_at_180(self):
        assert turn_phrase(180) == "Turn around"

    # large values wrap correctly
    def test_wraps_360(self):
        assert turn_phrase(360) == "Continue straight"

    def test_wraps_350(self):
        # 350 normalises to -10 → straight
        assert turn_phrase(350) == "Continue straight"


# ── landmark_hint ─────────────────────────────────────────────────────────────
class TestLandmarkHint:
    def _node(self, ntype, tags=None):
        return NavigationNode("N1", "Some Node", (0, 0), ntype, None, tags or {})

    def test_door_type_fires(self):
        assert landmark_hint(self._node("door")) != ""

    def test_stair_type_fires(self):
        assert landmark_hint(self._node("stair")) != ""

    def test_lift_type_fires(self):
        assert landmark_hint(self._node("lift")) != ""

    def test_elevator_type_fires(self):
        assert landmark_hint(self._node("elevator")) != ""

    def test_escalator_type_fires(self):
        assert landmark_hint(self._node("escalator")) != ""

    def test_landmark_type_fires(self):
        assert landmark_hint(self._node("landmark")) != ""

    def test_admin_label_fires(self):
        hint = landmark_hint(self._node("junction", {"admin_label": "Nike Store"}))
        assert "Nike Store" in hint

    def test_plain_junction_no_hint(self):
        assert landmark_hint(self._node("junction")) == ""

    def test_zone_centroid_no_hint(self):
        assert landmark_hint(self._node("zone_centroid")) == ""


# ── build_steps ───────────────────────────────────────────────────────────────

def _make_edge_lookup(edges):
    """Minimal edge lookup over a flat edge list."""
    fwd = {(e.source_id, e.target_id): e for e in edges}
    rev = {(e.target_id, e.source_id): e for e in edges}

    class _Lookup:
        def __call__(self, a_id, b_id):
            return fwd.get((a_id, b_id)) or rev.get((a_id, b_id))
        def node(self, node_id):
            return None

    return _Lookup()


class TestBuildSteps:
    def _two_node_graph(self):
        n_start = NavigationNode("A", "Start", (0.0, 0.0), "junction", None)
        n_dest  = NavigationNode("B", "Dest",  (10.0, 0.0), "door", None)
        edge    = NavigationEdge("E1", "A", "B", 10.0, True, 0.9, 0.8)
        return n_start, n_dest, edge

    def test_first_step_emitted_when_far_from_start(self):
        n_start, n_dest, edge = self._two_node_graph()
        lookup = _make_edge_lookup([edge])
        # User is 2.0 m from start node — should produce a "walk to start" step
        steps = build_steps((-2.0, 0.0), n_start, [n_start, n_dest], lookup)
        assert len(steps) == 2
        assert steps[0].from_node.node_id == "USER"

    def test_first_step_absent_when_on_start(self):
        n_start, n_dest, edge = self._two_node_graph()
        lookup = _make_edge_lookup([edge])
        # User is exactly on the start node (0 m away)
        steps = build_steps((0.0, 0.0), n_start, [n_start, n_dest], lookup)
        assert len(steps) == 1
        assert steps[0].from_node.node_id == "A"

    def test_first_step_absent_when_within_threshold(self):
        n_start, n_dest, edge = self._two_node_graph()
        lookup = _make_edge_lookup([edge])
        # User is 0.3 m from start — below 0.5 m threshold
        steps = build_steps((0.3, 0.0), n_start, [n_start, n_dest], lookup)
        assert len(steps) == 1

    def test_step_count_matches_path_length(self, tiny_graph):
        start = tiny_graph.node("SKE-START")
        mid   = tiny_graph.node("MID-X")
        dest  = tiny_graph.node("ZONE-DEST")
        lookup = _make_edge_lookup(tiny_graph.edges)
        steps = build_steps(start.position, start, [start, mid, dest], lookup)
        assert len(steps) == 2

    def test_instruction_contains_distance(self):
        n_start, n_dest, edge = self._two_node_graph()
        lookup = _make_edge_lookup([edge])
        steps = build_steps((0.0, 0.0), n_start, [n_start, n_dest], lookup)
        assert "10 m" in steps[0].instruction

    def test_landmark_node_mentioned_in_instruction(self):
        n_start, n_dest, edge = self._two_node_graph()
        lookup = _make_edge_lookup([edge])
        steps = build_steps((0.0, 0.0), n_start, [n_start, n_dest], lookup)
        # n_dest is type "door" — landmark_hint should fire
        assert "Dest" in steps[-1].instruction
