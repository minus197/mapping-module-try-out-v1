"""
tests/test_wp1_cost.py  —  M2 gate (WP1)
------------------------------------------
Unit tests for pathfinding/cost.py.
All tests must be green before WP4 / WP6 start.

Run:  pytest tests/test_wp1_cost.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from shared.types import FloorGraph, NavigationEdge, NavigationNode
from pathfinding.cost import CostWeights, compute_edge_costs, normalise_landmark_scores


# ── Helpers ───────────────────────────────────────────────────────────────────
def _edge(eid, dist, shore, safety, landmark) -> NavigationEdge:
    return NavigationEdge(
        edge_id=eid, source_id="A", target_id="B",
        distance=dist, shore_linable=shore,
        safety_score=safety, landmark_score=landmark,
    )

def _mini_graph(*edges) -> FloorGraph:
    fg = FloorGraph(floor_label="L1", source_file="test")
    fg.nodes = [NavigationNode("A", "A", (0,0), "junction", None),
                NavigationNode("B", "B", (1,0), "junction", None)]
    fg.edges = list(edges)
    fg.rebuild_index()
    return fg


# ── CostWeights ───────────────────────────────────────────────────────────────
class TestCostWeights:
    def test_default_sums_to_one(self):
        CostWeights().validate()   # must not raise

    def test_validate_raises_on_bad_sum(self):
        with pytest.raises(ValueError):
            CostWeights(lam=0.5, mu=0.5, nu=0.5).validate()

    def test_validate_raises_on_zero(self):
        with pytest.raises(ValueError):
            CostWeights(lam=0.0, mu=0.0, nu=0.0).validate()

    def test_custom_valid_weights(self):
        CostWeights(lam=0.5, mu=0.3, nu=0.2).validate()  # must not raise


# ── normalise_landmark_scores ─────────────────────────────────────────────────
class TestNormaliseLandmarkScores:
    def test_all_scores_clamped_to_one_or_below(self):
        fg = _mini_graph(
            _edge("e1", 1.0, False, 0.5, 999.0),  # outlier
            _edge("e2", 1.0, False, 0.5, 0.5),
        )
        normalise_landmark_scores(fg)
        for e in fg.edges:
            assert 0.0 <= e.landmark_score <= 1.0

    def test_outlier_clamps_to_exactly_one(self):
        fg = _mini_graph(
            _edge("e1", 1.0, False, 0.5, 100.0),
            _edge("e2", 1.0, False, 0.5, 0.5),
        )
        normalise_landmark_scores(fg)
        assert fg.edges[0].landmark_score == pytest.approx(1.0)

    def test_normal_edge_stays_in_range(self):
        fg = _mini_graph(_edge("e1", 1.0, False, 0.5, 0.6))
        normalise_landmark_scores(fg, landmark_max=1.0)
        assert 0.0 <= fg.edges[0].landmark_score <= 1.0

    def test_returns_landmark_max_used(self):
        fg = _mini_graph(_edge("e1", 1.0, False, 0.5, 0.8))
        lm = normalise_landmark_scores(fg, landmark_max=2.0)
        assert lm == pytest.approx(2.0)

    def test_empty_graph_returns_one(self):
        fg = FloorGraph(floor_label="L1", source_file="t")
        lm = normalise_landmark_scores(fg)
        assert lm == pytest.approx(1.0)

    def test_fixture_already_normalised(self, tiny_graph):
        # tiny_graph landmark scores are all ≤ 1 already — max stays 1
        lm = normalise_landmark_scores(tiny_graph, landmark_max=1.0)
        for e in tiny_graph.edges:
            assert 0.0 <= e.landmark_score <= 1.0


# ── compute_edge_costs ────────────────────────────────────────────────────────
class TestComputeEdgeCosts:
    def test_perfect_edge_cost_equals_distance(self):
        fg = _mini_graph(_edge("e1", 10.0, True, 1.0, 1.0))
        compute_edge_costs(fg, CostWeights())
        assert fg.edges[0].combined_cost == pytest.approx(10.0)

    def test_worst_edge_cost_equals_two_times_distance(self):
        fg = _mini_graph(_edge("e1", 10.0, False, 0.0, 0.0))
        compute_edge_costs(fg, CostWeights())
        assert fg.edges[0].combined_cost == pytest.approx(20.0)

    def test_no_negative_costs(self, tiny_graph):
        normalise_landmark_scores(tiny_graph, landmark_max=1.0)
        compute_edge_costs(tiny_graph, CostWeights())
        for e in tiny_graph.edges:
            assert e.combined_cost >= 0.0

    def test_golden_route_x_cost(self, tiny_graph, golden):
        normalise_landmark_scores(tiny_graph, landmark_max=1.0)
        compute_edge_costs(tiny_graph, CostWeights())
        idx = {e.edge_id: e for e in tiny_graph.edges}
        assert idx["EX1"].combined_cost == pytest.approx(golden["EX1_cost"], rel=1e-4)
        assert idx["EX2"].combined_cost == pytest.approx(golden["EX2_cost"], rel=1e-4)

    def test_golden_route_y_cost(self, tiny_graph, golden):
        normalise_landmark_scores(tiny_graph, landmark_max=1.0)
        compute_edge_costs(tiny_graph, CostWeights())
        idx = {e.edge_id: e for e in tiny_graph.edges}
        assert idx["EY1"].combined_cost == pytest.approx(golden["EY1_cost"], rel=1e-4)
        assert idx["EY2"].combined_cost == pytest.approx(golden["EY2_cost"], rel=1e-4)

    def test_route_y_cheaper_than_route_x(self, tiny_graph, golden):
        normalise_landmark_scores(tiny_graph, landmark_max=1.0)
        compute_edge_costs(tiny_graph, CostWeights())
        idx = {e.edge_id: e for e in tiny_graph.edges}
        total_x = idx["EX1"].combined_cost + idx["EX2"].combined_cost
        total_y = idx["EY1"].combined_cost + idx["EY2"].combined_cost
        assert total_y < total_x, "Option A: higher-quality route Y must win on combined_cost"

    def test_cost_proportional_to_distance(self):
        # Same quality, different lengths → costs scale proportionally
        fg = _mini_graph(
            _edge("short", 5.0, False, 0.5, 0.5),
            _edge("long",  10.0, False, 0.5, 0.5),
        )
        compute_edge_costs(fg, CostWeights())
        short_c, long_c = fg.edges[0].combined_cost, fg.edges[1].combined_cost
        assert long_c == pytest.approx(2 * short_c, rel=1e-9)
