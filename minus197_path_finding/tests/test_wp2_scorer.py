"""
tests/test_wp2_scorer.py  —  M2 gate (WP2)
--------------------------------------------
Unit tests for pathfinding/scorer.py.

Run:  pytest tests/test_wp2_scorer.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from shared.types import NavigationEdge
from pathfinding.scorer import score_path


def _edge(eid, dist, shore, safety, landmark) -> NavigationEdge:
    return NavigationEdge(
        edge_id=eid, source_id="A", target_id="B",
        distance=dist, shore_linable=shore,
        safety_score=safety, landmark_score=landmark,
    )


class TestScorePath:
    def test_empty_returns_zeros(self):
        assert score_path([]) == (0.0, 0.0, 0.0)

    def test_zero_distance_returns_zeros(self):
        e = _edge("e1", 0.0, True, 0.9, 0.8)
        assert score_path([e]) == (0.0, 0.0, 0.0)

    def test_all_shore_linable_gives_shore_one(self):
        edges = [_edge("e1", 5.0, True, 0.8, 0.5),
                 _edge("e2", 5.0, True, 0.8, 0.5)]
        _, shore, _ = score_path(edges)
        assert shore == pytest.approx(1.0)

    def test_no_shore_linable_gives_shore_zero(self):
        edges = [_edge("e1", 5.0, False, 0.8, 0.5),
                 _edge("e2", 3.0, False, 0.8, 0.5)]
        _, shore, _ = score_path(edges)
        assert shore == pytest.approx(0.0)

    def test_distance_weighted_long_edge_dominates(self):
        # Long low-safety edge should drag safety score down
        edges = [
            _edge("e1",  1.0, False, 1.0, 0.0),
            _edge("e2", 9.0, False, 0.0, 0.0),
        ]
        safety, _, _ = score_path(edges)
        assert safety < 0.2

    def test_golden_route_x_scores(self, tiny_graph, golden):
        ex1 = next(e for e in tiny_graph.edges if e.edge_id == "EX1")
        ex2 = next(e for e in tiny_graph.edges if e.edge_id == "EX2")
        safety, shore, landmark = score_path([ex1, ex2])
        assert safety   == pytest.approx(golden["route_X_safety"],   rel=1e-4)
        assert shore    == pytest.approx(golden["route_X_shore"],     abs=1e-9)
        assert landmark == pytest.approx(golden["route_X_landmark"],  rel=1e-4)

    def test_golden_route_y_scores(self, tiny_graph, golden):
        ey1 = next(e for e in tiny_graph.edges if e.edge_id == "EY1")
        ey2 = next(e for e in tiny_graph.edges if e.edge_id == "EY2")
        safety, shore, landmark = score_path([ey1, ey2])
        assert safety   == pytest.approx(golden["route_Y_safety"],   rel=1e-4)
        assert shore    == pytest.approx(golden["route_Y_shore"],     abs=1e-9)
        assert landmark == pytest.approx(golden["route_Y_landmark"],  rel=1e-4)

    def test_single_edge_scores_equal_edge_values(self):
        e = _edge("e1", 4.0, True, 0.7, 0.6)
        safety, shore, landmark = score_path([e])
        assert safety   == pytest.approx(0.7)
        assert shore    == pytest.approx(1.0)
        assert landmark == pytest.approx(0.6)
