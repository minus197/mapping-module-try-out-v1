"""
tests/pathfinding/test_scorer.py
---------------------------------
Tests for Sprint 6: PathScorer

Run:  pytest tests/pathfinding/test_scorer.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from shared.types import NavigationEdge
from pathfinding.scorer import score_path, select_best, ScoredPath


def _edge(eid, dist, shore, safety, landmark) -> NavigationEdge:
    return NavigationEdge(
        edge_id=eid, source_id="A", target_id="B",
        distance=dist, shore_linable=shore,
        safety_score=safety, landmark_score=landmark,
    )


class TestScorePath:

    def test_empty_path_returns_zero(self):
        sp = score_path([])
        assert sp.composite == 0.0
        assert sp.total_distance == 0.0

    def test_all_shore_gets_high_shore_score(self):
        edges = [_edge("e1", 5.0, True, 0.8, 0.5),
                 _edge("e2", 5.0, True, 0.8, 0.5)]
        sp = score_path(edges)
        assert sp.shore_score == pytest.approx(1.0)

    def test_no_shore_gets_zero_shore_score(self):
        edges = [_edge("e1", 5.0, False, 0.8, 0.5)]
        sp = score_path(edges)
        assert sp.shore_score == pytest.approx(0.0)

    def test_composite_in_range(self):
        edges = [_edge("e1", 3.0, True, 0.9, 0.7)]
        sp = score_path(edges)
        assert 0.0 <= sp.composite <= 1.0

    def test_distance_weighted_safety(self):
        # Long edge has lower safety — should pull composite down more
        edges = [
            _edge("e1", 1.0, False, 1.0, 0.0),
            _edge("e2", 9.0, False, 0.0, 0.0),
        ]
        sp = score_path(edges)
        assert sp.safety_score < 0.2   # weighted toward low-safety long edge

    def test_total_distance_correct(self):
        edges = [_edge("e1", 3.5, False, 0.5, 0.0),
                 _edge("e2", 6.5, False, 0.5, 0.0)]
        sp = score_path(edges)
        assert sp.total_distance == pytest.approx(10.0)


class TestSelectBest:

    def test_single_path_is_selected(self):
        sp = ScoredPath([], 10.0, 0.8, 0.8, 0.8, 0.8)
        best, alts = select_best([sp])
        assert best is sp
        assert alts == []

    def test_selects_shortest_among_top_tier(self):
        long_good  = ScoredPath([], 20.0, 0.9, 0.9, 0.9, 0.9)
        short_good = ScoredPath([], 10.0, 0.88, 0.88, 0.88, 0.88)
        bad_short  = ScoredPath([], 5.0,  0.1, 0.1, 0.1, 0.1)
        best, alts = select_best([long_good, short_good, bad_short])
        assert best.total_distance == pytest.approx(10.0)

    def test_alternatives_do_not_include_best(self):
        paths = [
            ScoredPath([], float(i), 0.8, 0.8, 0.8, 0.8)
            for i in range(1, 6)
        ]
        best, alts = select_best(paths)
        assert best not in alts

    def test_raises_on_empty_input(self):
        with pytest.raises(ValueError):
            select_best([])
