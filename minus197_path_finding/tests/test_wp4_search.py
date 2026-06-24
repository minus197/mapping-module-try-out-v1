"""
tests/test_wp4_search.py  —  M3 gate (WP4)
--------------------------------------------
Unit tests for pathfinding/search.py.

All tests must be green before WP6 starts.

Run:  pytest tests/test_wp4_search.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from pathfinding.cost import CostWeights, compute_edge_costs, normalise_landmark_scores
from pathfinding.search import (
    _edge_set,
    build_nx_graph,
    k_best_paths,
    optimal_path,
    optimal_path_astar,
)


# ── Fixture: tiny_graph with costs already computed ───────────────────────────
@pytest.fixture
def costed_graph(tiny_graph):
    normalise_landmark_scores(tiny_graph, landmark_max=1.0)
    compute_edge_costs(tiny_graph, CostWeights())
    return tiny_graph

@pytest.fixture
def G(costed_graph):
    return build_nx_graph(costed_graph)


# ── build_nx_graph ────────────────────────────────────────────────────────────
class TestBuildNxGraph:
    def test_node_count(self, costed_graph):
        G = build_nx_graph(costed_graph)
        assert G.number_of_nodes() == len(costed_graph.nodes)

    def test_edge_count(self, costed_graph):
        G = build_nx_graph(costed_graph)
        assert G.number_of_edges() == len(costed_graph.edges)

    def test_edge_has_weight(self, costed_graph):
        G = build_nx_graph(costed_graph)
        for u, v, attrs in G.edges(data=True):
            assert "weight" in attrs
            assert attrs["weight"] > 0

    def test_edge_has_data(self, costed_graph):
        G = build_nx_graph(costed_graph)
        for u, v, attrs in G.edges(data=True):
            assert "data" in attrs

    def test_node_has_data(self, costed_graph):
        G = build_nx_graph(costed_graph)
        assert G.nodes["SKE-START"]["data"].node_type == "junction"


# ── optimal_path (Dijkstra) ───────────────────────────────────────────────────
class TestOptimalPath:
    def test_returns_list_of_strings(self, G):
        path = optimal_path(G, "SKE-START", "ZONE-DEST")
        assert isinstance(path, list)
        assert all(isinstance(n, str) for n in path)

    def test_starts_and_ends_correctly(self, G):
        path = optimal_path(G, "SKE-START", "ZONE-DEST")
        assert path[0] == "SKE-START"
        assert path[-1] == "ZONE-DEST"

    def test_missing_start_returns_none(self, G):
        assert optimal_path(G, "NO-SUCH-NODE", "ZONE-DEST") is None

    def test_missing_dest_returns_none(self, G):
        assert optimal_path(G, "SKE-START", "NO-SUCH-NODE") is None

    def test_same_node_returns_singleton(self, G):
        path = optimal_path(G, "SKE-START", "SKE-START")
        assert path == ["SKE-START"]

    def test_chooses_route_y_over_route_x(self, G):
        # Option A proof at the search level: combined_cost prefers Route Y
        path = optimal_path(G, "SKE-START", "ZONE-DEST")
        assert "MID-Y" in path, (
            "Dijkstra should choose Route Y (via MID-Y) because it has lower combined_cost"
        )
        assert "MID-X" not in path


# ── optimal_path_astar (A*) ───────────────────────────────────────────────────
class TestOptimalPathAstar:
    def test_returns_same_as_dijkstra(self, G):
        d_path = optimal_path(G, "SKE-START", "ZONE-DEST")
        a_path = optimal_path_astar(G, "SKE-START", "ZONE-DEST")
        assert d_path == a_path, "A* and Dijkstra must agree on the optimum"

    def test_missing_start_returns_none(self, G):
        assert optimal_path_astar(G, "GHOST", "ZONE-DEST") is None

    def test_missing_dest_returns_none(self, G):
        assert optimal_path_astar(G, "SKE-START", "GHOST") is None


# ── k_best_paths (Yen's) ─────────────────────────────────────────────────────
class TestKBestPaths:
    def test_returns_list(self, G):
        paths = k_best_paths(G, "SKE-START", "ZONE-DEST", k=4)
        assert isinstance(paths, list)

    def test_first_path_matches_dijkstra(self, G):
        dijkstra = optimal_path(G, "SKE-START", "ZONE-DEST")
        best = k_best_paths(G, "SKE-START", "ZONE-DEST", k=4)
        assert best[0] == dijkstra

    def test_has_at_least_two_distinct_paths(self, G):
        # tiny_graph has Route X and Route Y — both must appear
        paths = k_best_paths(G, "SKE-START", "ZONE-DEST", k=4)
        assert len(paths) >= 2, "Expected at least Route Y (winner) and Route X (alt)"

    def test_all_paths_start_and_end_correctly(self, G):
        for path in k_best_paths(G, "SKE-START", "ZONE-DEST", k=4):
            assert path[0] == "SKE-START"
            assert path[-1] == "ZONE-DEST"

    def test_winner_uses_route_y(self, G):
        paths = k_best_paths(G, "SKE-START", "ZONE-DEST", k=4)
        assert "MID-Y" in paths[0]

    def test_route_x_appears_as_alternative(self, G):
        paths = k_best_paths(G, "SKE-START", "ZONE-DEST", k=4)
        node_sets = [set(p) for p in paths]
        assert any("MID-X" in s for s in node_sets), "Route X should appear as an alternative"

    def test_missing_start_returns_empty(self, G):
        assert k_best_paths(G, "GHOST", "ZONE-DEST") == []

    def test_missing_dest_returns_empty(self, G):
        assert k_best_paths(G, "SKE-START", "GHOST") == []

    def test_diversity_filter_removes_near_duplicates(self, G):
        # All returned paths must differ from the winner by more than 20% of edges
        paths = k_best_paths(G, "SKE-START", "ZONE-DEST", k=4)
        if len(paths) < 2:
            pytest.skip("Not enough paths to test diversity")
        winner_edges = _edge_set(paths[0])
        for alt in paths[1:]:
            alt_edges = _edge_set(alt)
            if alt_edges:
                overlap = len(winner_edges & alt_edges) / len(alt_edges)
                assert overlap <= 0.80
