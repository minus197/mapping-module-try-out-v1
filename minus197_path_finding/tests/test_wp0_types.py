"""
tests/test_wp0_types.py  —  M1 gate
-------------------------------------
Unit tests for shared/types.py (WP0).
All tests must be green before WP1 starts.

Run:  pytest tests/test_wp0_types.py -v
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from shared.types import (
    FloorGraph, NavigationEdge, NavigationNode, PathResult, PathStep,
)


# ── Sample _graph.json block (mirrors the schema from map_extraction) ─────────
SAMPLE_JSON = {
    "floor_label": "L1",
    "source_file": "test_building.ifc",
    "spatial_meta": {"origin_x": 0.0, "origin_y": 0.0},
    "nodes": [
        {
            "node_id": "SKE-0001",
            "label": "Junction A",
            "position": [1.0, 2.0],
            "node_type": "junction",
            "zone_id": None,
            "tags": {"floor_label": "L1"},
        },
        {
            "node_id": "ZONE-FOOD",
            "label": "Food Court",
            "position": [10.0, 5.0],
            "node_type": "zone_centroid",
            "zone_id": "Z42",
            "tags": {"category": "food_court", "floor_label": "L1"},
        },
    ],
    "edges": [
        {
            "edge_id": "EDGE-001",
            "source_id": "SKE-0001",
            "target_id": "ZONE-FOOD",
            "distance": 9.43,
            "shore_linable": True,
            "safety_score": 0.75,
            "landmark_score": 0.5,
            "tags": {"edge_type": "intra_floor"},
        },
    ],
}


class TestFromDict:
    def test_node_count(self):
        fg = FloorGraph.from_dict(SAMPLE_JSON)
        assert len(fg.nodes) == 2

    def test_edge_count(self):
        fg = FloorGraph.from_dict(SAMPLE_JSON)
        assert len(fg.edges) == 1

    def test_floor_label(self):
        fg = FloorGraph.from_dict(SAMPLE_JSON)
        assert fg.floor_label == "L1"

    def test_node_fields(self):
        fg = FloorGraph.from_dict(SAMPLE_JSON)
        n = fg.node("SKE-0001")
        assert n is not None
        assert n.label == "Junction A"
        assert n.position == (1.0, 2.0)
        assert n.node_type == "junction"
        assert n.zone_id is None
        assert n.tags["floor_label"] == "L1"

    def test_edge_fields(self):
        fg = FloorGraph.from_dict(SAMPLE_JSON)
        e = fg.edges[0]
        assert e.edge_id == "EDGE-001"
        assert e.distance == pytest.approx(9.43)
        assert e.shore_linable is True
        assert e.safety_score == pytest.approx(0.75)
        assert e.landmark_score == pytest.approx(0.5)

    def test_combined_cost_defaults_to_zero(self):
        fg = FloorGraph.from_dict(SAMPLE_JSON)
        assert fg.edges[0].combined_cost == 0.0

    def test_node_index_built(self):
        fg = FloorGraph.from_dict(SAMPLE_JSON)
        assert fg.node("ZONE-FOOD") is not None
        assert fg.node("DOES-NOT-EXIST") is None

    def test_missing_spatial_meta_tolerated(self):
        data = dict(SAMPLE_JSON)
        del data["spatial_meta"]
        fg = FloorGraph.from_dict(data)
        assert fg.spatial_meta == {}

    def test_empty_nodes_and_edges(self):
        fg = FloorGraph.from_dict({"floor_label": "L0", "source_file": "x.ifc"})
        assert fg.nodes == []
        assert fg.edges == []


class TestDataclassDefaults:
    def test_navigation_node_tags_default(self):
        n = NavigationNode("N1", "Node 1", (0.0, 0.0), "junction", None)
        assert n.tags == {}

    def test_navigation_edge_combined_cost_default(self):
        e = NavigationEdge("E1", "A", "B", 5.0, False, 0.5, 0.5)
        assert e.combined_cost == 0.0

    def test_navigation_edge_tags_default(self):
        e = NavigationEdge("E1", "A", "B", 5.0, False, 0.5, 0.5)
        assert e.tags == {}

    def test_path_result_defaults(self):
        r = PathResult()
        assert r.found is False
        assert r.steps == []
        assert r.alternatives == []
        assert r.total_cost == 0.0


class TestFixtures:
    """Confirm that the test fixtures themselves are consistent."""

    def test_tiny_graph_node_count(self, tiny_graph):
        assert len(tiny_graph.nodes) == 5

    def test_tiny_graph_edge_count(self, tiny_graph):
        assert len(tiny_graph.edges) == 5

    def test_tiny_graph_index(self, tiny_graph):
        assert tiny_graph.node("SKE-START") is not None
        assert tiny_graph.node("ZONE-DEST") is not None
        assert tiny_graph.node("DECOY") is not None

    def test_golden_route_y_cheaper_combined(self, golden):
        # Core Option A claim: Route Y (higher quality) wins on combined_cost
        assert golden["route_Y_combined_cost"] < golden["route_X_combined_cost"]

    def test_golden_route_x_shorter_distance(self, golden):
        # Route X is physically shorter — the naive shortest-path picks it
        assert golden["route_X_distance"] < golden["route_Y_distance"]

    def test_fake_wall_checker_blocks_across_wall(self, fake_wall_checker):
        # Segment from (0,0) to (8,0) crosses the wall at x=5
        assert fake_wall_checker(0, 0, 8, 0) is True

    def test_fake_wall_checker_allows_same_side(self, fake_wall_checker):
        # Both endpoints left of x=5 — no crossing
        assert fake_wall_checker(0, 0, 4, 0) is False

    def test_fake_wall_checker_decoy_blocked(self, fake_wall_checker):
        # DECOY is at x=4.9 — segment START(0,0)→DECOY(4.9,0) does NOT cross x=5
        assert fake_wall_checker(0, 0, 4.9, 0) is False

    def test_fake_wall_checker_start_to_dest_blocked(self, fake_wall_checker):
        # Direct line START(0,0)→DEST(8,0) crosses the wall
        assert fake_wall_checker(0, 0, 8, 0) is True
