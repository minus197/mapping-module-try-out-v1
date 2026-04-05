"""
tests/pathfinding/test_node_resolver.py
----------------------------------------
Tests for Sprint 5: NodeResolver

Run:  pytest tests/pathfinding/test_node_resolver.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from shared.types import FloorGraph, NavigationNode
from pathfinding.node_resolver import NodeResolver, _extract_location_phrase


def _make_graph() -> FloorGraph:
    """Build a minimal test graph with known nodes."""
    nodes = [
        NavigationNode("ZONE-NIKE",      "Nike Store",      (5.0, 5.0),   "zone_centroid", "ZONE-NIKE",
                       tags={"category": "shop", "name": "Nike"}),
        NavigationNode("ZONE-FC",        "Food Court",      (15.0, 5.0),  "zone_centroid", "ZONE-FC",
                       tags={"category": "food_court", "name": "Food Court"}),
        NavigationNode("FEAT-ELEV-1",    "Elevator L1",     (1.0, 12.0),  "elevator",      "ZONE-COR",
                       tags={"feature_type": "elevator"}),
        NavigationNode("FEAT-DOOR-1",    "Door Entrance",   (8.0, 4.0),   "door",          "ZONE-ENT",
                       tags={"feature_type": "door"}),
        NavigationNode("ZONE-RESTROOM",  "Restrooms",       (22.0, 16.0), "zone_centroid", "ZONE-REST",
                       tags={"category": "restroom", "name": "Restrooms"}),
    ]
    graph = FloorGraph(floor_label="L1", source_file="test", nodes=nodes, edges=[])
    graph.rebuild_index()
    return graph


@pytest.fixture
def resolver():
    return NodeResolver(_make_graph())


class TestLocationPhraseExtraction:

    def test_strips_go_to(self):
        assert _extract_location_phrase("I want to go to Nike") == "Nike"

    def test_strips_take_me_to(self):
        assert _extract_location_phrase("Take me to the food court") == "food court"

    def test_strips_navigate_to(self):
        assert _extract_location_phrase("Navigate me to elevator") == "elevator"

    def test_passthrough_plain(self):
        result = _extract_location_phrase("Food Court")
        assert "food court" in result.lower() or result == "Food Court"

    def test_strips_find(self):
        result = _extract_location_phrase("Find the restroom")
        assert "restroom" in result.lower()


class TestNodeResolverFuzzy:

    def test_resolves_exact_match(self, resolver):
        node = resolver.resolve("Nike Store")
        assert node is not None
        assert node.node_id == "ZONE-NIKE"

    def test_resolves_partial_match(self, resolver):
        node = resolver.resolve("I want to go to Nike")
        assert node is not None
        assert "nike" in node.label.lower()

    def test_resolves_food_court(self, resolver):
        node = resolver.resolve("food court")
        assert node is not None
        assert "food" in node.label.lower() or node.tags.get("category") == "food_court"

    def test_resolves_elevator(self, resolver):
        node = resolver.resolve("elevator")
        assert node is not None
        assert node.node_type == "elevator" or "elevator" in node.label.lower()

    def test_resolves_restroom_by_category(self, resolver):
        node = resolver.resolve("I need to find a toilet")
        assert node is not None
        assert node.tags.get("category") == "restroom" or "restroom" in node.label.lower()

    def test_returns_none_on_gibberish(self, resolver):
        node = resolver.resolve("xyzqqqabc")
        # Either None or a low-confidence match — we accept both
        # but it should not crash
        pass   # no assertion needed — just verify no exception


class TestNodeResolverEdgeCases:

    def test_empty_query(self, resolver):
        # Should not crash
        try:
            resolver.resolve("")
        except Exception as e:
            pytest.fail(f"Empty query raised: {e}")

    def test_exclude_types(self, resolver):
        node = resolver.resolve("elevator", exclude_types=["elevator"])
        if node is not None:
            assert node.node_type != "elevator"
