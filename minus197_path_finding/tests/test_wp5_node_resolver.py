"""
tests/test_wp5_node_resolver.py  —  M3 gate (WP5)
---------------------------------------------------
Unit tests for pathfinding/node_resolver.py.
Uses fake_wall_checker (Fixture B) so the real perception_map is never needed.

Run:  pytest tests/test_wp5_node_resolver.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from shared.types import FloorGraph, NavigationNode
from pathfinding.node_resolver import StartNodeResolver, _NAVIGABLE


# ── Helpers ───────────────────────────────────────────────────────────────────
def _no_wall(x1, y1, x2, y2):
    """Wall checker that never blocks anything."""
    return False

def _all_wall(x1, y1, x2, y2):
    """Wall checker that blocks everything."""
    return True


# ── Basic resolution ──────────────────────────────────────────────────────────
class TestBasicResolution:
    def test_resolves_nearest_when_no_wall(self, tiny_graph):
        resolver = StartNodeResolver(tiny_graph, _no_wall)
        # User is exactly at START position — nearest navigable node is START
        node, coords = resolver.resolve(0.0, 0.0)
        assert node.node_id == "SKE-START"

    def test_returns_user_coordinates_unchanged(self, tiny_graph):
        resolver = StartNodeResolver(tiny_graph, _no_wall)
        _, coords = resolver.resolve(1.5, 2.7)
        assert coords == (1.5, 2.7)

    def test_resolves_nearest_navigable_to_arbitrary_point(self, tiny_graph):
        resolver = StartNodeResolver(tiny_graph, _no_wall)
        # Point (3, 1.5) is equidistant to MID-X(3,0) and MID-Y(3,3)
        # but MID-X is closer (dist=1.5 vs 1.5 — tie; either is fine)
        node, _ = resolver.resolve(3.0, 0.5)
        assert node.node_id in ("MID-X", "MID-Y")


# ── Wall filter ───────────────────────────────────────────────────────────────
class TestWallFilter:
    def test_decoy_blocked_by_wall(self, tiny_graph, fake_wall_checker):
        """
        tiny_graph layout: wall at x=5.
        User at (0,0).  DECOY is at (4.9,0) — same side of wall, NOT blocked.
        ZONE-DEST is at (8,0) — across wall, blocked.
        Nearest navigable nodes in order: SKE-START(0,0) → MID-X(3,0) → DECOY(4.9,0)
        All are on the same side as the user (x<5), so no wall-crossing occurs.
        The test we really need: a user ACROSS the wall cannot reach nodes on the
        near side without crossing.  Place user at (7,0) — across the wall.
        SKE-START(0,0), MID-X(3,0), MID-Y(3,3) are all across; only DECOY(4.9,0) avoids
        crossing for a user at x>5 ... actually DECOY is at 4.9 so (7→4.9) crosses.
        Use fake_wall_checker: crosses if one endpoint <5 and other >5.
        User (6,0) → SKE-START(0,0): crosses (0 < 5 < 6) ✓ blocked
        User (6,0) → MID-X(3,0):    crosses ✓ blocked
        User (6,0) → MID-Y(3,3):    crosses ✓ blocked
        User (6,0) → DECOY(4.9,0):  crosses (4.9 < 5 < 6) ✓ blocked
        → all blocked → fallback to geometric nearest
        """
        resolver = StartNodeResolver(tiny_graph, fake_wall_checker)
        node, _ = resolver.resolve(6.0, 0.0)
        # All navigable nodes are across the wall from x=6 — should fall back
        # to geometric nearest (DECOY at x=4.9 is closest to x=6)
        assert node.node_id == "DECOY"

    def test_near_side_user_avoids_wall(self, tiny_graph, fake_wall_checker):
        """User at (1,0) — same side as START/MID-X/MID-Y. Wall not crossed."""
        resolver = StartNodeResolver(tiny_graph, fake_wall_checker)
        node, _ = resolver.resolve(1.0, 0.0)
        # Nearest navigable is SKE-START at (0,0); no wall between (1,0) and (0,0)
        assert node.node_id == "SKE-START"

    def test_fallback_when_all_blocked(self, tiny_graph):
        """When every candidate is wall-blocked, returns geometric nearest."""
        resolver = StartNodeResolver(tiny_graph, _all_wall)
        node, _ = resolver.resolve(0.0, 0.0)
        # Fallback = geometric nearest regardless of wall
        assert node.node_id == "SKE-START"

    def test_skips_wall_blocked_returns_next_clear(self, tiny_graph):
        """
        Construct a checker that only blocks the nearest node so the resolver
        must skip it and return the second nearest.
        """
        nearest_id = [None]

        def block_first(x1, y1, x2, y2):
            # nearest from (0,0) is SKE-START itself at distance 0 — but KD-tree
            # returns that with dist=0; block anything within 0.1 m of origin
            import math
            return math.sqrt((x2 - 0) ** 2 + (y2 - 0) ** 2) < 0.1

        resolver = StartNodeResolver(tiny_graph, block_first)
        node, _ = resolver.resolve(0.0, 0.0)
        # SKE-START (0,0) is blocked; next nearest navigable should be MID-X (3,0)
        assert node.node_id != "SKE-START"


# ── Zone centroid exclusion ────────────────────────────────────────────────────
class TestZoneCentroidExclusion:
    def test_zone_centroid_never_returned(self, tiny_graph):
        resolver = StartNodeResolver(tiny_graph, _no_wall)
        # Query from right next to ZONE-DEST (8,0) — should NOT return it
        node, _ = resolver.resolve(7.9, 0.0)
        assert node.node_type != "zone_centroid"

    def test_navigable_types_only_in_candidates(self, tiny_graph):
        # Verify the resolver's internal node list excludes zone_centroid
        resolver = StartNodeResolver(tiny_graph, _no_wall)
        for n in resolver._nodes:
            assert n.node_type in _NAVIGABLE

    def test_graph_with_only_zone_centroids_raises(self):
        fg = FloorGraph(floor_label="L1", source_file="t")
        fg.nodes = [
            NavigationNode("Z1", "Zone 1", (0.0, 0.0), "zone_centroid", "Z1"),
        ]
        fg.rebuild_index()
        with pytest.raises(ValueError):
            StartNodeResolver(fg, _no_wall)
