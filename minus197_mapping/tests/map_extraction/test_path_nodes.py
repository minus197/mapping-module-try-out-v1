"""
tests/map_extraction/test_path_nodes.py
-----------------------------------------
Tests for PathNodeBuilder — cane-trailing path nodes placed along the
corridor-facing side of walls.

Run:  pytest tests/map_extraction/test_path_nodes.py -v
"""

import math
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from map_extraction.ifc_parser import IFCParser
from map_extraction.semantic_floor_map import SemanticFloorMapBuilder
from map_extraction.path_nodes import (
    PathNodeBuilder,
    _spacing_positions,
    PATH_GAP_M,
    SPACING_MIN,
    SPACING_MAX,
    CIRCULATION_CATEGORIES,
)

IFC_CONVENIENCE = Path("data/ifc_files/20201022mapping IFC4 Convenience store.ifc")
skip_no_ifc = pytest.mark.skipif(
    not IFC_CONVENIENCE.exists(),
    reason="IFC test files not in data/ifc_files/"
)


# ── Pure unit tests: spacing helper (no IFC needed) ──────────────────────────

class TestSpacingPositions:

    @pytest.mark.parametrize("seg_len", [
        5.5, 6.33, 7.11, 8.5, 9.28, 10.0, 10.67, 11.0, 12.14,
        13.67, 16.30, 20.0, 21.81, 26.77, 37.83, 73.4, 110.0,
    ])
    def test_inter_node_gaps_in_band(self, seg_len):
        """Every gap between consecutive nodes on one wall is in [5.5, 6.0]."""
        pos = _spacing_positions(seg_len, SPACING_MAX)
        gaps = [b - a for a, b in zip(pos, pos[1:])]
        for g in gaps:
            assert SPACING_MIN - 1e-6 <= g <= SPACING_MAX + 1e-6, \
                f"gap {g:.3f} out of band for wall len {seg_len}"

    @pytest.mark.parametrize("seg_len", [
        5.5, 9.28, 10.0, 16.30, 21.81, 26.77, 37.83, 73.4,
    ])
    def test_nodes_stay_on_wall(self, seg_len):
        """No node falls before the start or past the end of the wall."""
        pos = _spacing_positions(seg_len, SPACING_MAX)
        assert pos[0] >= -1e-6
        assert pos[-1] <= seg_len + 1e-6

    def test_short_wall_single_midpoint(self):
        """A wall shorter than the min spacing gets one node at its midpoint."""
        pos = _spacing_positions(4.0, SPACING_MAX)
        assert pos == [2.0]


# ── Integration tests on a real IFC file ─────────────────────────────────────

@pytest.fixture(scope="module")
def sfm():
    pr = IFCParser(IFC_CONVENIENCE).parse()
    return SemanticFloorMapBuilder(pr, floor_label="L1").build()


@pytest.fixture(scope="module")
def built(sfm):
    return PathNodeBuilder(sfm).build()


@pytest.fixture(scope="module")
def corridor_union(sfm):
    polys = [
        Polygon(z.boundary_polygon).buffer(0)
        for z in sfm.zones
        if z.category in CIRCULATION_CATEGORIES and len(z.boundary_polygon) >= 3
    ]
    return unary_union(polys) if polys else None


class TestPathNodes:

    @skip_no_ifc
    def test_nodes_produced(self, built):
        assert len(built.nodes) > 0, "Expected path nodes along corridor walls"

    @skip_no_ifc
    def test_gap_to_wall_is_030(self, built, sfm):
        """Every node sits PATH_GAP_M from its wall centreline."""
        wall_lines = defaultdict(list)
        for w in sfm.walls:
            wall_lines[w.wall_id].append(LineString([w.start, w.end]))
        for n in built.nodes:
            d = min(l.distance(Point(n.position)) for l in wall_lines[n.wall_id])
            assert abs(d - PATH_GAP_M) < 1e-3, \
                f"node {n.node_id} gap {d:.3f} != {PATH_GAP_M}"

    @skip_no_ifc
    def test_nodes_on_corridor_side(self, built, corridor_union):
        """Every node lies inside the corridor (the corridor-facing side)."""
        if corridor_union is None:
            pytest.skip("no corridor zone in this IFC")
        cu = corridor_union.buffer(PATH_GAP_M + 0.05)
        for n in built.nodes:
            assert cu.contains(Point(n.position)), \
                f"node {n.node_id} not on corridor-facing side"

    @skip_no_ifc
    def test_no_nodes_in_shop_interior(self, built, sfm):
        """No node lands inside a non-circulation (shop/room) zone interior."""
        shops = [
            Polygon(z.boundary_polygon).buffer(0)
            for z in sfm.zones
            if z.category not in CIRCULATION_CATEGORIES
            and len(z.boundary_polygon) >= 3
        ]
        if not shops:
            pytest.skip("no shop zones in this IFC")
        su = unary_union(shops)
        for n in built.nodes:
            assert not su.contains(Point(n.position)), \
                f"node {n.node_id} inside a shop interior"

    @skip_no_ifc
    def test_output_has_no_edges(self, built):
        """The serialised layer carries only nodes — no edges."""
        d = built._to_dict()
        assert "edges" not in d
        assert "path_nodes" in d

    @skip_no_ifc
    def test_node_ids_contiguous(self, built):
        ids = [n.node_id for n in built.nodes]
        assert ids == [f"PATH-{i:04d}" for i in range(len(ids))]
