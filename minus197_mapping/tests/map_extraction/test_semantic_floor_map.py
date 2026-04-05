"""
tests/map_extraction/test_semantic_floor_map.py
------------------------------------------------
Tests for Sprint 2: SemanticFloorMapBuilder

Run:  pytest tests/map_extraction/test_semantic_floor_map.py -v
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from map_extraction.ifc_parser import IFCParser
from map_extraction.semantic_floor_map import SemanticFloorMapBuilder

IFC_CONVENIENCE = Path("data/ifc_files/20201022mapping_IFC4_Convenience_store.ifc")
skip_no_ifc = pytest.mark.skipif(
    not IFC_CONVENIENCE.exists(),
    reason="IFC test files not in data/ifc_files/"
)


@pytest.fixture(scope="module")
def sfm():
    pr = IFCParser(IFC_CONVENIENCE).parse()
    return SemanticFloorMapBuilder(pr, floor_label="L1").build()


class TestZones:

    @skip_no_ifc
    def test_zone_count(self, sfm):
        assert len(sfm.zones) == 8

    @skip_no_ifc
    def test_no_zero_area_zones(self, sfm):
        for z in sfm.zones:
            assert z.area > 0

    @skip_no_ifc
    def test_corridor_zone_exists(self, sfm):
        corridors = [z for z in sfm.zones if z.category == "corridor"]
        assert len(corridors) >= 1

    @skip_no_ifc
    def test_zone_polygon_min_vertices(self, sfm):
        for z in sfm.zones:
            assert len(z.boundary_polygon) >= 3


class TestWalls:

    @skip_no_ifc
    def test_wall_count(self, sfm):
        assert len(sfm.walls) == 33

    @skip_no_ifc
    def test_shore_linable_walls_exist(self, sfm):
        shore = [w for w in sfm.walls if w.shore_linable]
        assert len(shore) >= 1, "Expected at least one shore-linable wall near corridor"


class TestFeatures:

    @skip_no_ifc
    def test_all_features_in_bounding_box(self, sfm):
        bb = sfm.bounding_box
        for f in sfm.features:
            px, py = f.position
            assert bb["min_x"] - 1.0 <= px <= bb["max_x"] + 1.0
            assert bb["min_y"] - 1.0 <= py <= bb["max_y"] + 1.0

    @skip_no_ifc
    def test_door_features_have_zone(self, sfm):
        doors = [f for f in sfm.features if f.feature_type == "door"]
        placed = [d for d in doors if d.zone_id is not None]
        # At least half of doors should be inside a zone polygon
        assert len(placed) >= len(doors) // 2


class TestSerialization:

    @skip_no_ifc
    def test_json_round_trip(self, sfm):
        d = sfm.to_dict()
        serialised = json.dumps(d)
        recovered  = json.loads(serialised)
        assert len(recovered["zones"])    == len(sfm.zones)
        assert len(recovered["walls"])    == len(sfm.walls)
        assert len(recovered["features"]) == len(sfm.features)

    @skip_no_ifc
    def test_no_numpy_types_in_output(self, sfm):
        """Ensure all values are plain Python types (no numpy floats/ints)."""
        d = sfm.to_dict()
        text = json.dumps(d)   # would raise TypeError if numpy types present
        assert isinstance(text, str)
