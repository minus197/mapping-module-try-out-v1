"""
tests/map_extraction/test_ifc_parser.py
----------------------------------------
Tests for Sprint 1: IFCParser

Run:  pytest tests/map_extraction/test_ifc_parser.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from map_extraction.ifc_parser import IFCParser, IFCParseResult

IFC_CONVENIENCE = Path("data/ifc_files/20201022mapping_IFC4_Convenience_store.ifc")
IFC_SIMPLE      = Path("data/ifc_files/ifc_simple_house.ifc")

# Skip if test files are not present
skip_no_ifc = pytest.mark.skipif(
    not IFC_CONVENIENCE.exists(),
    reason="IFC test files not in data/ifc_files/ — copy them there to run"
)


class TestIFCParserConvenienceStore:

    @pytest.fixture(scope="class")
    def result(self):
        return IFCParser(IFC_CONVENIENCE).parse()

    @skip_no_ifc
    def test_schema_detected(self, result):
        assert result.schema == "IFC4"

    @skip_no_ifc
    def test_unit_scale_millimetre(self, result):
        assert abs(result.unit_scale - 0.001) < 1e-9

    @skip_no_ifc
    def test_spaces_extracted(self, result):
        assert len(result.spaces) == 8

    @skip_no_ifc
    def test_all_spaces_have_polygon(self, result):
        for s in result.spaces:
            assert len(s.polygon) >= 3, f"Space {s.name} has no polygon"

    @skip_no_ifc
    def test_all_spaces_have_positive_area(self, result):
        for s in result.spaces:
            assert s.area > 0, f"Space {s.name} has zero area"

    @skip_no_ifc
    def test_walls_extracted(self, result):
        assert len(result.walls) == 33

    @skip_no_ifc
    def test_walls_have_nonzero_length(self, result):
        nonzero = [w for w in result.walls if w.length > 0.01]
        assert len(nonzero) == len(result.walls)

    @skip_no_ifc
    def test_doors_extracted(self, result):
        doors = [f for f in result.features if f.feature_type == "door"]
        assert len(doors) == 12

    @skip_no_ifc
    def test_coordinates_in_metres(self, result):
        # All space centroids should be within a 100 m bounding box
        for s in result.spaces:
            cx, cy = s.centroid
            assert -10 <= cx <= 100, f"Centroid x={cx} looks wrong (check mm→m conversion)"
            assert -10 <= cy <= 100, f"Centroid y={cy} looks wrong"


class TestIFCParserSimpleHouse:

    @pytest.fixture(scope="class")
    def result(self):
        return IFCParser(IFC_SIMPLE).parse()

    @skip_no_ifc
    def test_spaces_extracted(self, result):
        assert len(result.spaces) >= 4

    @skip_no_ifc
    def test_furnishings_classified(self, result):
        benches    = [f for f in result.features if f.feature_type == "bench"]
        info_desks = [f for f in result.features if f.feature_type == "info_desk"]
        assert len(benches) + len(info_desks) > 0
