"""
tests/admin_naming/test_shop_name_gui.py
----------------------------------------
Unit tests for the *pure* (non-Tk) logic behind the graphical naming UI, plus
the patcher's new admin_description handling. No display / Tk is required.

Run:  pytest tests/admin_naming/test_shop_name_gui.py -v
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from admin_naming.shop_name_gui import (
    load_zones,
    hit_test,
    load_existing_names,
    save_names,
)
from admin_naming.shop_name_patcher import run_patcher


# A synthetic SFM: a small zone fully inside a big one, plus a polygon-less zone.
SFM = {
    "meta": {"floor_label": "L1", "source_file": "data/ifc_files/demo.ifc"},
    "bounding_box": {"min_x": 0, "min_y": 0, "max_x": 10, "max_y": 10},
    "zones": [
        {
            "zone_id": "OUTER", "name": "401", "category": "corridor",
            "area_m2": 100.0, "centroid": [5, 5],
            "boundary_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
        },
        {
            "zone_id": "INNER", "name": "402", "category": "unknown",
            "area_m2": 4.0, "centroid": [5, 5],
            "boundary_polygon": [[4, 4], [6, 4], [6, 6], [4, 6]],
        },
        {
            "zone_id": "NOPOLY", "name": "403", "category": "shop",
            "area_m2": 12.0, "centroid": [9, 9], "boundary_polygon": None,
        },
    ],
}


class TestLoadZones:

    def test_builds_view_models(self):
        zones = load_zones(SFM)
        assert len(zones) == 3
        by_id = {z["zone_id"]: z for z in zones}
        assert by_id["OUTER"]["ifc_name"] == "401"
        assert by_id["OUTER"]["path"] is not None
        # A zone with no polygon is kept but has no hit-test path.
        assert by_id["NOPOLY"]["path"] is None


class TestHitTest:

    def test_overlap_prefers_smallest_area(self):
        zones = load_zones(SFM)
        hit = hit_test(5, 5, zones)          # inside both OUTER and INNER
        assert hit is not None
        assert hit["zone_id"] == "INNER"

    def test_outer_only(self):
        zones = load_zones(SFM)
        hit = hit_test(1, 1, zones)          # inside OUTER only
        assert hit["zone_id"] == "OUTER"

    def test_miss_returns_none(self):
        zones = load_zones(SFM)
        assert hit_test(50, 50, zones) is None


class TestExistingNames:

    def test_reads_cli_and_gui_files(self, tmp_path):
        # CLI-style entry (no admin_description) + GUI-style entry.
        p = tmp_path / "demo_shop_names.json"
        p.write_text(json.dumps({"mappings": [
            {"zone_id": "OUTER", "ifc_name": "401", "admin_name": "Lobby"},
            {"zone_id": "INNER", "ifc_name": "402",
             "admin_name": "Kiosk", "admin_description": "Newsstand"},
        ]}), encoding="utf-8")
        existing = load_existing_names(p)
        assert existing["OUTER"]["admin_name"] == "Lobby"
        assert existing["OUTER"]["admin_description"] == ""   # tolerated
        assert existing["INNER"]["admin_description"] == "Newsstand"

    def test_missing_file(self, tmp_path):
        assert load_existing_names(tmp_path / "nope.json") == {}


class TestSaveNames:

    def test_schema_and_skip_empty(self, tmp_path):
        p = tmp_path / "demo_shop_names.json"
        mappings = {
            "INNER": {"ifc_name": "402", "admin_name": "Kiosk",
                      "admin_description": "Newsstand near door"},
            "OUTER": {"ifc_name": "401", "admin_name": "", "admin_description": ""},
        }
        records = save_names(p, "demo", "L1", mappings)
        # Empty entry skipped; written entry has all four keys.
        assert len(records) == 1
        rec = records[0]
        assert set(rec) == {"zone_id", "ifc_name", "admin_name", "admin_description"}
        assert rec["admin_description"] == "Newsstand near door"

        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["source_stem"] == "demo"
        assert data["floor_label"] == "L1"
        assert data["mappings"][0]["zone_id"] == "INNER"


class TestPatcherDescription:

    def test_patches_description_into_outputs(self, tmp_path):
        stem = "demo"
        (tmp_path / f"{stem}_sfm.json").write_text(json.dumps({
            "zones": [{"zone_id": "INNER", "name": "402", "category": "unknown"}]
        }), encoding="utf-8")
        (tmp_path / f"{stem}_graph.json").write_text(json.dumps({
            "nodes": [{"node_id": "ZONE-INNER", "node_type": "zone_centroid",
                       "zone_id": "INNER", "label": "402", "tags": {}}]
        }), encoding="utf-8")
        names_path = tmp_path / f"{stem}_shop_names.json"
        names_path.write_text(json.dumps({
            "source_stem": stem,
            "mappings": [{"zone_id": "INNER", "ifc_name": "402",
                          "admin_name": "Kiosk",
                          "admin_description": "Newsstand near door"}],
        }), encoding="utf-8")

        run_patcher(names_path=names_path, output_dir=tmp_path)

        sfm = json.loads((tmp_path / f"{stem}_sfm.json").read_text(encoding="utf-8"))
        assert sfm["zones"][0]["admin_name"] == "Kiosk"
        assert sfm["zones"][0]["admin_description"] == "Newsstand near door"

        graph = json.loads((tmp_path / f"{stem}_graph.json").read_text(encoding="utf-8"))
        tags = graph["nodes"][0]["tags"]
        assert tags["admin_label"] == "Kiosk"        # title path unchanged
        assert tags["admin_name"] == "Kiosk"
        assert tags["admin_description"] == "Newsstand near door"
        assert graph["nodes"][0]["label"] == "Kiosk"
