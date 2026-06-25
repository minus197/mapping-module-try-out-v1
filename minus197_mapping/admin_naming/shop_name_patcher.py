"""
admin_naming/shop_name_patcher.py
----------------------------------
Patches the three pipeline output JSON files with admin-supplied shop names.

Given {stem}_shop_names.json it updates:
  - {stem}_sfm.json        zones[].admin_name
  - {stem}_graph.json      nodes[].label, nodes[].tags.admin_label/admin_name
                           (zone_centroid nodes only, matched by zone_id)
  - {stem}_occupancy.json  zones[].admin_name  (if the file has a zones block)

Usage (standalone):
    python -m admin_naming.shop_name_patcher \\
        --names data/outputs/mall_L1_shop_names.json \\
        --output data/outputs/

Called programmatically:
    from admin_naming.shop_name_patcher import run_patcher
    run_patcher(names_path="data/outputs/mall_L1_shop_names.json",
                output_dir="data/outputs/")
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_patcher(
    names_path: str | Path,
    output_dir: str | Path,
) -> None:
    """
    Apply admin names from names_path to the three pipeline output files.

    Parameters
    ----------
    names_path : path to {stem}_shop_names.json
    output_dir : directory containing the three pipeline output files
    """
    names_path = Path(names_path)
    output_dir = Path(output_dir)

    names_data  = json.loads(names_path.read_text(encoding="utf-8"))
    stem        = names_data.get("source_stem", "")
    mappings    = names_data.get("mappings", [])

    if not mappings:
        print("[AdminNaming] shop_names file has no mappings — nothing to patch.")
        return

    # Build lookup: zone_id → admin_name
    name_map: Dict[str, str] = {
        m["zone_id"]: m["admin_name"]
        for m in mappings
        if m.get("admin_name", "").strip()
    }

    patched: List[str] = []

    sfm_path = output_dir / f"{stem}_sfm.json"
    if sfm_path.exists():
        _patch_sfm(sfm_path, name_map)
        patched.append(sfm_path.name)
    else:
        print(f"[AdminNaming] WARNING: {sfm_path} not found — skipped.")

    graph_path = output_dir / f"{stem}_graph.json"
    if graph_path.exists():
        _patch_graph(graph_path, name_map)
        patched.append(graph_path.name)
    else:
        print(f"[AdminNaming] WARNING: {graph_path} not found — skipped.")

    occ_path = output_dir / f"{stem}_occupancy.json"
    if occ_path.exists():
        _patch_occupancy(occ_path, name_map)
        patched.append(occ_path.name)
    else:
        print(f"[AdminNaming] WARNING: {occ_path} not found — skipped.")

    if patched:
        print(f" Patched {', '.join(patched)}")


# ---------------------------------------------------------------------------
# Per-file patchers
# ---------------------------------------------------------------------------

def _patch_sfm(path: Path, name_map: Dict[str, str]) -> None:
    data   = json.loads(path.read_text(encoding="utf-8"))
    count  = 0
    for zone in data.get("zones", []):
        zone_id = zone.get("zone_id", "")
        if zone_id in name_map:
            zone["admin_name"] = name_map[zone_id]
            count += 1
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"[AdminNaming] _sfm.json: {count} zone(s) updated.")


def _patch_graph(path: Path, name_map: Dict[str, str]) -> None:
    data  = json.loads(path.read_text(encoding="utf-8"))
    count = 0
    for node in data.get("nodes", []):
        if node.get("node_type") != "zone_centroid":
            continue
        zone_id = node.get("zone_id", "")
        if zone_id in name_map:
            admin_name = name_map[zone_id]
            node["label"] = admin_name
            tags = node.setdefault("tags", {})
            tags["admin_label"] = admin_name
            tags["admin_name"]  = admin_name
            count += 1
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"[AdminNaming] _graph.json: {count} node(s) updated.")


def _patch_occupancy(path: Path, name_map: Dict[str, str]) -> None:
    data  = json.loads(path.read_text(encoding="utf-8"))
    count = 0
    # Occupancy grid may carry a top-level "zones" metadata block
    for zone in data.get("zones", []):
        zone_id = zone.get("zone_id", "")
        if zone_id in name_map:
            zone["admin_name"] = name_map[zone_id]
            count += 1
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"[AdminNaming] _occupancy.json: {count} zone(s) updated.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch pipeline JSON outputs with admin shop names"
    )
    parser.add_argument("--names",  required=True,
                        help="Path to {stem}_shop_names.json")
    parser.add_argument("--output", required=True,
                        help="Directory containing the pipeline output files")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    run_patcher(names_path=args.names, output_dir=args.output)


if __name__ == "__main__":
    main()
