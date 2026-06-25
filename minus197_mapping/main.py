"""
main.py
-------
Entry point for the Minus197 Mapping Module.

Single-floor:
    python main.py --ifc data/ifc_files/mall_L1.ifc --floor L1

Single-floor with destination query:
    python main.py --ifc data/ifc_files/mall_L1.ifc --floor L1 --query "food court"

Multi-floor (one IFC per floor, floors in order bottom→top):
    python main.py \\
        --ifc data/ifc_files/mall_L1.ifc --floor L1 \\
        --ifc data/ifc_files/mall_L2.ifc --floor L2 \\
        --ifc data/ifc_files/mall_L3.ifc --floor L3 \\
        --building "One Galle Face Mall"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from map_extraction import MapExtractionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Minus197 — Map Extraction + Pathfinding"
    )
    parser.add_argument("--ifc",      action="append", required=True,
                        help="Path to .ifc file (repeat for multiple floors)")
    parser.add_argument("--floor",    action="append",
                        help="Floor label e.g. L1  (one per --ifc)")
    parser.add_argument("--building", default="Building",
                        help="Building name for multi-floor output")
    parser.add_argument("--query",    default=None,
                        help="(Reserved) Destination query — handled by minus197_path_finding engine")
    parser.add_argument("--no-save",  action="store_true",
                        help="Do not save output JSON files")
    args = parser.parse_args()

    ifc_paths   = args.ifc
    floor_labels = args.floor or [f"L{i+1}" for i in range(len(ifc_paths))]

    if len(floor_labels) != len(ifc_paths):
        print(f"ERROR: {len(ifc_paths)} --ifc paths but "
              f"{len(floor_labels)} --floor labels. "
              f"Supply one --floor per --ifc.")
        sys.exit(1)

    # ── Single-floor ──────────────────────────────────────────────────────────
    if len(ifc_paths) == 1:
        pipeline = MapExtractionPipeline(
            ifc_path    = ifc_paths[0],
            floor_label = floor_labels[0],
        )
        graph = pipeline.run()

        if not args.no_save:
            pipeline.save("data/outputs/")

        if args.query:
            print("\n[Info] Pathfinding is handled by the minus197_path_finding module.")
            print(f"       Load the saved graph and run PathfindingEngine from there.")

    # ── Multi-floor ───────────────────────────────────────────────────────────
    else:
        floors_spec = [
            (ifc_paths[i], floor_labels[i], {})
            for i in range(len(ifc_paths))
        ]
        pipeline = MapExtractionPipeline.multi_floor(
            floors        = floors_spec,
            building_name = args.building,
        )
        building = pipeline.run_multi()

        if not args.no_save:
            pipeline.save_multi("data/outputs/")


if __name__ == "__main__":
    main()
