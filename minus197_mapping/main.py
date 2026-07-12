"""
main.py
-------
Entry point for the Minus197 Mapping Module.

Single-floor:
    python main.py --ifc data/ifc_files/mall_L1.ifc --floor L1

Single-floor with destination query:
    python main.py --ifc data/ifc_files/mall_L1.ifc --floor L1 --query "food court"

Single-floor with admin shop naming (text CLI):
    python main.py --ifc data/ifc_files/mall_L1.ifc --floor L1 --admin-name

Single-floor with admin shop naming (click-to-name GUI):
    python main.py --ifc data/ifc_files/mall_L1.ifc --floor L1 --admin-gui

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
    parser.add_argument("--ifc",        action="append", required=True,
                        help="Path to .ifc file (repeat for multiple floors)")
    parser.add_argument("--floor",      action="append",
                        help="Floor label e.g. L1  (one per --ifc)")
    parser.add_argument("--building",   default="Building",
                        help="Building name for multi-floor output")
    parser.add_argument("--query",      default=None,
                        help="(Reserved) Destination query — handled by minus197_path_finding engine")
    parser.add_argument("--no-save",    action="store_true",
                        help="Do not save output JSON files")
    parser.add_argument("--admin-name", action="store_true",
                        help="Launch the text-based shop-naming CLI after saving outputs")
    parser.add_argument("--admin-gui",  action="store_true",
                        help="Launch the graphical click-to-name UI after saving outputs")
    args = parser.parse_args()

    ifc_paths    = args.ifc
    floor_labels = args.floor or [f"L{i+1}" for i in range(len(ifc_paths))]

    if len(floor_labels) != len(ifc_paths):
        print(f"ERROR: {len(ifc_paths)} --ifc paths but "
              f"{len(floor_labels)} --floor labels. "
              f"Supply one --floor per --ifc.")
        sys.exit(1)

    OUTPUT_DIR = "data/outputs/"

    # ── Single-floor ──────────────────────────────────────────────────────────
    if len(ifc_paths) == 1:
        pipeline = MapExtractionPipeline(
            ifc_path    = ifc_paths[0],
            floor_label = floor_labels[0],
        )
        graph = pipeline.run()

        if not args.no_save:
            pipeline.save(OUTPUT_DIR)

        if args.query:
            print("\n[Info] Pathfinding is handled by the minus197_path_finding module.")
            print(f"       Load the saved graph and run PathfindingEngine from there.")

        if args.admin_name and not args.no_save:
            stem     = Path(ifc_paths[0]).stem
            sfm_path = Path(OUTPUT_DIR) / f"{stem}_sfm.json"
            from admin_naming.shop_name_ui import run_ui
            run_ui(sfm_path=sfm_path, output_dir=OUTPUT_DIR)

        if args.admin_gui and not args.no_save:
            stem     = Path(ifc_paths[0]).stem
            sfm_path = Path(OUTPUT_DIR) / f"{stem}_sfm.json"
            from admin_naming.shop_name_gui import run_gui
            run_gui(sfm_path=sfm_path, output_dir=OUTPUT_DIR)

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
            pipeline.save_multi(OUTPUT_DIR)

        if args.admin_name and not args.no_save:
            from admin_naming.shop_name_ui import run_ui
            for ifc_path, floor_label, _ in floors_spec:
                stem     = Path(ifc_path).stem
                sfm_path = Path(OUTPUT_DIR) / f"{stem}_sfm.json"
                if sfm_path.exists():
                    print(f"\n[AdminNaming] Floor {floor_label} ({stem})")
                    run_ui(sfm_path=sfm_path, output_dir=OUTPUT_DIR)

        if args.admin_gui and not args.no_save:
            from admin_naming.shop_name_gui import run_gui
            for ifc_path, floor_label, _ in floors_spec:
                stem     = Path(ifc_path).stem
                sfm_path = Path(OUTPUT_DIR) / f"{stem}_sfm.json"
                if sfm_path.exists():
                    print(f"\n[AdminNamingGUI] Floor {floor_label} ({stem})")
                    run_gui(sfm_path=sfm_path, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
