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
from pathfinding import PathfindingEngine


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
                        help="Destination query e.g. 'food court'")
    parser.add_argument("--start",    default=None,
                        help="Start node_id (default: first node)")
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
            _run_pathfinding(graph, args.start, args.query)

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

        if args.query:
            # Use ground floor for pathfinding demo
            ground_fg = building.floors[0]
            _run_pathfinding(ground_fg, args.start, args.query)


def _run_pathfinding(graph, start_id, query):
    from pathfinding import PathfindingEngine
    engine   = PathfindingEngine(graph)
    start_id = start_id or graph.nodes[0].node_id

    print(f"\n[Pathfinding] Start : {start_id}")
    print(f"[Pathfinding] Query : {query!r}")

    result = engine.find_path(start_id, query)
    print("\n" + result.summary())

    if result.found:
        print("Steps:")
        for i, step in enumerate(result.steps, 1):
            print(f"  {i:2d}. {step.instruction}")
    else:
        print("No path found for that destination.")


if __name__ == "__main__":
    main()
