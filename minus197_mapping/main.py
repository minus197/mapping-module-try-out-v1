"""
main.py
-------
Top-level entry point for the Minus197 Mapping Module.
Runs the full Map Extraction → Pathfinding pipeline on a given IFC file.

Usage
-----
    python main.py --ifc data/ifc_files/building.ifc --query "Nike Store"
    python main.py --ifc data/ifc_files/building.ifc --floor L2 --query "food court"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from map_extraction import MapExtractionPipeline
from pathfinding import PathfindingEngine


def main():
    parser = argparse.ArgumentParser(
        description="Minus197 Indoor Navigation — Map Extraction + Pathfinding"
    )
    parser.add_argument("--ifc",   required=True, help="Path to .ifc file")
    parser.add_argument("--floor", default="L1",  help="Floor label (default: L1)")
    parser.add_argument("--query", default=None,  help="Destination query (e.g. 'food court')")
    parser.add_argument("--start", default=None,  help="Start node_id (default: first node)")
    parser.add_argument("--save",  default=True,  action=argparse.BooleanOptionalAction,
                        help="Save outputs to data/outputs/ (default: True)")
    args = parser.parse_args()

    # ── Map Extraction ────────────────────────────────────────────────────────
    pipeline = MapExtractionPipeline(args.ifc, floor_label=args.floor)
    graph    = pipeline.run()

    if args.save:
        ifc_stem = Path(args.ifc).stem
        pipeline.save(f"data/outputs/{ifc_stem}_graph.json")
        if pipeline.sfm:
            pipeline.sfm.save(f"data/outputs/{ifc_stem}_sfm.json")

    # ── Pathfinding ───────────────────────────────────────────────────────────
    if args.query:
        engine     = PathfindingEngine(graph)
        start_id   = args.start or graph.nodes[0].node_id
        print(f"\n[Pathfinding] Start: {start_id}")
        print(f"[Pathfinding] Query: {args.query!r}")

        result = engine.find_path(start_id, args.query)
        print("\n" + result.summary())

        if result.found:
            print("Steps:")
            for i, step in enumerate(result.steps, 1):
                print(f"  {i:2d}. {step.instruction}")
        else:
            print("No path found for that destination.")


if __name__ == "__main__":
    main()
