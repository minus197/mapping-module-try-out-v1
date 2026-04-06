"""
main.py
-------
Top-level entry point for the Minus197 Mapping Module.
Runs map extraction on a given IFC file.

Usage
-----
    python main.py --ifc data/ifc_files/building.ifc
    python main.py --ifc data/ifc_files/building.ifc --floor L2
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from map_extraction import MapExtractionPipeline
# ...existing code...


def main():
    parser = argparse.ArgumentParser(
        description="Minus197 Indoor Navigation — Map Extraction"
    )
    parser.add_argument("--ifc",   required=True, help="Path to .ifc file")
    parser.add_argument("--floor", default="L1",  help="Floor label (default: L1)")
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

    print(f"[Done] Extracted graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")


if __name__ == "__main__":
    main()