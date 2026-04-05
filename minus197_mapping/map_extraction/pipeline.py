"""
map_extraction/pipeline.py
--------------------------
Orchestrates the complete Map Extraction pipeline:

  IFC file
    └── IFCParser              (Sprint 1)
          └── SemanticFloorMapBuilder   (Sprint 2)
                └── GraphBuilder       (Sprint 3–4)
                      └── FloorGraph   → consumed by pathfinding

Usage
-----
    from map_extraction import MapExtractionPipeline

    pipeline = MapExtractionPipeline("building.ifc", floor_label="L1")
    graph    = pipeline.run()
    pipeline.save("data/outputs/floor_graph.json")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from map_extraction.ifc_parser import IFCParser
from map_extraction.semantic_floor_map import SemanticFloorMapBuilder
from map_extraction.graph_builder import GraphBuilder
from shared.types import FloorGraph


class MapExtractionPipeline:
    """
    Single entry point for the Map Extraction module.

    Parameters
    ----------
    ifc_path    : str | Path  — path to .ifc file
    floor_label : str         — floor identifier, e.g. 'L1', 'B1', 'Ground'
    grid_res    : float       — skeletonisation grid resolution in metres (default 0.1)
    """

    def __init__(self,
                 ifc_path:    str | Path,
                 floor_label: str   = "L1",
                 grid_res:    float = 0.1):
        self.ifc_path    = Path(ifc_path)
        self.floor_label = floor_label
        self.grid_res    = grid_res

        # Internal state — populated by run()
        self._sfm:   Optional[object]     = None
        self._graph: Optional[FloorGraph] = None

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self) -> FloorGraph:
        """Execute the full extraction pipeline and return a FloorGraph."""

        # Sprint 1: Parse IFC
        print(f"[MapExtraction] Parsing {self.ifc_path.name} ...")
        parse_result = IFCParser(self.ifc_path).parse()
        print(parse_result.summary())

        # Sprint 2: Build Semantic Floor Map Object
        print("[MapExtraction] Building Semantic Floor Map Object ...")
        self._sfm = SemanticFloorMapBuilder(
            parse_result, floor_label=self.floor_label
        ).build()
        print(self._sfm.summary())

        # Sprint 3–4: Build navigation graph
        print("[MapExtraction] Building navigation graph ...")
        self._graph = GraphBuilder(self._sfm, grid_resolution=self.grid_res).build()
        print(f"[MapExtraction] Graph: {len(self._graph.nodes)} nodes, "
              f"{len(self._graph.edges)} edges")

        return self._graph

    def save(self, path: str | Path = "data/outputs/floor_graph.json") -> Path:
        """Serialise the FloorGraph to JSON."""
        if self._graph is None:
            raise RuntimeError("Call run() before save().")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "floor_label": self._graph.floor_label,
            "source_file": self._graph.source_file,
            "nodes": [
                {
                    "node_id":   n.node_id,
                    "label":     n.label,
                    "position":  list(n.position),
                    "node_type": n.node_type,
                    "zone_id":   n.zone_id,
                    "tags":      n.tags,
                }
                for n in self._graph.nodes
            ],
            "edges": [
                {
                    "edge_id":        e.edge_id,
                    "source_id":      e.source_id,
                    "target_id":      e.target_id,
                    "distance":       e.distance,
                    "shore_linable":  e.shore_linable,
                    "safety_score":   e.safety_score,
                    "landmark_score": e.landmark_score,
                    "tags":           e.tags,
                }
                for e in self._graph.edges
            ],
        }
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[MapExtraction] Saved graph → {p.resolve()}")
        return p

    @property
    def sfm(self):
        """Access the intermediate SemanticFloorMap (for debugging)."""
        return self._sfm

    @property
    def graph(self) -> Optional[FloorGraph]:
        return self._graph
