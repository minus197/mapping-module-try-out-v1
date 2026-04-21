"""
map_extraction/pipeline.py
--------------------------
Orchestrates the complete Map Extraction pipeline.

Single-floor usage (one IFC file):
    pipeline = MapExtractionPipeline("building.ifc", floor_label="L1")
    graph    = pipeline.run()          # → FloorGraph
    pipeline.save("data/outputs/")

Multi-floor usage (one IFC file per floor):
    pipeline = MapExtractionPipeline.multi_floor(
        floors=[
            ("data/ifc_files/mall_L1.ifc", "L1",
             {"floor_height_m": 4.0, "nodes": {...}}),
            ("data/ifc_files/mall_L2.ifc", "L2",
             {"floor_height_m": 4.0}),
            ("data/ifc_files/mall_L3.ifc", "L3", {}),
        ],
        building_name="One Galle Face Mall",
    )
    building = pipeline.run_multi()    # → BuildingGraph
    pipeline.save_multi("data/outputs/")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from map_extraction.ifc_parser import IFCParser
from map_extraction.semantic_floor_map import SemanticFloorMapBuilder
from map_extraction.graph_builder import GraphBuilder
from map_extraction.inter_floor_linker import InterFloorLinker, AdminConfig
from shared.types import BuildingGraph, FloorGraph


class MapExtractionPipeline:
    """
    Entry point for the Map Extraction module.

    Parameters
    ----------
    ifc_path    : str | Path  — path to .ifc file
    floor_label : str         — floor identifier, e.g. 'L1', 'Ground'
    grid_res    : float       — skeleton grid resolution in metres (default 0.1)
    admin_config: dict        — optional admin tag overrides for this floor
    """

    def __init__(self,
                 ifc_path:    str | Path,
                 floor_label: str   = "L1",
                 grid_res:    float = 0.1,
                 admin_config: Optional[AdminConfig] = None):
        self.ifc_path    = Path(ifc_path)
        self.floor_label = floor_label
        self.grid_res    = grid_res
        self.admin_config = admin_config or {}

        self._sfm:   Optional[object]     = None
        self._graph: Optional[FloorGraph] = None

    # ── Single-floor API ──────────────────────────────────────────────────────

    def run(self) -> FloorGraph:
        """Execute the single-floor extraction pipeline."""

        print(f"[MapExtraction] Parsing {self.ifc_path.name} ...")
        parse_result = IFCParser(self.ifc_path).parse()
        print(parse_result.summary())

        print("[MapExtraction] Building Semantic Floor Map Object ...")
        self._sfm = SemanticFloorMapBuilder(
            parse_result, floor_label=self.floor_label
        ).build()
        print(self._sfm.summary())

        print("[MapExtraction] Building navigation graph ...")
        self._graph = GraphBuilder(
            self._sfm, grid_resolution=self.grid_res
        ).build()
        print(f"[MapExtraction] Graph: {len(self._graph.nodes)} nodes, "
              f"{len(self._graph.edges)} edges")

        # Inject admin tags if supplied
        if self.admin_config:
            linker = InterFloorLinker()
            linker._inject_admin_tags(self._graph, self.admin_config)
            print(f"[MapExtraction] Admin tags injected for "
                  f"{len(self.admin_config.get('nodes', {}))} nodes")

        return self._graph

    def save(self, output_dir: str | Path = "data/outputs/") -> Path:
        """Save FloorGraph JSON to output_dir."""
        if self._graph is None:
            raise RuntimeError("Call run() before save().")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        stem  = self.ifc_path.stem
        graph_path = out / f"{stem}_graph.json"
        sfm_path   = out / f"{stem}_sfm.json"

        _save_floor_graph(self._graph, graph_path)
        if self._sfm:
            self._sfm.save(sfm_path)

        return graph_path

    # ── Multi-floor API ───────────────────────────────────────────────────────

    @classmethod
    def multi_floor(
        cls,
        floors: List[Tuple[str | Path, str, Dict[str, Any]]],
        building_name: str = "Building",
        grid_res: float    = 0.1,
    ) -> "MapExtractionPipeline":
        """
        Factory for multi-floor pipelines.

        Parameters
        ----------
        floors : list of (ifc_path, floor_label, admin_config) tuples
        building_name : str
        grid_res      : float  skeleton grid resolution

        Returns a pipeline configured for multi-floor extraction.
        Call run_multi() to execute.
        """
        instance = cls.__new__(cls)
        instance._floors_spec   = floors
        instance._building_name = building_name
        instance._grid_res      = grid_res
        instance._building:  Optional[BuildingGraph] = None
        instance._sfm        = None
        instance._graph      = None
        return instance

    def run_multi(self) -> BuildingGraph:
        """Execute the multi-floor extraction pipeline."""
        linker = InterFloorLinker(building_name=self._building_name)

        for ifc_path, floor_label, admin_cfg in self._floors_spec:
            print(f"\n{'─'*50}")
            print(f"[MapExtraction] Floor {floor_label}: "
                  f"{Path(ifc_path).name}")
            print(f"{'─'*50}")

            p = MapExtractionPipeline(
                ifc_path    = ifc_path,
                floor_label = floor_label,
                grid_res    = self._grid_res,
            )
            fg = p.run()
            linker.add_floor(fg, admin_cfg)

        self._building = linker.build()
        return self._building

    def save_multi(self,
                   output_dir: str | Path = "data/outputs/") -> Path:
        """Save BuildingGraph JSON."""
        if self._building is None:
            raise RuntimeError("Call run_multi() before save_multi().")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        name = self._building_name.replace(" ", "_")
        path = out / f"{name}_building_graph.json"
        linker = InterFloorLinker(self._building_name)
        linker._result = self._building
        linker.save(path)
        return path

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def sfm(self):
        return self._sfm

    @property
    def graph(self) -> Optional[FloorGraph]:
        return self._graph

    @property
    def building(self) -> Optional[BuildingGraph]:
        return getattr(self, "_building", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_floor_graph(fg: FloorGraph, path: Path) -> None:
    data = {
        "floor_label": fg.floor_label,
        "source_file": fg.source_file,
        "nodes": [
            {
                "node_id":   n.node_id,
                "label":     n.label,
                "position":  [float(v) for v in n.position],
                "node_type": n.node_type,
                "zone_id":   n.zone_id,
                "tags":      n.tags,
            }
            for n in fg.nodes
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
            for e in fg.edges
        ],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"[MapExtraction] Saved graph → {path.resolve()}")
