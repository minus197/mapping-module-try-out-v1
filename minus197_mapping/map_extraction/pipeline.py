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

Multi-floor usage (one IFC file with several storeys):
    pipeline = MapExtractionPipeline.multi_floor_single_ifc(
        "data/ifc_files/floors_3-4_combined.ifc",
        building_name="ITFAC Mall",
    )
    building = pipeline.run_multi_single()   # → BuildingGraph
    pipeline.save_multi_single("data/outputs/")

Outputs produced per floor
--------------------------
  <stem>_graph.json        -- navigation graph (nodes + edges)
  <stem>_sfm.json          -- semantic floor map
  <stem>_occupancy.json    -- hybrid occupancy grid for perception module
  <stem>_path_nodes.json   -- cane-trailing waypoints
  <stem>_shop_names.json   -- admin shop-name mappings (empty stub if unnamed)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from map_extraction.ifc_parser import IFCParser
from map_extraction.semantic_floor_map import SemanticFloorMapBuilder
from map_extraction.graph_builder import GraphBuilder
from map_extraction.inter_floor_linker import InterFloorLinker, AdminConfig
from map_extraction.occupancy_grid import OccupancyGridExporter
from map_extraction.path_nodes import PathNodeBuilder
from shared.types import BuildingGraph, FloorGraph


class MapExtractionPipeline:
    """
    Entry point for the Map Extraction module.

    Parameters
    ----------
    ifc_path    : str | Path  -- path to .ifc file
    floor_label : str         -- floor identifier, e.g. 'L1', 'Ground'
    grid_res    : float       -- skeleton grid resolution in metres (default 0.1)
    admin_config: dict        -- optional admin tag overrides for this floor
    """

    def __init__(self,
                 ifc_path:    str | Path,
                 floor_label: str   = "L1",
                 grid_res:    float = 0.1,
                 admin_config: Optional[AdminConfig] = None,
                 target_storey_id: "int | None" = None,
                 output_stem: "str | None" = None):
        self.ifc_path         = Path(ifc_path)
        self.floor_label      = floor_label
        self.grid_res         = grid_res
        self.admin_config     = admin_config or {}
        self.target_storey_id = target_storey_id   # None = whole file (legacy)
        self.output_stem      = output_stem        # per-floor stem override

        self._sfm:   Optional[object]     = None
        self._graph: Optional[FloorGraph] = None

    # -- Single-floor API -----------------------------------------------------

    def run(self) -> FloorGraph:
        """Execute the single-floor extraction pipeline."""

        print(f"[MapExtraction] Parsing {self.ifc_path.name} ...")
        parse_result = IFCParser(
            self.ifc_path,
            target_storey_id=self.target_storey_id,
        ).parse()
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
        """
        Save all five outputs to output_dir:
          <stem>_graph.json       -- navigation graph
          <stem>_sfm.json         -- semantic floor map
          <stem>_occupancy.json   -- hybrid occupancy grid (perception module)
          <stem>_path_nodes.json  -- cane-trailing waypoints
          <stem>_shop_names.json  -- admin shop-name mappings (empty stub if unnamed)

        When an output_stem override is supplied (multi-floor mode), files are
        prefixed with that per-floor stem so floors do not overwrite each other.
        """
        if self._graph is None:
            raise RuntimeError("Call run() before save().")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        stem       = self.output_stem or self.ifc_path.stem
        graph_path = out / f"{stem}_graph.json"
        sfm_path   = out / f"{stem}_sfm.json"
        occ_path   = out / f"{stem}_occupancy.json"
        path_path  = out / f"{stem}_path_nodes.json"
        names_path = out / f"{stem}_shop_names.json"

        bb = self._sfm.bounding_box if self._sfm else None
        _save_floor_graph(self._graph, graph_path,
                        bounding_box=bb,
                        grid_res=self.grid_res)

        if self._sfm:
            self._sfm.save(sfm_path)

            # Occupancy grid for perception module
            print("[MapExtraction] Building occupancy grid ...")
            OccupancyGridExporter(self._sfm).build().save(occ_path)

            # Path nodes — cane-trailing waypoints along corridor-facing walls
            print("[MapExtraction] Building path nodes ...")
            PathNodeBuilder(self._sfm).build().save(path_path)

        # Always emit the 5th file as a stub so every floor has all five
        # artifacts even before an admin naming session. An empty {} is the
        # "no mappings" case the admin patcher already tolerates; the admin
        # UI/GUI overwrites it later, keyed to the same <stem>_sfm.json.
        if not names_path.exists():
            names_path.write_text(json.dumps({}, indent=2), encoding="utf-8")
            print(f"[MapExtraction] Wrote empty shop-names stub → {names_path}")

        return graph_path

    # -- Multi-floor API -------------------------------------------------------

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
        instance._floor_sfms: List[Tuple[str, object]] = []
        return instance

    def run_multi(self) -> BuildingGraph:
        """Execute the multi-floor extraction pipeline."""
        linker = InterFloorLinker(building_name=self._building_name)

        self._floor_sfms = []    # (ifc_stem, sfm) per floor

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

            # Collect the SFM for occupancy grid generation in save_multi()
            if p._sfm:
                self._floor_sfms.append((Path(ifc_path).stem, p._sfm))

        self._building = linker.build()
        return self._building

    def save_multi(self,
                   output_dir: str | Path = "data/outputs/") -> Path:
        """
        Save BuildingGraph JSON and one occupancy grid per floor.

          <building_name>_building_graph.json
          <stem_L1>_occupancy.json
          <stem_L2>_occupancy.json
          ...
        """
        if self._building is None:
            raise RuntimeError("Call run_multi() before save_multi().")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        name = self._building_name.replace(" ", "_")
        path = out / f"{name}_building_graph.json"
        linker = InterFloorLinker(self._building_name)
        linker._result = self._building
        linker.save(path)

        # One occupancy grid + path-node layer per floor
        for stem, sfm in getattr(self, "_floor_sfms", []):
            occ_path = out / f"{stem}_occupancy.json"
            print(f"[MapExtraction] Building occupancy grid for {stem} ...")
            OccupancyGridExporter(sfm).build().save(occ_path)

            path_path = out / f"{stem}_path_nodes.json"
            print(f"[MapExtraction] Building path nodes for {stem} ...")
            PathNodeBuilder(sfm).build().save(path_path)

        return path

    # -- Multi-floor from a single multi-storey IFC ---------------------------

    @classmethod
    def multi_floor_single_ifc(cls,
                               ifc_path: str | Path,
                               building_name: str = "Building",
                               grid_res: float = 0.1) -> "MapExtractionPipeline":
        """
        Factory for a single IFC file containing several IfcBuildingStoreys.

        The file is split internally by storey; each populated storey runs the
        existing per-floor extraction once, and the floors are stitched into one
        building graph. Call run_multi_single() then save_multi_single().
        """
        inst = cls.__new__(cls)
        inst._single_ifc      = str(ifc_path)
        inst._building_name   = building_name
        inst._grid_res        = grid_res
        inst._building        = None
        inst._sfm             = None
        inst._graph           = None
        inst._floor_sfms:      List[Tuple[str, object]] = []
        inst._floor_out_stems: List[Tuple[str, "MapExtractionPipeline"]] = []
        return inst

    def run_multi_single(self) -> BuildingGraph:
        """
        Enumerate populated storeys in the single IFC, run one sub-pipeline per
        storey with a per-floor output stem, compute true floor heights from
        storey elevations, and link the floors into one BuildingGraph.
        """
        import ifcopenshell
        from map_extraction.ifc_parser import list_populated_storeys

        model   = ifcopenshell.open(self._single_ifc)
        storeys = list_populated_storeys(model)   # [(id, name, elev)] bottom→top
        if not storeys:
            raise ValueError("No populated storeys found in the IFC.")

        ifc_stem = Path(self._single_ifc).stem
        linker   = InterFloorLinker(building_name=self._building_name)

        self._floor_sfms      = []
        self._floor_out_stems = []

        for i, (sid, sname, elev) in enumerate(storeys):
            floor_label = _label_from_storey_name(sname)   # "Level 3" -> "L3"
            out_stem    = f"{ifc_stem}_{floor_label}"

            # true floor height = elevation delta to the storey above
            if i < len(storeys) - 1:
                floor_height = round(storeys[i + 1][2] - elev, 3)
            else:
                floor_height = None

            print(f"\n{'═'*50}\n[MapExtraction] Storey {sname!r} "
                  f"(id={sid}, elev={elev}) → floor {floor_label}\n{'═'*50}")

            p = MapExtractionPipeline(
                ifc_path         = self._single_ifc,
                floor_label      = floor_label,
                grid_res         = self._grid_res,
                target_storey_id = sid,
                output_stem      = out_stem,
            )
            fg = p.run()

            admin_cfg = {"floor_height_m": floor_height} if floor_height else {}
            linker.add_floor(fg, admin_cfg)

            if p._sfm:
                self._floor_sfms.append((out_stem, p._sfm))
            self._floor_out_stems.append((out_stem, p))

        self._building = linker.build()
        return self._building

    def save_multi_single(self,
                          output_dir: str | Path = "data/outputs/") -> Path:
        """
        Save all five artifacts per floor plus one building graph tying the
        floors together (via the shared stair / elevator shafts).
        """
        if self._building is None:
            raise RuntimeError("Call run_multi_single() before save_multi_single().")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 5 files per floor (graph/sfm/occupancy/path_nodes/shop_names stub)
        for out_stem, p in self._floor_out_stems:
            p.save(out)

        # one building graph tying the floors together
        name   = self._building_name.replace(" ", "_")
        linker = InterFloorLinker(self._building_name)
        linker._result = self._building
        linker.save(out / f"{name}_building_graph.json")
        return out

    # -- Properties -----------------------------------------------------------

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

def _label_from_storey_name(name: str) -> str:
    """'Level 3' -> 'L3';  falls back to the sanitised name when no digits."""
    digits = "".join(ch for ch in name if ch.isdigit())
    return f"L{digits}" if digits else name.strip().replace(" ", "_")


def _save_floor_graph(fg: FloorGraph,
                      path: Path,
                      bounding_box: dict = None,
                      grid_res: float = 0.1) -> None:
    data = {
        "floor_label": fg.floor_label,
        "source_file": fg.source_file,

        # ── Spatial metadata ──────────────────────────────────────────────
        # Navigation module needs this to interpret node positions.
        # All node positions are in the same IFC coordinate system.
        "spatial_meta": {
            "units": "metres",
            "coordinate_frame": {
                "units": "metres",
                "source": "IFC project coordinate system",
                "x_axis": "IFC project X axis",
                "y_axis": "IFC project Y axis",
                "origin_description": (
                    "IFC project origin — node positions are exact "
                    "Shapely coordinates in this frame"
                ),
            },
            "bounding_box": (
                {k: float(v) for k, v in bounding_box.items()}
                if bounding_box else None
            ),
            "skeleton_grid_resolution_m": grid_res,
            "note": (
                "Node positions are exact Shapely coordinates — "
                "not quantised to the skeleton grid resolution"
            ),
        },

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