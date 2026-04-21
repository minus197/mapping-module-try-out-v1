"""
map_extraction/inter_floor_linker.py  —  Sprint 7
--------------------------------------------------
Combines per-floor FloorGraphs into a single BuildingGraph by:

  1. Accepting a list of (FloorGraph, AdminConfig) pairs.
  2. Injecting admin-mentioned node tags into every node
     (human label, accessibility, connects_to floors).
  3. Matching vertical connector nodes across floors
     (elevator / escalator / stair at the same world XY position).
  4. Creating inter-floor NavigationEdges between matched connectors.
  5. Returning a BuildingGraph with all floors and inter-floor edges.

Admin tags
----------
Each floor can optionally receive an AdminConfig dict:

    {
        "floor_label": "L1",
        "floor_height_m": 4.0,          # vertical travel distance to next floor
        "nodes": {
            "FEAT-<guid>": {
                "admin_label":   "Main Escalator",
                "is_accessible": true,
                "connects_to":   ["L1", "L2"]
            }
        }
    }

When no AdminConfig is supplied the linker still works — it matches
connectors by XY proximity and uses default edge weights.

Inter-floor edge weights
------------------------
  safety_score   : elevator=1.0  escalator=0.8  stair=0.6
  shore_linable  : False  (vertical travel has no wall contact)
  landmark_score : 1.0    (vertical connectors are always major landmarks)
  distance       : floor_height_m (default 4.0 m if not specified)

Usage
-----
    from map_extraction.inter_floor_linker import InterFloorLinker

    linker = InterFloorLinker(building_name="Mall L1-L3")
    linker.add_floor(floor_graph_L1, admin_config_L1)
    linker.add_floor(floor_graph_L2, admin_config_L2)
    linker.add_floor(floor_graph_L3)           # no admin config — OK
    building = linker.build()

    print(building.summary())
    linker.save("data/outputs/building_graph.json")
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shared.types import (
    BuildingGraph,
    FloorGraph,
    NavigationEdge,
    NavigationNode,
    Point2D,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum XY distance (metres) for two connectors on different floors
# to be considered the "same" shaft (elevator / escalator / stair core)
CONNECTOR_MATCH_RADIUS = 1.5

DEFAULT_FLOOR_HEIGHT = 4.0   # metres — used when admin config omits it

# Safety scores for inter-floor travel by connector type
CONNECTOR_SAFETY = {
    "elevator":  1.0,
    "escalator": 0.8,
    "stair":     0.6,
}


# ---------------------------------------------------------------------------
# AdminConfig type alias
# ---------------------------------------------------------------------------

AdminConfig = Dict[str, Any]
# Expected structure:
# {
#   "floor_label":    "L1",
#   "floor_height_m": 4.0,
#   "nodes": {
#     "<node_id>": {
#       "admin_label":   "Main Escalator",
#       "is_accessible": true,
#       "connects_to":   ["L1", "L2"]
#     }
#   }
# }


# ---------------------------------------------------------------------------
# InterFloorLinker
# ---------------------------------------------------------------------------

class InterFloorLinker:
    """
    Builds a BuildingGraph from multiple per-floor FloorGraphs.

    Parameters
    ----------
    building_name : str   e.g. "One Galle Face Mall"
    """

    def __init__(self, building_name: str = "Building"):
        self.building_name  = building_name
        self._floors:  List[FloorGraph]    = []
        self._configs: List[AdminConfig]   = []
        self._result:  Optional[BuildingGraph] = None

    # ── Public ────────────────────────────────────────────────────────────────

    def add_floor(self,
                  floor_graph: FloorGraph,
                  admin_config: Optional[AdminConfig] = None) -> None:
        """
        Register one floor.  Floors are linked in the order they are added.

        Parameters
        ----------
        floor_graph  : FloorGraph produced by GraphBuilder.build()
        admin_config : optional dict with human labels and accessibility flags
        """
        self._floors.append(floor_graph)
        self._configs.append(admin_config or {})

    def build(self) -> BuildingGraph:
        """
        Run the full inter-floor linking pipeline and return a BuildingGraph.
        """
        if not self._floors:
            raise ValueError("No floors added. Call add_floor() first.")

        print(f"[InterFloorLinker] Building '{self.building_name}' "
              f"({len(self._floors)} floor(s)) ...")

        # Step 1: inject admin tags into all nodes
        for fg, cfg in zip(self._floors, self._configs):
            self._inject_admin_tags(fg, cfg)

        # Step 2: match vertical connectors across adjacent floor pairs
        inter_edges: List[NavigationEdge] = []
        for i in range(len(self._floors) - 1):
            fg_lower  = self._floors[i]
            fg_upper  = self._floors[i + 1]
            cfg_lower = self._configs[i]
            height    = float(cfg_lower.get("floor_height_m",
                                             DEFAULT_FLOOR_HEIGHT))
            edges = self._link_floors(fg_lower, fg_upper, height)
            inter_edges.extend(edges)
            print(f"[InterFloorLinker] {fg_lower.floor_label} ↔ "
                  f"{fg_upper.floor_label}: {len(edges)} inter-floor edges")

        building = BuildingGraph(
            building_name      = self.building_name,
            floors             = self._floors,
            inter_floor_edges  = inter_edges,
        )
        building.rebuild_index()
        self._result = building
        print(building.summary())
        return building

    def save(self, path: str | Path = "data/outputs/building_graph.json") -> Path:
        """Serialise the BuildingGraph to JSON."""
        if self._result is None:
            raise RuntimeError("Call build() before save().")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        data = self._to_dict(self._result)
        p.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[InterFloorLinker] Saved → {p.resolve()}")
        return p

    # ── Step 1: admin tag injection ───────────────────────────────────────────

    def _inject_admin_tags(self,
                           fg: FloorGraph,
                           cfg: AdminConfig) -> None:
        """
        Write admin-supplied labels and flags into node tags.
        Leaves existing tags untouched if no config entry exists.
        """
        node_cfgs: Dict[str, Dict] = cfg.get("nodes", {})

        for node in fg.nodes:
            # Always stamp floor_label (may already be set by GraphBuilder)
            node.tags["floor_label"] = fg.floor_label

            ncfg = node_cfgs.get(node.node_id, {})
            if not ncfg:
                continue

            if "admin_label" in ncfg:
                node.tags["admin_label"]   = str(ncfg["admin_label"])
                node.label = str(ncfg["admin_label"])   # update display label

            if "is_accessible" in ncfg:
                node.tags["is_accessible"] = str(ncfg["is_accessible"]).lower()

            if "connects_to" in ncfg:
                node.tags["connects_to"] = ",".join(
                    str(f) for f in ncfg["connects_to"]
                )

    # ── Step 2: inter-floor edge creation ────────────────────────────────────

    def _link_floors(self,
                     fg_lower: FloorGraph,
                     fg_upper: FloorGraph,
                     floor_height: float) -> List[NavigationEdge]:
        """
        Match vertical connectors (elevator/escalator/stair) between
        two adjacent floors by XY proximity and create inter-floor edges.

        Matching rule: two connectors of the same type on different floors
        are the same shaft if their XY positions are within
        CONNECTOR_MATCH_RADIUS metres.
        """
        lower_verts = fg_lower.vertical_connectors()
        upper_verts = fg_upper.vertical_connectors()

        edges: List[NavigationEdge] = []

        for lo in lower_verts:
            for hi in upper_verts:
                if lo.node_type != hi.node_type:
                    continue   # must be same connector type

                xy_dist = math.hypot(
                    hi.position[0] - lo.position[0],
                    hi.position[1] - lo.position[1],
                )
                if xy_dist > CONNECTOR_MATCH_RADIUS:
                    continue   # different shafts

                safety = CONNECTOR_SAFETY.get(lo.node_type, 0.7)

                # Annotate both nodes with which floors they connect
                for node in (lo, hi):
                    existing = node.tags.get("connects_to", "")
                    floors   = set(existing.split(",")) if existing else set()
                    floors.update([fg_lower.floor_label,
                                   fg_upper.floor_label])
                    floors.discard("")
                    node.tags["connects_to"] = ",".join(sorted(floors))

                eid = (f"INTER-"
                       f"{fg_lower.floor_label}-{fg_upper.floor_label}-"
                       f"{lo.node_type.upper()}-"
                       f"{lo.node_id[-8:]}")

                # Upward edge
                edges.append(NavigationEdge(
                    edge_id        = eid + "-UP",
                    source_id      = lo.node_id,
                    target_id      = hi.node_id,
                    distance       = round(floor_height, 3),
                    shore_linable  = False,
                    safety_score   = safety,
                    landmark_score = 1.0,
                    tags           = {
                        "edge_type":  "inter_floor",
                        "direction":  "up",
                        "from_floor": fg_lower.floor_label,
                        "to_floor":   fg_upper.floor_label,
                        "connector":  lo.node_type,
                        "xy_offset":  str(round(xy_dist, 3)),
                    },
                ))

                # Downward edge (reverse)
                edges.append(NavigationEdge(
                    edge_id        = eid + "-DOWN",
                    source_id      = hi.node_id,
                    target_id      = lo.node_id,
                    distance       = round(floor_height, 3),
                    shore_linable  = False,
                    safety_score   = safety,
                    landmark_score = 1.0,
                    tags           = {
                        "edge_type":  "inter_floor",
                        "direction":  "down",
                        "from_floor": fg_upper.floor_label,
                        "to_floor":   fg_lower.floor_label,
                        "connector":  lo.node_type,
                        "xy_offset":  str(round(xy_dist, 3)),
                    },
                ))

        return edges

    # ── Serialisation ─────────────────────────────────────────────────────────

    def _to_dict(self, bg: BuildingGraph) -> Dict[str, Any]:
        return {
            "meta": {
                "building_name":       bg.building_name,
                "floor_count":         len(bg.floors),
                "total_nodes":         len(bg.all_nodes),
                "total_intra_edges":   sum(len(f.edges) for f in bg.floors),
                "total_inter_edges":   len(bg.inter_floor_edges),
                "floor_labels":        [f.floor_label for f in bg.floors],
            },
            "floors": [
                {
                    "floor_label": fg.floor_label,
                    "source_file": fg.source_file,
                    "node_count":  len(fg.nodes),
                    "edge_count":  len(fg.edges),
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
                for fg in bg.floors
            ],
            "inter_floor_edges": [
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
                for e in bg.inter_floor_edges
            ],
        }
