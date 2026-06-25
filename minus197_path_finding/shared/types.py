"""
shared/types.py  —  WP0
------------------------
Dataclasses shared between map_extraction output and the pathfinding pipeline.
Nothing here imports from any other local module.

Key additions over the mapping-module stub:
  • NavigationEdge.combined_cost  — populated by pathfinding/cost.py (WP1)
  • PathStep / PathResult         — consumed by pathfinding/engine.py (WP6)
  • FloorGraph.from_dict()        — parses the _graph.json schema produced by
                                    map_extraction so the engine can load any
                                    saved graph without re-running the IFC pipeline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Primitive ─────────────────────────────────────────────────────────────────
Point2D = Tuple[float, float]   # (x, y) in metres


# ── Node ──────────────────────────────────────────────────────────────────────
@dataclass
class NavigationNode:
    """One node in the navigation graph."""
    node_id:   str
    label:     str
    position:  Point2D
    node_type: str          # zone_centroid | door | elevator | escalator |
                            # stair | junction | landmark
    zone_id:   Optional[str]
    tags:      Dict[str, str] = field(default_factory=dict)
                            # floor_label, admin_label, connects_to, is_accessible …


# ── Edge ──────────────────────────────────────────────────────────────────────
@dataclass
class NavigationEdge:
    """One edge in the navigation graph."""
    edge_id:        str
    source_id:      str
    target_id:      str
    distance:       float           # metres
    shore_linable:  bool
    safety_score:   float           # 0.0–1.0
    landmark_score: float           # raw (≥ 0); normalised by cost.py WP1
    tags:           Dict[str, str]  = field(default_factory=dict)
    combined_cost:  float           = 0.0   # populated by cost.compute_edge_costs()


# ── Graph ─────────────────────────────────────────────────────────────────────
@dataclass
class FloorGraph:
    """Complete navigation graph for one floor (or building-wide multi-floor)."""
    floor_label:  str
    source_file:  str
    nodes:        List[NavigationNode] = field(default_factory=list)
    edges:        List[NavigationEdge] = field(default_factory=list)
    spatial_meta: Dict[str, Any]       = field(default_factory=dict)

    _node_index: Dict[str, NavigationNode] = field(
        default_factory=dict, repr=False
    )

    def rebuild_index(self) -> None:
        self._node_index = {n.node_id: n for n in self.nodes}

    def node(self, node_id: str) -> Optional[NavigationNode]:
        return self._node_index.get(node_id)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FloorGraph":
        """
        Parse the _graph.json schema produced by map_extraction.

        Expected top-level keys: floor_label, source_file, nodes, edges,
        spatial_meta (optional).
        """
        nodes = [
            NavigationNode(
                node_id=n["node_id"],
                label=n["label"],
                position=(float(n["position"][0]), float(n["position"][1])),
                node_type=n["node_type"],
                zone_id=n.get("zone_id"),
                tags=n.get("tags", {}),
            )
            for n in data.get("nodes", [])
        ]
        edges = [
            NavigationEdge(
                edge_id=e["edge_id"],
                source_id=e["source_id"],
                target_id=e["target_id"],
                distance=float(e["distance"]),
                shore_linable=bool(e["shore_linable"]),
                safety_score=float(e["safety_score"]),
                landmark_score=float(e["landmark_score"]),
                tags=e.get("tags", {}),
                combined_cost=float(e.get("combined_cost", 0.0)),
            )
            for e in data.get("edges", [])
        ]
        fg = cls(
            floor_label=data.get("floor_label", ""),
            source_file=data.get("source_file", ""),
            nodes=nodes,
            edges=edges,
            spatial_meta=data.get("spatial_meta", {}),
        )
        fg.rebuild_index()
        return fg


# ── Path output types ─────────────────────────────────────────────────────────
@dataclass
class PathStep:
    """One segment in a navigation path, with a human-readable instruction."""
    from_node:   NavigationNode
    to_node:     NavigationNode
    edge:        NavigationEdge
    bearing:     float      # degrees clockwise from north (0–360)
    distance:    float      # metres for this step
    instruction: str        # e.g. "Turn left. Walk 12 m. You will reach the lift."


@dataclass
class PathResult:
    """Output of PathfindingEngine.find_path()."""
    found:            bool                = False
    start_node:       Optional[NavigationNode] = None
    destination_node: Optional[NavigationNode] = None
    steps:            List[PathStep]      = field(default_factory=list)
    alternatives:     List["PathResult"]  = field(default_factory=list)
    total_cost:       float               = 0.0
    total_distance:   float               = 0.0
    safety_score:     float               = 0.0
    shore_score:      float               = 0.0
    landmark_score:   float               = 0.0
