"""
shared/types.py
---------------
Type aliases and lightweight dataclasses shared between
map_extraction and pathfinding modules.

Nothing in this file imports from either module — it is the
single source of truth for cross-module data contracts.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── Primitive ────────────────────────────────────────────────────────────────
Point2D = Tuple[float, float]          # (x, y) in metres


# ── Shared node descriptor ────────────────────────────────────────────────────
@dataclass
class NavigationNode:
    """
    One node in the navigation graph.
    Produced by map_extraction, consumed by pathfinding.
    """
    node_id:      str                  # e.g. "ZONE-ENTRANCE", "DOOR-001"
    label:        str                  # human-readable name
    position:     Point2D
    node_type:    str                  # zone_centroid | door | elevator |
                                       # escalator | stair | junction | landmark
    zone_id:      Optional[str]        # which zone this node belongs to
    tags:         Dict[str, str]       = field(default_factory=dict)
                                       # key-value metadata including:
                                       #   floor_label    — e.g. "L1"
                                       #   admin_label    — mall admin name
                                       #   connects_to    — comma-sep floor labels
                                       #   is_accessible  — "true"/"false"


@dataclass
class NavigationEdge:
    """
    One edge in the navigation graph.
    Produced by map_extraction, consumed by pathfinding.
    """
    edge_id:        str
    source_id:      str                # node_id
    target_id:      str                # node_id
    distance:       float              # metres
    shore_linable:  bool               # adjacent to a wall / tactile surface
    safety_score:   float              # 0.0–1.0 (1 = safest corridor)
    landmark_score: float              # 0.0–1.0 (1 = highest landmark density)
    tags:           Dict[str, str]     = field(default_factory=dict)
                                       # key-value metadata including:
                                       #   edge_type   — intra_floor | inter_floor
                                       #   from_floor  — e.g. "L1"
                                       #   to_floor    — e.g. "L2"


@dataclass
class FloorGraph:
    """
    Complete navigation graph for one floor.
    Produced by map_extraction.GraphBuilder.
    Consumed by InterFloorLinker and pathfinding.
    """
    floor_label:  str
    source_file:  str
    nodes:        List[NavigationNode] = field(default_factory=list)
    edges:        List[NavigationEdge] = field(default_factory=list)

    _node_index:  Dict[str, NavigationNode] = field(
        default_factory=dict, repr=False
    )

    def node(self, node_id: str) -> Optional[NavigationNode]:
        return self._node_index.get(node_id)

    def rebuild_index(self) -> None:
        self._node_index = {n.node_id: n for n in self.nodes}

    def vertical_connectors(self) -> List[NavigationNode]:
        """Return all elevator / escalator / stair nodes on this floor."""
        return [n for n in self.nodes
                if n.node_type in ("elevator", "escalator", "stair")]


@dataclass
class BuildingGraph:
    """
    Complete multi-floor navigation graph for one building.

    Contains all per-floor FloorGraphs plus inter-floor edges that
    connect vertical connector nodes (elevator / escalator / stair)
    across floors.

    Produced by map_extraction.InterFloorLinker.
    Consumed by pathfinding.PathfindingEngine (instead of FloorGraph
    when multi-floor routing is needed).
    """
    building_name:    str
    floors:           List[FloorGraph]        = field(default_factory=list)
    inter_floor_edges: List[NavigationEdge]   = field(default_factory=list)

    # Flat lookup across all floors — populated by rebuild_index()
    _node_index:  Dict[str, NavigationNode]   = field(
        default_factory=dict, repr=False
    )
    _floor_index: Dict[str, FloorGraph]       = field(
        default_factory=dict, repr=False
    )

    def rebuild_index(self) -> None:
        self._node_index  = {}
        self._floor_index = {}
        for fg in self.floors:
            fg.rebuild_index()
            self._node_index.update(fg._node_index)
            self._floor_index[fg.floor_label] = fg

    def node(self, node_id: str) -> Optional[NavigationNode]:
        return self._node_index.get(node_id)

    def floor(self, label: str) -> Optional[FloorGraph]:
        return self._floor_index.get(label)

    @property
    def all_nodes(self) -> List[NavigationNode]:
        return list(self._node_index.values())

    @property
    def all_edges(self) -> List[NavigationEdge]:
        edges = []
        for fg in self.floors:
            edges.extend(fg.edges)
        edges.extend(self.inter_floor_edges)
        return edges

    def summary(self) -> str:
        lines = [f"BuildingGraph — {self.building_name}"]
        for fg in self.floors:
            vert = len(fg.vertical_connectors())
            lines.append(
                f"  Floor {fg.floor_label:4s}: "
                f"{len(fg.nodes):3d} nodes  "
                f"{len(fg.edges):3d} intra-floor edges  "
                f"{vert} vertical connectors"
            )
        lines.append(
            f"  Inter-floor edges : {len(self.inter_floor_edges)}"
        )
        lines.append(
            f"  Total nodes       : {len(self._node_index)}"
        )
        return "\n".join(lines)
