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
                                       # arbitrary key-value metadata
                                       # e.g. {"category": "shop", "name": "Nike"}


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


@dataclass
class FloorGraph:
    """
    Complete navigation graph for one floor.
    This is the canonical output of map_extraction and the
    canonical input to pathfinding.
    """
    floor_label:  str
    source_file:  str
    nodes:        List[NavigationNode] = field(default_factory=list)
    edges:        List[NavigationEdge] = field(default_factory=list)

    # Fast lookup helpers (populated after construction)
    _node_index:  Dict[str, NavigationNode] = field(
        default_factory=dict, repr=False
    )

    def node(self, node_id: str) -> Optional[NavigationNode]:
        return self._node_index.get(node_id)

    def rebuild_index(self) -> None:
        self._node_index = {n.node_id: n for n in self.nodes}
