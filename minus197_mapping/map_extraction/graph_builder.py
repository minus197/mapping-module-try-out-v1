"""
map_extraction/graph_builder.py  —  Sprint 3–4
------------------------------------------------
Converts a SemanticFloorMap into a FloorGraph (navigation graph).

Sprint 3 — Skeletonisation
    Rasterises corridor polygons to a binary walkability grid,
    runs medial-axis skeletonisation (scikit-image), and extracts
    branch / endpoint nodes from the skeleton.

Sprint 4 — Graph construction
    Places nodes at:
      • Corridor skeleton branch/endpoints  (junction nodes)
      • Zone centroids                      (zone_centroid nodes)
      • Feature positions: door, elevator,
        escalator, stair, landmark          (feature nodes)
    Connects nodes with edges carrying:
      • distance   (Euclidean metres)
      • shore_linable  (True if edge adjacent to a wall)
      • safety_score   (based on corridor width + obstacle clearance)
      • landmark_score (density of landmark nodes on edge)

NOTE: This file is a Sprint 3–4 stub.
      The class interface and data contracts are fully defined so
      that pathfinding/engine.py can be written against it now.
      Implement _skeletonise() and _build_edges() in Sprint 3–4.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from shared.types import FloorGraph, NavigationEdge, NavigationNode, Point2D
from map_extraction.semantic_floor_map import SemanticFloorMap, Zone, Feature


class GraphBuilder:
    """
    Converts a SemanticFloorMap to a FloorGraph.

    Parameters
    ----------
    sfm : SemanticFloorMap
        Output of SemanticFloorMapBuilder.build().
    grid_resolution : float
        Metres per grid cell for the walkability raster (default 0.1 m).
    """

    def __init__(self, sfm: SemanticFloorMap, grid_resolution: float = 0.1):
        self.sfm  = sfm
        self.res  = grid_resolution
        self._nodes: List[NavigationNode] = []
        self._edges: List[NavigationEdge] = []
        self._node_map: Dict[str, NavigationNode] = {}

    # ── Public ───────────────────────────────────────────────────────────────

    def build(self) -> FloorGraph:
        """
        Full pipeline:
          1. Place feature nodes  (doors, elevators, stairs, landmarks)
          2. Place zone centroid nodes
          3. [Sprint 3] Skeletonise corridor zones → junction nodes
          4. [Sprint 4] Build and tag edges
        """
        self._place_feature_nodes()
        self._place_zone_centroid_nodes()
        self._skeletonise()          # Sprint 3 — fills junction nodes
        self._build_edges()          # Sprint 4 — fills edges

        graph = FloorGraph(
            floor_label = self.sfm.floor_label,
            source_file = self.sfm.source_file,
            nodes       = self._nodes,
            edges       = self._edges,
        )
        graph.rebuild_index()
        return graph

    # ── Node placement ────────────────────────────────────────────────────────

    def _place_feature_nodes(self) -> None:
        """Create one NavigationNode per feature (door/elevator/stair/etc.)."""
        for feat in self.sfm.features:
            ntype = _feature_type_to_node_type(feat.feature_type)
            node = NavigationNode(
                node_id   = f"FEAT-{feat.feature_id}",
                label     = feat.name,
                position  = feat.position,
                node_type = ntype,
                zone_id   = feat.zone_id,
                tags      = {
                    "feature_type": feat.feature_type,
                    "priority":     str(feat.priority),
                    "ifc_guid":     feat.ifc_guid,
                },
            )
            self._add_node(node)

    def _place_zone_centroid_nodes(self) -> None:
        """Create one centroid node per zone (shop, food_court, etc.)."""
        for zone in self.sfm.zones:
            node = NavigationNode(
                node_id   = f"ZONE-{zone.zone_id}",
                label     = zone.long_name or zone.name,
                position  = zone.centroid,
                node_type = "zone_centroid",
                zone_id   = zone.zone_id,
                tags      = {
                    "category": zone.category,
                    "name":     zone.name,
                    "area_m2":  str(round(zone.area, 2)),
                },
            )
            self._add_node(node)

    # ── Sprint 3 stub ─────────────────────────────────────────────────────────

    def _skeletonise(self) -> None:
        """
        [Sprint 3 — TODO]
        Steps:
          1. Compute bounding box of all corridor zones.
          2. Rasterise each corridor polygon to a binary grid
             (cell = 1 if inside corridor, 0 otherwise).
          3. Run skimage.morphology.medial_axis() on the binary grid.
          4. Find skeleton branch points (≥3 neighbours) and endpoints (1 neighbour).
          5. Convert grid coordinates back to world metres.
          6. Create NavigationNode(node_type='junction') for each.

        Libraries needed: scikit-image, numpy
        """
        # Placeholder — no-op until Sprint 3
        pass

    # ── Sprint 4 stub ─────────────────────────────────────────────────────────

    def _build_edges(self) -> None:
        """
        [Sprint 4 — TODO]
        Steps:
          1. For each pair of nodes within the same zone or adjacent zones,
             check line-of-sight against wall polygons.
          2. If clear, create a NavigationEdge with:
               distance      = Euclidean distance
               shore_linable = True if edge runs within SHORE_BUFFER of a wall
               safety_score  = f(corridor_width, obstacle_clearance)
               landmark_score= count of landmark nodes within 2 m of edge midpoint
          3. For multi-floor graphs, connect vertical connector nodes
             (elevator/escalator/stair) across floors.

        Libraries needed: shapely (line-of-sight), networkx (graph storage)
        """
        # Placeholder — simple fully-connected stub for testing
        # (connects every node to its nearest 3 neighbours)
        nodes = self._nodes
        for i, n1 in enumerate(nodes):
            distances = []
            for j, n2 in enumerate(nodes):
                if i == j:
                    continue
                d = _euclidean(n1.position, n2.position)
                distances.append((d, j))
            distances.sort()
            for d, j in distances[:3]:
                n2 = nodes[j]
                eid = f"EDGE-{n1.node_id}-{n2.node_id}"
                edge = NavigationEdge(
                    edge_id        = eid,
                    source_id      = n1.node_id,
                    target_id      = n2.node_id,
                    distance       = round(d, 4),
                    shore_linable  = False,   # TODO Sprint 4
                    safety_score   = 0.5,     # TODO Sprint 4
                    landmark_score = 0.0,     # TODO Sprint 4
                )
                self._edges.append(edge)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _add_node(self, node: NavigationNode) -> None:
        if node.node_id not in self._node_map:
            self._nodes.append(node)
            self._node_map[node.node_id] = node


# ── Module-level helpers ──────────────────────────────────────────────────────

def _feature_type_to_node_type(feature_type: str) -> str:
    mapping = {
        "elevator":   "elevator",
        "escalator":  "escalator",
        "stair":      "stair",
        "door":       "door",
        "info_desk":  "landmark",
        "bench":      "landmark",
        "furnishing": "landmark",
    }
    return mapping.get(feature_type, "landmark")


def _euclidean(a: Point2D, b: Point2D) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])
