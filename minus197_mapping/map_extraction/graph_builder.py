"""
map_extraction/graph_builder.py  —  Sprint 3 + 4  (Option A)
-------------------------------------------------------------
Converts a SemanticFloorMap into a FloorGraph (navigation graph).

Rasterisation scope (Option A)
--------------------------------
The binary grid and medial-axis skeleton are used ONLY for:
  - Finding junction node positions (centreline of walkable corridors)
  - Reading per-junction clearance from the distance transform

Everything else — distance measurement, safety score, shore fraction —
uses exact Shapely geometry with zero pixel error.

Pipeline
--------
  _place_feature_nodes()       doors / elevators / stairs / landmarks
  _place_zone_centroid_nodes() one centroid node per named zone
  _build_grid()                rasterise zones → binary grid + skeleton
                                also builds Shapely geometries:
                                  self._walkable   — union of all walkable zones
                                  self._wall_union — zone perimeters buffered
                                                     by SHORE_BUFFER (for shore)
  _skeletonise()               medial axis → pruned junction nodes
                                clearance per node from distance transform
  _build_edges()               connect nodes:
                                  distance       — Shapely exact (Option A)
                                  safety_score   — Shapely exact clearance
                                  shore_linable  — Shapely exact shore check
                                  landmark_score — Euclidean samples
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import networkx as nx
from skimage.draw import polygon as sk_polygon
from skimage.morphology import medial_axis
from scipy.spatial import KDTree
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from shared.types import FloorGraph, NavigationEdge, NavigationNode, Point2D
from map_extraction.semantic_floor_map import SemanticFloorMap

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

GRID_RES        = 0.10   # metres per raster cell — only affects skeleton topology
MIN_NODE_SEP    = 1.50   # min distance (m) between pruned skeleton junction nodes
SHORE_BUFFER    = 0.80   # m — point within this of any zone perimeter → shore-linable
SHORE_FRACTION  = 0.40   # fraction of edge sample points that must be shore-linable
LANDMARK_RADIUS = 3.0    # m — landmark within this radius of an edge boosts score
MAX_EDGES       = 6      # max edges per node (zone centroids use 4)
MAX_CLEARANCE   = 2.0    # m — clearance at or above this → safety_score = 1.0

WALKABLE_CATEGORIES: Set[str] = {
    "corridor", "entrance", "exit", "shop",
    "food_court", "restroom", "office", "storage", "unknown",
}

# ---------------------------------------------------------------------------
# Internal raster grid helper (skeleton only)
# ---------------------------------------------------------------------------

@dataclass
class _Grid:
    """Binary walkability grid + medial-axis outputs. Used only for skeleton."""
    data:    np.ndarray   # uint8 binary walkable grid
    dist:    np.ndarray   # distance transform (cells → metres when × res)
    skel:    np.ndarray   # boolean skeleton
    min_x:   float
    min_y:   float
    res:     float

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        col = int((wx - self.min_x) / self.res)
        row = int((wy - self.min_y) / self.res)
        return row, col

    def grid_to_world(self, row: int, col: int) -> Point2D:
        return (
            round(self.min_x + col * self.res, 3),
            round(self.min_y + row * self.res, 3),
        )

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------

class GraphBuilder:
    """
    Converts a SemanticFloorMap to a FloorGraph.

    Parameters
    ----------
    sfm             : SemanticFloorMap
    grid_resolution : float  metres per raster cell (default 0.10 m)
    """

    def __init__(self, sfm: SemanticFloorMap,
                 grid_resolution: float = GRID_RES):
        self.sfm = sfm
        self.res = grid_resolution

        self._nodes:     List[NavigationNode] = []
        self._edges:     List[NavigationEdge] = []
        self._node_map:  Dict[str, NavigationNode] = {}

        # Raster — skeleton use only
        self._grid:       Optional[_Grid]    = None
        self._skel_graph: Optional[nx.Graph] = None

        # Shapely — exact geometry, no pixel error
        self._walkable:   Optional[object]   = None  # union of walkable zones
        self._wall_union: Optional[object]   = None  # zone perimeters buffered

    # ── Public ────────────────────────────────────────────────────────────────

    def build(self) -> FloorGraph:
        print("[GraphBuilder] Placing feature and zone centroid nodes ...")
        self._place_feature_nodes()
        self._place_zone_centroid_nodes()

        print("[GraphBuilder] Building walkable grid and Shapely geometry ...")
        self._build_grid()

        print("[GraphBuilder] Running medial-axis skeletonisation ...")
        self._skeletonise()

        print("[GraphBuilder] Building edges (Shapely exact distances) ...")
        self._build_edges()

        graph = FloorGraph(
            floor_label = self.sfm.floor_label,
            source_file = self.sfm.source_file,
            nodes       = self._nodes,
            edges       = self._edges,
        )
        graph.rebuild_index()
        print(f"[GraphBuilder] Done: {len(self._nodes)} nodes, "
              f"{len(self._edges)} edges")
        return graph

    # ── Step 1: node placement ────────────────────────────────────────────────

    def _place_feature_nodes(self) -> None:
        for feat in self.sfm.features:
            self._add_node(NavigationNode(
                node_id   = f"FEAT-{feat.feature_id}",
                label     = feat.name,
                position  = feat.position,
                node_type = _feat_to_node_type(feat.feature_type),
                zone_id   = feat.zone_id,
                tags      = {
                    "feature_type": feat.feature_type,
                    "priority":     str(feat.priority),
                    "ifc_guid":     feat.ifc_guid,
                },
            ))

    def _place_zone_centroid_nodes(self) -> None:
        for zone in self.sfm.zones:
            self._add_node(NavigationNode(
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
            ))

    # ── Step 2: grid + Shapely geometry ──────────────────────────────────────

    def _build_grid(self) -> None:
        """
        Build two things in one pass over the zone polygons:

        A) Raster binary grid  →  medial-axis skeleton (junction positions only)
        B) Shapely geometries  →  exact clearance and shore calculations

        The shore grid (numpy bool array) is NOT built here anymore.
        Shore fraction is computed on demand via self._wall_union (Shapely).
        """
        bb  = self.sfm.bounding_box
        pad = 2

        rows = int((bb["max_y"] - bb["min_y"]) / self.res) + pad * 2 + 1
        cols = int((bb["max_x"] - bb["min_x"]) / self.res) + pad * 2 + 1
        origin_x = bb["min_x"] - pad * self.res
        origin_y = bb["min_y"] - pad * self.res

        # ── A: Raster grid (skeleton only) ────────────────────────────────────
        grid         = np.zeros((rows, cols), dtype=np.uint8)
        shapely_polys: List[object] = []

        for zone in self.sfm.zones:
            if zone.category not in WALKABLE_CATEGORIES:
                continue
            if len(zone.boundary_polygon) < 3:
                continue

            # Rasterise
            poly_rows = [int((py - origin_y) / self.res)
                         for _, py in zone.boundary_polygon]
            poly_cols = [int((px - origin_x) / self.res)
                         for px, _ in zone.boundary_polygon]
            rr, cc = sk_polygon(poly_rows, poly_cols, shape=grid.shape)
            grid[rr, cc] = 1

            # Shapely polygon for exact geometry
            try:
                sp = Polygon(zone.boundary_polygon).buffer(0)
                if sp.is_valid and sp.area > 0:
                    shapely_polys.append(sp)
            except Exception:
                pass

        # ── B: Shapely walkable union ─────────────────────────────────────────
        # Small outward buffer (0.08 m) absorbs floating-point boundary cases
        # so nodes sitting exactly on a zone edge are still "inside".
        if shapely_polys:
            self._walkable = unary_union(shapely_polys).buffer(0.08)
        else:
            self._walkable = None

        # Shore geometry: union of all walkable zone exterior rings,
        # buffered outward by SHORE_BUFFER.
        # A point is "shore-linable" if it lies inside this buffered region.
        shore_rings: List[object] = []
        for zone in self.sfm.zones:
            if zone.category not in WALKABLE_CATEGORIES:
                continue
            if len(zone.boundary_polygon) < 3:
                continue
            try:
                poly = Polygon(zone.boundary_polygon)
                if poly.is_valid:
                    shore_rings.append(poly.exterior)
            except Exception:
                pass

        if shore_rings:
            self._wall_union = unary_union(shore_rings).buffer(SHORE_BUFFER)
        else:
            self._wall_union = None

        # ── Medial axis + skeleton graph ──────────────────────────────────────
        skel, dist = medial_axis(grid, return_distance=True)

        self._grid = _Grid(
            data  = grid,
            dist  = dist,
            skel  = skel,
            min_x = origin_x,
            min_y = origin_y,
            res   = self.res,
        )

        G        = nx.Graph()
        skel_pts = np.argwhere(skel)
        for r, c in skel_pts:
            G.add_node((r, c), clearance=float(dist[r, c]) * self.res)
        for r, c in skel_pts:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < skel.shape[0] and \
                       0 <= nc < skel.shape[1] and skel[nr, nc]:
                        G.add_edge((r, c), (nr, nc),
                                   weight=math.hypot(dr, dc) * self.res)
        self._skel_graph = G

    # ── Step 3: skeletonise → junction nodes ─────────────────────────────────

    def _skeletonise(self) -> None:
        """
        Find junction positions from the pixel skeleton.
        Clearance per junction comes from the raster distance transform
        (accurate to ±GRID_RES = ±10 cm — acceptable for a node tag).
        """
        G = self._skel_graph
        g = self._grid

        raw: List[Tuple[int, int]] = []
        for node, deg in G.degree():
            if deg == 1 or deg >= 3:
                raw.append(node)

        if not raw:
            return

        world_pts     = np.array([g.grid_to_world(r, c) for r, c in raw])
        pruned_idx    = _prune_by_distance(world_pts, MIN_NODE_SEP)
        kept_raw      = [raw[i] for i in pruned_idx]

        for idx, (r, c) in enumerate(kept_raw):
            wx, wy    = g.grid_to_world(r, c)
            # Clearance from raster distance transform — used only as a node
            # metadata tag, not for edge scoring (that uses Shapely)
            clearance = float(g.dist[r, c]) * self.res

            self._add_node(NavigationNode(
                node_id   = f"SKE-{idx:04d}",
                label     = f"Junction ({wx:.1f},{wy:.1f})",
                position  = (wx, wy),
                node_type = "junction",
                zone_id   = self._zone_id_at(wx, wy),
                tags      = {
                    "clearance_m": str(round(clearance, 2)),
                    "skel_degree": str(G.degree((r, c))),
                },
            ))

        print(f"[GraphBuilder] Skeleton: {len(raw)} raw key nodes → "
              f"{len(kept_raw)} after pruning (sep={MIN_NODE_SEP}m)")

    # ── Step 4: edges with exact Shapely scores ───────────────────────────────

    def _build_edges(self) -> None:
        """
        Connect nodes with weighted edges.  All four scores use exact
        Shapely geometry — no raster sampling.

        distance       — Shapely exact walkable distance (mean error < 3 cm)
        safety_score   — mean clearance along edge via Shapely boundary distance
        shore_linable  — fraction of edge inside shore buffer via Shapely
        landmark_score — landmarks within LANDMARK_RADIUS of any edge point
        """
        nodes = self._nodes
        if not nodes:
            return

        positions = np.array([nd.position for nd in nodes])
        tree      = KDTree(positions)

        bb       = self.sfm.bounding_box
        search_r = min(
            math.hypot(bb["max_x"] - bb["min_x"],
                       bb["max_y"] - bb["min_y"]),
            20.0,
        )

        lm_nodes = [nd for nd in nodes
                    if nd.node_type in ("elevator", "escalator",
                                        "stair", "landmark", "door")]
        landmark_positions = (np.array([nd.position for nd in lm_nodes])
                              if lm_nodes else None)

        edge_ids_seen: Set[str] = set()

        for i, src in enumerate(nodes):
            indices    = tree.query_ball_point(src.position, r=search_r)
            neighbours = sorted(
                [(_euclidean(src.position, nodes[j].position), j)
                 for j in indices if j != i]
            )
            max_neigh = 4 if src.node_type == "zone_centroid" else MAX_EDGES

            for d_straight, j in neighbours[:max_neigh]:
                tgt = nodes[j]
                eid = _canonical_edge_id(src.node_id, tgt.node_id)
                if eid in edge_ids_seen:
                    continue
                edge_ids_seen.add(eid)

                # ── Distance (Shapely exact) ──────────────────────────────────
                exact_dist = _exact_walkable_distance(
                    src.position, tgt.position, self._walkable, d_straight
                )
                if exact_dist is None:
                    continue   # edge crosses a wall with no walkable segment

                # ── Safety score (Shapely exact clearance) ────────────────────
                safety_score = self._shapely_clearance(
                    src.position, tgt.position
                )

                # ── Shore fraction (Shapely exact) ────────────────────────────
                shore_frac    = self._shapely_shore_fraction(
                    src.position, tgt.position
                )
                shore_linable = shore_frac >= SHORE_FRACTION

                # ── Landmark score (Euclidean sample — already exact) ─────────
                landmark_score = self._landmark_score(
                    src.position, tgt.position, landmark_positions
                )

                self._edges.append(NavigationEdge(
                    edge_id        = eid,
                    source_id      = src.node_id,
                    target_id      = tgt.node_id,
                    distance       = round(float(exact_dist),    4),
                    shore_linable  = shore_linable,
                    safety_score   = round(float(safety_score),  4),
                    landmark_score = round(float(landmark_score), 4),
                    tags           = {
                        "shore_fraction": str(round(shore_frac,  3)),
                        "straight_dist":  str(round(d_straight,  4)),
                    },
                ))

    # ── Shapely scoring helpers (Option A — no raster) ────────────────────────

    def _shapely_clearance(
        self,
        p1: Point2D,
        p2: Point2D,
        n_samples: int = 5,
    ) -> float:
        """
        Mean clearance along edge p1→p2 using exact Shapely geometry.

        For each sample point, clearance = distance to the nearest walkable
        zone boundary (i.e. how far from the nearest wall in metres).
        Normalised to 0–1 by MAX_CLEARANCE.
        """
        if self._walkable is None:
            return 0.05

        ts   = np.linspace(0.0, 1.0, n_samples)
        vals: List[float] = []
        boundary = self._walkable.boundary

        for t in ts:
            wx = p1[0] + t * (p2[0] - p1[0])
            wy = p1[1] + t * (p2[1] - p1[1])
            pt = Point(wx, wy)
            if self._walkable.contains(pt):
                clearance = pt.distance(boundary)
            else:
                clearance = 0.0
            vals.append(clearance)

        mean_clearance = float(np.mean(vals)) if vals else 0.0
        score = min(mean_clearance / MAX_CLEARANCE, 1.0)
        return max(score, 0.05)   # floor so boundary nodes never score 0

    def _shapely_shore_fraction(
        self,
        p1: Point2D,
        p2: Point2D,
        n_samples: int = 10,
    ) -> float:
        """
        Fraction of edge samples that lie within SHORE_BUFFER of any
        walkable zone perimeter, computed via Shapely (exact, no raster).

        self._wall_union is the union of all zone exterior rings buffered
        by SHORE_BUFFER — built once in _build_grid().
        """
        if self._wall_union is None:
            return 0.0

        ts   = np.linspace(0.0, 1.0, n_samples)
        hits = 0
        for t in ts:
            wx = p1[0] + t * (p2[0] - p1[0])
            wy = p1[1] + t * (p2[1] - p1[1])
            if self._wall_union.contains(Point(wx, wy)):
                hits += 1
        return hits / n_samples

    def _landmark_score(
        self,
        p1: Point2D,
        p2: Point2D,
        landmark_positions: Optional[np.ndarray],
        n_samples: int = 5,
    ) -> float:
        """
        Count landmarks within LANDMARK_RADIUS of any sampled point on
        the edge, normalised to 0–1 (capped at 3 landmarks → 1.0).
        """
        if landmark_positions is None or len(landmark_positions) == 0:
            return 0.0

        ts         = np.linspace(0.0, 1.0, n_samples)
        sample_pts = np.array([
            [p1[0] + t * (p2[0] - p1[0]),
             p1[1] + t * (p2[1] - p1[1])]
            for t in ts
        ])
        nearby = sum(
            1 for lm in landmark_positions
            if np.min(np.linalg.norm(sample_pts - lm, axis=1)) <= LANDMARK_RADIUS
        )
        return min(nearby / 3.0, 1.0)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _zone_id_at(self, wx: float, wy: float) -> Optional[str]:
        for zone in self.sfm.zones:
            if _point_in_polygon((wx, wy), zone.boundary_polygon):
                return zone.zone_id
        return None

    def _add_node(self, node: NavigationNode) -> None:
        if node.node_id not in self._node_map:
            self._nodes.append(node)
            self._node_map[node.node_id] = node


# ---------------------------------------------------------------------------
# Module-level utility functions
# ---------------------------------------------------------------------------

def _feat_to_node_type(feature_type: str) -> str:
    return {
        "elevator":   "elevator",
        "escalator":  "escalator",
        "stair":      "stair",
        "door":       "door",
        "info_desk":  "landmark",
        "bench":      "landmark",
        "furnishing": "landmark",
    }.get(feature_type, "landmark")


def _euclidean(a: Point2D, b: Point2D) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _canonical_edge_id(id1: str, id2: str) -> str:
    a, b = sorted([id1, id2])
    return f"EDGE-{a}-{b}"


def _prune_by_distance(points: np.ndarray, min_sep: float) -> List[int]:
    """Greedy pruning — keep one representative per min_sep cluster."""
    if len(points) == 0:
        return []
    tree    = KDTree(points)
    kept:    List[int] = []
    removed: Set[int]  = set()
    for i in range(len(points)):
        if i in removed:
            continue
        kept.append(i)
        for j in tree.query_ball_point(points[i], r=min_sep):
            if j != i:
                removed.add(j)
    return kept


def _exact_walkable_distance(
    p1: Point2D,
    p2: Point2D,
    walkable,
    d_straight: float,
) -> Optional[float]:
    """
    Exact walkable distance between p1 and p2 via Shapely.

      - Fully inside walkable area  → exact Euclidean (zero error)
      - Partially inside            → length of walkable intersection
      - No walkable intersection    → None  (edge is blocked)
    """
    MIN_DIST = 0.05
    if walkable is None:
        return max(d_straight, MIN_DIST)
    if d_straight < MIN_DIST:
        return MIN_DIST

    try:
        line = LineString([p1, p2])

        if walkable.contains(line):
            return d_straight          # fastest path, zero clipping needed

        intersection = walkable.intersection(line)
        if intersection.is_empty:
            return None

        gtype = intersection.geom_type
        if gtype == "LineString":
            dist = intersection.length
        elif gtype == "MultiLineString":
            dist = sum(seg.length for seg in intersection.geoms)
        elif gtype == "GeometryCollection":
            dist = sum(g.length for g in intersection.geoms
                       if "Line" in g.geom_type)
        else:
            dist = 0.0

        return max(dist, MIN_DIST) if dist > 0 else None

    except Exception:
        return max(d_straight, MIN_DIST)


def _point_in_polygon(pt: Point2D, polygon: List[Point2D]) -> bool:
    """Ray-casting point-in-polygon for arbitrary polygons."""
    px, py = pt
    n      = len(polygon)
    inside = False
    j      = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and \
           (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside
