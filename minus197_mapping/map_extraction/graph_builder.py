"""
map_extraction/graph_builder.py  —  Sprint 3 + 4
-------------------------------------------------
Converts a SemanticFloorMap into a FloorGraph (navigation graph).

Pipeline
--------
  _place_feature_nodes()      Place door / elevator / stair / landmark nodes
  _place_zone_centroid_nodes()Place one centroid per named zone
  _skeletonise()              Sprint 3 — medial-axis skeleton of walkable grid
                               → pruned junction nodes along corridor spines
  _build_edges()              Sprint 4 — connect nodes with weighted edges
                               (distance, safety_score, shore_linable,
                                landmark_score)

Skeletonisation detail (Sprint 3)
----------------------------------
  1. Rasterise every walkable zone polygon onto a binary grid at
     GRID_RES metres/cell using skimage.draw.polygon.
  2. Run skimage.morphology.medial_axis() to get the skeleton and
     the distance-transform (clearance in cells).
  3. Build a pixel-level skeleton graph (networkx) connecting
     8-connected neighbours.
  4. Find raw key nodes: degree-1 (endpoints) and degree≥3 (branches).
  5. Prune: keep only nodes that are ≥ MIN_NODE_SEP metres apart
     (cluster nearby raw nodes → single representative via KDTree).
  6. Convert surviving grid coordinates → world metres.
  7. Create NavigationNode(node_type='junction') for each.

Edge construction detail (Sprint 4)
--------------------------------------
  1. For every pair of nodes (u, v):
       - If they share a zone or are adjacent zones, compute Euclidean
         distance.
       - Walk the skeleton path between them (Dijkstra on pixel graph)
         to get the real corridor distance and min clearance.
  2. Tag each edge:
       shore_linable  — True when ≥ SHORE_FRACTION of the path runs
                        within SHORE_BUFFER m of a shore-linable wall.
       safety_score   — normalised mean clearance along path (0–1).
       landmark_score — count of landmark nodes within LANDMARK_RADIUS m
                        of any point on the path, normalised to 0–1.
  3. Only add edges where a skeleton path exists (no walls crossed).
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
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from shared.types import FloorGraph, NavigationEdge, NavigationNode, Point2D
from map_extraction.semantic_floor_map import SemanticFloorMap, Feature, Zone

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

GRID_RES       = 0.10   # metres per raster cell  (10 cm)
MIN_NODE_SEP   = 1.50   # min distance (m) between pruned skeleton nodes
SHORE_BUFFER   = 0.80   # m — path within this of a wall → shore-linable cell
SHORE_FRACTION = 0.40   # fraction of path cells that must be shore-linable
LANDMARK_RADIUS = 3.0   # m — landmark within this radius boosts landmark_score
MAX_SKELETON_EDGES = 6  # max edges per skeleton junction node

# Zone categories treated as walkable floor space
WALKABLE_CATEGORIES: Set[str] = {
    "corridor", "entrance", "exit", "shop",
    "food_court", "restroom", "office", "storage", "unknown",
}

# ---------------------------------------------------------------------------
# Internal grid-coordinate helpers
# ---------------------------------------------------------------------------

@dataclass
class _Grid:
    """Holds the raster grid and coordinate conversion helpers."""
    data:    np.ndarray          # shape (rows, cols), dtype uint8
    dist:    np.ndarray          # medial-axis distance transform (cells)
    skel:    np.ndarray          # skeleton boolean grid
    min_x:   float
    min_y:   float
    res:     float

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        col = int((wx - self.min_x) / self.res)
        row = int((wy - self.min_y) / self.res)
        return row, col

    def grid_to_world(self, row: int, col: int) -> Point2D:
        wx = self.min_x + col * self.res
        wy = self.min_y + row * self.res
        return (round(wx, 3), round(wy, 3))

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
    sfm           : SemanticFloorMap
    grid_resolution : float  metres per raster cell (default GRID_RES)
    """

    def __init__(self, sfm: SemanticFloorMap,
                 grid_resolution: float = GRID_RES):
        self.sfm  = sfm
        self.res  = grid_resolution

        self._nodes:    List[NavigationNode] = []
        self._edges:    List[NavigationEdge] = []
        self._node_map: Dict[str, NavigationNode] = {}
        self._grid:     Optional[_Grid] = None
        self._skel_graph: Optional[nx.Graph] = None
        self._shore_grid: Optional[np.ndarray] = None   # bool grid
        self._walkable:   Optional[object]     = None   # Shapely polygon

    # ── Public ────────────────────────────────────────────────────────────────

    def build(self) -> FloorGraph:
        print("[GraphBuilder] Placing feature and zone centroid nodes ...")
        self._place_feature_nodes()
        self._place_zone_centroid_nodes()

        print("[GraphBuilder] Rasterising walkable zones ...")
        self._build_grid()

        print("[GraphBuilder] Running medial-axis skeletonisation ...")
        self._skeletonise()

        print("[GraphBuilder] Building edges ...")
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
            ntype = _feat_to_node_type(feat.feature_type)
            self._add_node(NavigationNode(
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

    # ── Step 2: rasterise ────────────────────────────────────────────────────

    def _build_grid(self) -> None:
        bb  = self.sfm.bounding_box
        pad = 2  # extra cells on each side

        rows = int((bb["max_y"] - bb["min_y"]) / self.res) + pad * 2 + 1
        cols = int((bb["max_x"] - bb["min_x"]) / self.res) + pad * 2 + 1

        origin_x = bb["min_x"] - pad * self.res
        origin_y = bb["min_y"] - pad * self.res

        grid = np.zeros((rows, cols), dtype=np.uint8)

        shapely_polys = []
        for zone in self.sfm.zones:
            if zone.category not in WALKABLE_CATEGORIES:
                continue
            poly_rows = [int((py - origin_y) / self.res)
                         for _, py in zone.boundary_polygon]
            poly_cols = [int((px - origin_x) / self.res)
                         for px, _ in zone.boundary_polygon]
            rr, cc = sk_polygon(poly_rows, poly_cols, shape=grid.shape)
            grid[rr, cc] = 1
            # Also build exact Shapely polygon for distance calculation
            if len(zone.boundary_polygon) >= 3:
                try:
                    sp = Polygon(zone.boundary_polygon).buffer(0)
                    if sp.is_valid and sp.area > 0:
                        shapely_polys.append(sp)
                except Exception:
                    pass

        # Shapely union of all walkable zones — used for exact distance
        # measurement and line-of-sight checks (no pixel error)
        if shapely_polys:
            self._walkable = unary_union(shapely_polys).buffer(0.08)
        else:
            self._walkable = None

        # Shore-linable grid
        # Mark cells within SHORE_BUFFER of:
        #   (a) walls already tagged shore_linable by SemanticFloorMapBuilder, AND
        #   (b) the perimeter edges of every walkable zone polygon
        # This ensures the main shop floor, entrance, and storage area walls are
        # all captured — not just corridor-adjacent walls.
        shore     = np.zeros((rows, cols), dtype=bool)
        buf_cells = int(SHORE_BUFFER / self.res) + 1

        def _mark_segment(x0: float, y0: float, x1: float, y1: float) -> None:
            length = math.hypot(x1 - x0, y1 - y0)
            steps  = max(int(length / self.res * 2), 2)
            for t in np.linspace(0, 1, steps):
                wx = x0 + t * (x1 - x0)
                wy = y0 + t * (y1 - y0)
                wr = int((wy - origin_y) / self.res)
                wc = int((wx - origin_x) / self.res)
                for dr in range(-buf_cells, buf_cells + 1):
                    for dc in range(-buf_cells, buf_cells + 1):
                        nr, nc = wr + dr, wc + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            shore[nr, nc] = True

        # (a) explicit shore-linable walls from semantic map
        for wall in self.sfm.walls:
            if wall.shore_linable:
                _mark_segment(wall.start[0], wall.start[1],
                              wall.end[0],   wall.end[1])

        # (b) perimeter edges of every walkable zone polygon
        for zone in self.sfm.zones:
            if zone.category not in WALKABLE_CATEGORIES:
                continue
            poly = zone.boundary_polygon
            n_pts = len(poly)
            for k in range(n_pts):
                ax, ay = poly[k]
                bx, by = poly[(k + 1) % n_pts]
                _mark_segment(ax, ay, bx, by)

        skel, dist = medial_axis(grid, return_distance=True)

        self._grid = _Grid(
            data  = grid,
            dist  = dist,
            skel  = skel,
            min_x = origin_x,
            min_y = origin_y,
            res   = self.res,
        )
        self._shore_grid = shore

        # Build pixel-level skeleton graph
        G = nx.Graph()
        skel_pts = np.argwhere(skel)
        for r, c in skel_pts:
            G.add_node((r, c),
                       clearance=float(dist[r, c]) * self.res,
                       shore=bool(shore[r, c]))
        for r, c in skel_pts:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if skel[nr, nc]:
                        w = math.hypot(dr, dc) * self.res
                        G.add_edge((r, c), (nr, nc), weight=w)
        self._skel_graph = G

    # ── Step 3: skeletonise → junction nodes (Sprint 3) ──────────────────────

    def _skeletonise(self) -> None:
        G   = self._skel_graph
        g   = self._grid

        # Raw key nodes: endpoints (deg 1) and branch points (deg ≥ 3)
        raw: List[Tuple[int, int]] = []
        for node, deg in G.degree():
            if deg == 1 or deg >= 3:
                raw.append(node)

        if not raw:
            return

        # Convert to world coords
        world_pts = np.array([g.grid_to_world(r, c) for r, c in raw])

        # Cluster nearby points → keep one representative per cluster
        pruned_indices = _prune_by_distance(world_pts, MIN_NODE_SEP)
        kept_raw = [raw[i] for i in pruned_indices]

        for idx, (r, c) in enumerate(kept_raw):
            wx, wy    = g.grid_to_world(r, c)
            clearance = float(g.dist[r, c]) * self.res
            deg       = G.degree((r, c))
            ntype     = "junction"

            self._add_node(NavigationNode(
                node_id   = f"SKE-{idx:04d}",
                label     = f"Junction ({wx:.1f},{wy:.1f})",
                position  = (wx, wy),
                node_type = ntype,
                zone_id   = self._zone_id_at(wx, wy),
                tags      = {
                    "clearance_m": str(round(clearance, 2)),
                    "skel_degree": str(deg),
                },
            ))

        print(f"[GraphBuilder] Skeleton: {len(raw)} raw key nodes → "
              f"{len(kept_raw)} after pruning (sep={MIN_NODE_SEP}m)")

    # ── Step 4: build edges with exact distances (Sprint 4) ──────────────────

    def _build_edges(self) -> None:
        """
        Connect nodes with weighted edges using EXACT geometry distances.

        Distance strategy (no raster error):
          1. Build a Shapely union of all walkable zone polygons.
          2. For each candidate node pair, draw a straight LineString.
          3. If the line is fully inside the walkable polygon →
               distance = exact Euclidean (no raster error at all).
          4. If the line crosses a wall/boundary →
               distance = length of the walkable intersection segment
               (Shapely clips the line to the walkable area exactly).
          5. If no walkable intersection exists → skip this edge.

        Shore / safety / landmark scores still use the raster grid
        (they measure corridor properties, not distances — pixel-level
        accuracy is fine for those).
        """
        G       = self._skel_graph
        g       = self._grid
        walkable = self._walkable

        nodes = self._nodes
        if not nodes:
            return

        positions = np.array([nd.position for nd in nodes])
        tree      = KDTree(positions)

        bb       = self.sfm.bounding_box
        max_r    = math.hypot(bb["max_x"] - bb["min_x"],
                              bb["max_y"] - bb["min_y"])
        search_r = min(max_r, 20.0)

        # Build landmark position array for scoring
        landmark_positions = None
        lm_nodes = [nd for nd in nodes
                    if nd.node_type in ("elevator", "escalator",
                                        "stair", "landmark", "door")]
        if lm_nodes:
            landmark_positions = np.array([nd.position for nd in lm_nodes])

        edge_ids_seen: Set[str] = set()

        for i, src in enumerate(nodes):
            indices = tree.query_ball_point(src.position, r=search_r)

            neighbours = sorted(
                [(  _euclidean(src.position, nodes[j].position), j)
                 for j in indices if j != i]
            )

            max_neigh = MAX_SKELETON_EDGES
            if src.node_type == "zone_centroid":
                max_neigh = 4

            for d_straight, j in neighbours[:max_neigh]:
                tgt = nodes[j]
                eid = _canonical_edge_id(src.node_id, tgt.node_id)
                if eid in edge_ids_seen:
                    continue
                edge_ids_seen.add(eid)

                # ── Exact distance via Shapely ──────────────────────────────
                exact_dist = _exact_walkable_distance(
                    src.position, tgt.position, walkable, d_straight
                )
                if exact_dist is None:
                    # No walkable path between these nodes — skip edge
                    continue

                # ── Safety score: mean clearance along path ─────────────────
                # Sample the raster distance-transform along the straight
                # line — gives corridor width estimate without skeleton detour
                mean_clearance = self._clearance_at_world(
                    src.position, tgt.position, g
                )
                MAX_CLEARANCE = 2.0
                safety_score  = min(mean_clearance / MAX_CLEARANCE, 1.0)
                # Nodes sitting exactly on zone boundaries get low clearance
                # from the fallback — clamp minimum to 0.05 so score > 0
                safety_score  = max(safety_score, 0.05)

                # ── Shore score: fraction of straight-line path near wall ───
                shore_frac    = self._shore_fraction_linear(
                    src.position, tgt.position
                )
                shore_linable = shore_frac >= SHORE_FRACTION

                # ── Landmark score: landmarks near any point on the edge ─────
                landmark_score = self._landmark_score(
                    src.position, tgt.position, landmark_positions
                )

                self._edges.append(NavigationEdge(
                    edge_id        = eid,
                    source_id      = src.node_id,
                    target_id      = tgt.node_id,
                    distance       = round(float(exact_dist), 4),
                    shore_linable  = shore_linable,
                    safety_score   = round(safety_score, 4),
                    landmark_score = round(landmark_score, 4),
                    tags           = {
                        "shore_fraction":  str(round(shore_frac, 3)),
                        "mean_clearance":  str(round(mean_clearance, 3)),
                        "straight_dist":   str(round(d_straight, 4)),
                    },
                ))

    # ── Shore / clearance / landmark helpers ─────────────────────────────────

    def _landmark_score(
        self,
        p1: Point2D,
        p2: Point2D,
        landmark_positions: Optional[np.ndarray],
        n_samples: int = 5,
    ) -> float:
        """
        Sample n_samples evenly-spaced points along the edge (p1→p2),
        count how many landmarks lie within LANDMARK_RADIUS of ANY
        sample point, then normalise to 0–1 (capped at 3 unique
        landmarks → 1.0).

        Sampling the full path rather than only the midpoint prevents
        long edges from missing landmarks that sit near one end.
        """
        if landmark_positions is None or len(landmark_positions) == 0:
            return 0.0

        # Sample points along the edge
        ts = np.linspace(0.0, 1.0, n_samples)
        sample_pts = np.array([
            [p1[0] + t * (p2[0] - p1[0]),
             p1[1] + t * (p2[1] - p1[1])]
            for t in ts
        ])

        # For each landmark, check if it is within radius of ANY sample
        nearby_count = 0
        for lm_pos in landmark_positions:
            dists = np.linalg.norm(sample_pts - lm_pos, axis=1)
            if np.min(dists) <= LANDMARK_RADIUS:
                nearby_count += 1

        return min(nearby_count / 3.0, 1.0)

    def _clearance_at_world(
        self,
        p1: Point2D,
        p2: Point2D,
        g: _Grid,
        n_samples: int = 5,
    ) -> float:
        """
        Estimate mean clearance for an edge that had no skeleton path
        by sampling the distance-transform at n_samples points along
        the straight line p1→p2, clamping to grid bounds.
        Returns 0.0 if all samples fall outside the walkable grid.
        """
        ts       = np.linspace(0.0, 1.0, n_samples)
        vals: List[float] = []
        for t in ts:
            wx = p1[0] + t * (p2[0] - p1[0])
            wy = p1[1] + t * (p2[1] - p1[1])
            r, c = g.world_to_grid(wx, wy)
            r = max(0, min(r, g.shape[0] - 1))
            c = max(0, min(c, g.shape[1] - 1))
            if g.data[r, c]:                     # inside walkable area
                vals.append(float(g.dist[r, c]) * self.res)
        return float(np.mean(vals)) if vals else 0.0

    def _shore_fraction_linear(
        self,
        p1: Point2D,
        p2: Point2D,
        n_samples: int = 10,
    ) -> float:
        """
        Estimate shore fraction for an edge that had no skeleton path
        by sampling n_samples points along p1→p2 and checking the
        shore grid at each.
        """
        g = self._grid
        ts   = np.linspace(0.0, 1.0, n_samples)
        hits = 0
        for t in ts:
            wx = p1[0] + t * (p2[0] - p1[0])
            wy = p1[1] + t * (p2[1] - p1[1])
            r, c = g.world_to_grid(wx, wy)
            r = max(0, min(r, g.shape[0] - 1))
            c = max(0, min(c, g.shape[1] - 1))
            if self._shore_grid[r, c]:
                hits += 1
        return hits / n_samples

    def _zone_id_at(self, wx: float, wy: float) -> Optional[str]:
        for zone in self.sfm.zones:
            if _point_in_polygon((wx, wy), zone.boundary_polygon):
                return zone.zone_id
        return None

    # ── Node registry ─────────────────────────────────────────────────────────

    def _add_node(self, node: NavigationNode) -> None:
        if node.node_id not in self._node_map:
            self._nodes.append(node)
            self._node_map[node.node_id] = node
        for zone in self.sfm.zones:
            if _point_in_polygon((wx, wy), zone.boundary_polygon):
                return zone.zone_id
        return None

    # ── Node registry ─────────────────────────────────────────────────────────

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
    """Consistent edge ID regardless of direction."""
    a, b = sorted([id1, id2])
    return f"EDGE-{a}-{b}"


def _prune_by_distance(points: np.ndarray, min_sep: float) -> List[int]:
    """
    Greedy pruning: keep a point only if it is ≥ min_sep from all
    already-kept points. Returns indices of kept points.
    """
    if len(points) == 0:
        return []
    tree    = KDTree(points)
    kept    = []
    removed: Set[int] = set()
    for i in range(len(points)):
        if i in removed:
            continue
        kept.append(i)
        # Remove all points within min_sep of this one
        neighbours = tree.query_ball_point(points[i], r=min_sep)
        for j in neighbours:
            if j != i:
                removed.add(j)
    return kept


def _exact_walkable_distance(
    p1: Point2D,
    p2: Point2D,
    walkable,        # Shapely polygon or None
    d_straight: float,
) -> Optional[float]:
    """
    Return the exact walkable distance between p1 and p2.

    Rules:
      1. If the straight line p1→p2 is fully inside the walkable polygon
         → return exact Euclidean distance (zero raster error).
      2. If the line partially intersects the walkable area
         → return the length of the walkable intersection segment.
         This handles edges crossing zone boundaries.
      3. If no walkable intersection → return None (edge is blocked).
      4. If walkable polygon is not available → return d_straight as fallback.

    Minimum distance floor of 0.05 m prevents division-by-zero downstream.
    """
    MIN_DIST = 0.05
    if walkable is None:
        return max(d_straight, MIN_DIST)

    if d_straight < MIN_DIST:
        return MIN_DIST

    try:
        line = LineString([p1, p2])

        # Fast path: fully inside → exact Euclidean, no clipping needed
        if walkable.contains(line):
            return d_straight

        # Partial intersection: clip to walkable and measure
        intersection = walkable.intersection(line)
        if intersection.is_empty:
            return None

        # Sum all walkable sub-segments
        geom_type = intersection.geom_type
        if geom_type == "LineString":
            dist = intersection.length
        elif geom_type == "MultiLineString":
            dist = sum(seg.length for seg in intersection.geoms)
        elif geom_type == "GeometryCollection":
            dist = sum(
                g.length for g in intersection.geoms
                if "Line" in g.geom_type
            )
        else:
            dist = 0.0

        return max(dist, MIN_DIST) if dist > 0 else None

    except Exception:
        # Shapely geometry error — fall back to Euclidean
        return max(d_straight, MIN_DIST)


def _point_in_polygon(pt: Point2D, polygon: List[Point2D]) -> bool:
    """Ray-casting point-in-polygon."""
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
