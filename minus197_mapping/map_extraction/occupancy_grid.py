"""
map_extraction/occupancy_grid.py
---------------------------------
Generates a 2-D Occupancy Grid from a SemanticFloorMap and exports it as
<stem>_occupancy.json — a self-contained file consumed by the perception
module's particle filter.

Cell values
-----------
  0  walkable            — particle can be here, no further check needed
  1  wall / blocked      — particle is penalised / killed, no further check
  2  door threshold      — walkable passage boundary, treated as passable
  3  outside building    — outside all zone polygons, treated as wall
  4  uncertain boundary  — sits on the raster boundary between walkable and
                           wall; the perception module must call the exact
                           Shapely point-in-polygon test to decide

Hybrid strategy (Method 1 — neighbour disagreement)
----------------------------------------------------
After the three rasterisation passes (zones → walls → doors), one extra
pass scans every cell.  Any walkable cell (0) that has at least one wall
cell (1) among its 8 neighbours — or any wall cell touching a walkable
cell — is promoted to UNCERTAIN (4).  Deep-interior cells stay as hard
0 or 1 and are resolved in O(1) by the lookup.  Only the thin boundary
layer needs the exact Shapely test at query time.

The Shapely walkable union is serialised as WKT inside the JSON so the
perception module can load it once and call:
    walkable_poly.contains(Point(wx, wy))
for every uncertain cell it encounters.

Resolution
----------
Default 0.05 m/cell.  For the mall (40 m × 20 m) that is 800 × 400 =
320 000 cells.  The uncertain boundary layer is typically 1–2 cells wide
on each side of every zone edge — a small fraction of total cells.

Perception module usage
-----------------------
    import json, math
    from shapely.geometry import Point
    from shapely import wkt as shapely_wkt

    occ          = json.load(open("mall_occupancy.json"))
    grid         = occ["grid"]           # list[list[int]]  [row][col]
    res          = occ["resolution_m"]
    ox           = occ["origin"]["x"]
    oy           = occ["origin"]["y"]
    walkable_poly = shapely_wkt.loads(occ["walkable_wkt"])

    def world_to_cell(wx, wy):
        col = math.floor((wx - ox) / res)
        row = math.floor((wy - oy) / res)
        return row, col

    def is_walkable(wx, wy):
        row, col = world_to_cell(wx, wy)
        if row < 0 or row >= occ["height_cells"]: return False
        if col < 0 or col >= occ["width_cells"]:  return False
        cell = grid[row][col]
        if cell == 0: return True                          # O(1)
        if cell == 1: return False                         # O(1)
        if cell == 2: return True                          # O(1) door
        if cell == 3: return False                         # O(1)
        # cell == 4: uncertain — exact Shapely test
        return walkable_poly.contains(Point(wx, wy))

    def crosses_wall(x1, y1, x2, y2):
        # Bresenham line walk — if any cell is 1 or 3 → wall crossed
        # if any cell is 4 → use exact Shapely segment intersection
        r1, c1 = world_to_cell(x1, y1)
        r2, c2 = world_to_cell(x2, y2)
        for r, c in _bresenham(r1, c1, r2, c2):
            v = grid[r][c]
            if v in (1, 3):
                return True
            if v == 4:
                # exact check: is the cell centre non-walkable?
                cx = ox + (c + 0.5) * res
                cy = oy + (r + 0.5) * res
                if not walkable_poly.contains(Point(cx, cy)):
                    return True
        return False
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.draw import polygon as sk_polygon

from map_extraction.semantic_floor_map import SemanticFloorMap

# ---------------------------------------------------------------------------
# Cell value constants
# ---------------------------------------------------------------------------

CELL_WALKABLE   = 0   # definitely walkable         — O(1) lookup
CELL_WALL       = 1   # definitely wall / blocked   — O(1) lookup
CELL_DOOR       = 2   # door threshold (passable)   — O(1) lookup
CELL_OUTSIDE    = 3   # outside building boundary   — O(1) lookup
CELL_UNCERTAIN  = 4   # boundary cell               — exact Shapely test

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

DEFAULT_RESOLUTION_M: float = 0.05   # metres per cell
WALL_THICKNESS_M:     float = 0.15   # half-thickness for wall segment buffer
GRID_PAD_M:           float = 0.20   # padding around bounding box

WALKABLE_CATEGORIES = {
    "corridor", "entrance", "exit", "shop",
    "food_court", "restroom", "office", "storage", "unknown",
}

# 8-connectivity structuring element used for neighbour disagreement check
_STRUCT_8 = np.ones((3, 3), dtype=bool)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class OccupancyGridExporter:
    """
    Builds and saves an occupancy grid from a SemanticFloorMap.

    The grid uses a hybrid strategy:
      - Deep walkable / wall cells   → hard 0 / 1  (O(1) lookup)
      - Boundary cells               → uncertain 4  (exact Shapely at query)

    Parameters
    ----------
    sfm         : SemanticFloorMap   — the already-built floor map
    resolution  : float              — metres per cell (default 0.05)

    Usage
    -----
        exporter = OccupancyGridExporter(sfm)
        exporter.build()
        exporter.save("data/outputs/mall_occupancy.json")
    """

    def __init__(self,
                 sfm:        SemanticFloorMap,
                 resolution: float = DEFAULT_RESOLUTION_M):
        self.sfm        = sfm
        self.resolution = resolution

        self._grid:          Optional[np.ndarray] = None
        self._origin_x:      float  = 0.0
        self._origin_y:      float  = 0.0
        self._rows:          int    = 0
        self._cols:          int    = 0
        self._walkable_geom: object = None   # Shapely union — set in build()

    # ── Public build / save ───────────────────────────────────────────────────

    def build(self) -> "OccupancyGridExporter":
        """
        Rasterise the floor map into the hybrid occupancy grid.
        Pass order:
          1. init grid          — fill everything with CELL_OUTSIDE
          2. stamp zones        — fill walkable zone polygons with CELL_WALKABLE
          3. stamp walls        — buffer wall segments with CELL_WALL
          4. stamp doors        — mark door positions with CELL_DOOR
          5. stamp uncertain    — promote boundary cells to CELL_UNCERTAIN
          6. build Shapely union— cached for exact query-time resolution
        Returns self for chaining.
        """
        self._init_grid()
        self._stamp_walkable_zones()
        self._stamp_walls()
        self._stamp_doors()
        self._stamp_uncertain_boundary()    # Method 1 — neighbour disagreement
        self._walkable_geom = self._build_walkable_union()

        n_walkable   = int((self._grid == CELL_WALKABLE).sum())
        n_wall       = int((self._grid == CELL_WALL).sum())
        n_door       = int((self._grid == CELL_DOOR).sum())
        n_outside    = int((self._grid == CELL_OUTSIDE).sum())
        n_uncertain  = int((self._grid == CELL_UNCERTAIN).sum())
        total        = self._rows * self._cols

        print(
            f"[OccupancyGrid] Built {self._rows}×{self._cols} grid "
            f"({self.resolution} m/cell) — floor {self.sfm.floor_label}\n"
            f"  walkable  : {n_walkable:>7}  ({n_walkable/total*100:.1f}%)\n"
            f"  wall      : {n_wall:>7}  ({n_wall/total*100:.1f}%)\n"
            f"  door      : {n_door:>7}\n"
            f"  outside   : {n_outside:>7}  ({n_outside/total*100:.1f}%)\n"
            f"  uncertain : {n_uncertain:>7}  ({n_uncertain/total*100:.1f}%)"
            f"  ← exact Shapely test at query time"
        )
        return self

    def save(self, path: str | Path = "occupancy.json") -> Path:
        """Serialise and write the occupancy grid JSON file."""
        if self._grid is None:
            raise RuntimeError("Call build() before save().")

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self._to_dict(), separators=(",", ":")),
            encoding="utf-8",
        )
        print(f"[OccupancyGrid] Saved → {p.resolve()}")
        return p

    # ── Pass 1: grid initialisation ───────────────────────────────────────────

    def _init_grid(self) -> None:
        """Allocate grid filled with CELL_OUTSIDE; compute dimensions."""
        bb  = self.sfm.bounding_box
        res = self.resolution
        pad = GRID_PAD_M

        self._origin_x = bb["min_x"] - pad
        self._origin_y = bb["min_y"] - pad

        self._cols = math.ceil((bb["max_x"] + pad - self._origin_x) / res) + 1
        self._rows = math.ceil((bb["max_y"] + pad - self._origin_y) / res) + 1

        self._grid = np.full(
            (self._rows, self._cols), CELL_OUTSIDE, dtype=np.uint8
        )

    # ── Pass 2: walkable zones ────────────────────────────────────────────────

    def _stamp_walkable_zones(self) -> None:
        """Fill cells inside walkable zone polygons with CELL_WALKABLE (0)."""
        for zone in self.sfm.zones:
            if zone.category not in WALKABLE_CATEGORIES:
                continue
            poly = zone.boundary_polygon
            if len(poly) < 3:
                continue

            r_coords = [self._world_to_row(py) for _, py in poly]
            c_coords = [self._world_to_col(px) for px, _ in poly]
            rr, cc = sk_polygon(r_coords, c_coords, shape=self._grid.shape)
            self._grid[rr, cc] = CELL_WALKABLE

    # ── Pass 3: walls ─────────────────────────────────────────────────────────

    def _stamp_walls(self) -> None:
        """
        Buffer each wall segment by WALL_THICKNESS_M and stamp CELL_WALL (1).
        Uses point-to-segment distance for accurate non-axis-aligned walls.
        """
        t   = WALL_THICKNESS_M
        res = self.resolution

        for wall in self.sfm.walls:
            sx, sy = wall.start
            ex, ey = wall.end

            min_r = max(0,
                        self._world_to_row(min(sy, ey)) - int(t / res) - 1)
            max_r = min(self._rows - 1,
                        self._world_to_row(max(sy, ey)) + int(t / res) + 1)
            min_c = max(0,
                        self._world_to_col(min(sx, ex)) - int(t / res) - 1)
            max_c = min(self._cols - 1,
                        self._world_to_col(max(sx, ex)) + int(t / res) + 1)

            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    wx, wy = self._cell_to_world(r, c)
                    if _point_to_segment_dist(wx, wy, sx, sy, ex, ey) <= t:
                        self._grid[r, c] = CELL_WALL

    def _stamp_doors(self) -> None:
        """
        Mark door feature positions as CELL_DOOR (2).

        Case A — perimeter doors:
            Cell is CELL_WALL or CELL_OUTSIDE → stamp directly.

        Case B — interior shop-front doors:
            Cell is CELL_WALKABLE but within DOOR_SNAP_M of a wall cell
            → stamp as CELL_DOOR so the perception module knows this is
            a zone-boundary passage, not open corridor.

        Walkable cells far from any wall are left untouched.
        """
        DOOR_SNAP_M = 0.30
        snap_cells  = max(1, int(DOOR_SNAP_M / self.resolution))

        for feat in self.sfm.features:
            if feat.feature_type != "door":
                continue
            px, py = feat.position
            r = self._world_to_row(py)
            c = self._world_to_col(px)
            if not (0 <= r < self._rows and 0 <= c < self._cols):
                continue

            cell = self._grid[r, c]

            # Case A: perimeter door sitting on wall or outside cell
            if cell in (CELL_WALL, CELL_OUTSIDE):
                self._grid[r, c] = CELL_DOOR

            # Case B: interior shop-front door inside walkable zone
            elif cell == CELL_WALKABLE:
                r0 = max(0, r - snap_cells)
                r1 = min(self._rows - 1, r + snap_cells)
                c0 = max(0, c - snap_cells)
                c1 = min(self._cols - 1, c + snap_cells)
                neighbourhood = self._grid[r0:r1+1, c0:c1+1]
                near_wall = np.any(
                    (neighbourhood == CELL_WALL) |
                    (neighbourhood == CELL_OUTSIDE)
                )
                if near_wall:
                    self._grid[r, c] = CELL_DOOR
    # ── Pass 5: uncertain boundary (Method 1 — neighbour disagreement) ────────

    def _stamp_uncertain_boundary(self) -> None:
        """
        Promote boundary cells to CELL_UNCERTAIN (4).

        A cell is uncertain when it sits on the raster boundary between
        walkable and non-walkable space — i.e. when the rasteriser may have
        assigned the wrong hard value.

        Method 1 — neighbour disagreement:
          Any CELL_WALKABLE that has at least one CELL_WALL neighbour
          (in the 8-connected sense) is uncertain.
          Any CELL_WALL that has at least one CELL_WALKABLE neighbour
          is also uncertain.

        CELL_DOOR cells (2) are intentionally excluded — they are already
        a soft "passable boundary" marker and are always treated as walkable.
        CELL_OUTSIDE cells deep away from the building edge are left as 3.

        At query time the perception module resolves uncertain cells with:
            walkable_poly.contains(Point(wx, wy))
        using the Shapely WKT stored in the JSON.
        """
        walkable_mask = (self._grid == CELL_WALKABLE)
        wall_mask     = (self._grid == CELL_WALL)

        # Expand each mask by one cell in all 8 directions
        walkable_dilated = binary_dilation(walkable_mask, _STRUCT_8)
        wall_dilated     = binary_dilation(wall_mask,     _STRUCT_8)

        # Boundary = walkable cell touching a wall, OR wall cell touching walkable
        boundary = (walkable_mask & wall_dilated) | (wall_mask & walkable_dilated)

        self._grid[boundary] = CELL_UNCERTAIN

    # ── Shapely walkable union ────────────────────────────────────────────────

    def _build_walkable_union(self) -> Optional[object]:
        """
        Build the Shapely union of all walkable zone polygons.

        This geometry is used at query time to resolve CELL_UNCERTAIN cells
        with an exact point-in-polygon test.  It is serialised as WKT in the
        JSON output so the perception module only needs to load it once.
        """
        try:
            from shapely.geometry import Polygon
            from shapely.ops import unary_union

            polys: List[object] = []
            for zone in self.sfm.zones:
                if zone.category not in WALKABLE_CATEGORIES:
                    continue
                if len(zone.boundary_polygon) < 3:
                    continue
                try:
                    p = Polygon(zone.boundary_polygon).buffer(0)
                    if p.is_valid and p.area > 0:
                        polys.append(p)
                except Exception:
                    pass

            if not polys:
                return None
            return unary_union(polys)

        except ImportError:
            return None

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _world_to_row(self, wy: float) -> int:
        return int((wy - self._origin_y) / self.resolution)

    def _world_to_col(self, wx: float) -> int:
        return int((wx - self._origin_x) / self.resolution)

    def _cell_to_world(self, r: int, c: int) -> Tuple[float, float]:
        """Return the world-space centre of cell (r, c)."""
        wx = self._origin_x + (c + 0.5) * self.resolution
        wy = self._origin_y + (r + 0.5) * self.resolution
        return wx, wy

    # ── Serialisation ─────────────────────────────────────────────────────────

    def _to_dict(self) -> dict:
        bb = self.sfm.bounding_box

        # Serialise Shapely walkable union as WKT
        walkable_wkt = ""
        if self._walkable_geom is not None:
            try:
                walkable_wkt = self._walkable_geom.wkt
            except Exception:
                pass

        return {
            # ── Metadata ──────────────────────────────────────────────────────
            "floor_label":  self.sfm.floor_label,
            "source_file":  self.sfm.source_file,
            "resolution_m": self.resolution,
            "origin": {
                "x": round(self._origin_x, 6),
                "y": round(self._origin_y, 6),
            },
            "width_cells":  int(self._cols),
            "height_cells": int(self._rows),
            "bounding_box": {k: float(v) for k, v in bb.items()},

            # ── Cell legend ───────────────────────────────────────────────────
            "cell_legend": {
                "0": "walkable",
                "1": "wall",
                "2": "door_threshold",
                "3": "outside_building",
                "4": "uncertain_boundary",
            },

            # ── Cell counts ───────────────────────────────────────────────────
            "cell_counts": {
                "walkable":  int((self._grid == CELL_WALKABLE).sum()),
                "wall":      int((self._grid == CELL_WALL).sum()),
                "door":      int((self._grid == CELL_DOOR).sum()),
                "outside":   int((self._grid == CELL_OUTSIDE).sum()),
                "uncertain": int((self._grid == CELL_UNCERTAIN).sum()),
            },

            # ── Grid (row-major, row 0 = min Y) ──────────────────────────────
            "grid": self._grid.tolist(),

            # ── Exact geometry for uncertain cell resolution ───────────────────
            # Load once in the perception module:
            #   from shapely import wkt as shapely_wkt
            #   poly = shapely_wkt.loads(occ["walkable_wkt"])
            "walkable_wkt": walkable_wkt,

            # ── Coordinate frame declaration ──────────────────────────────────
            # All coordinates in this file are in this frame.
            # Perception module must align sensor readings to this frame
            # before calling is_walkable() or crosses_wall().
            "coordinate_frame": {
                "units": "metres",
                "source": "IFC project coordinate system",
                "x_axis": "IFC project X axis",
                "y_axis": "IFC project Y axis",
                "origin_description": (
                    "grid cell (row=0, col=0) top-left corner — "
                    "bounding box minimum minus padding"
                ),
            "grid_origin_x": round(self._origin_x, 6),
            "grid_origin_y": round(self._origin_y, 6),
            "building_min_x": float(bb["min_x"]),
            "building_min_y": float(bb["min_y"]),
            "building_max_x": float(bb["max_x"]),
            "building_max_y": float(bb["max_y"]),
            },

            # ── Query hints for the perception module ─────────────────────────
            "query_hint": {
                "world_to_cell": (
                    "col = floor((wx - origin.x) / resolution_m); "
                    "row = floor((wy - origin.y) / resolution_m)"
                ),
                "is_walkable": (
                    "cell=grid[row][col]; "
                    "0→True; 1→False; 2→True; 3→False; "
                    "4→walkable_poly.contains(Point(wx, wy))"
                ),
                "crosses_wall": (
                    "Bresenham line from cell(x1,y1) to cell(x2,y2); "
                    "if any cell is 1 or 3 → True; "
                    "if any cell is 4 → exact Shapely segment test"
                ),
            },
        }


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

def _point_to_segment_dist(px: float, py: float,
                            ax: float, ay: float,
                            bx: float, by: float) -> float:
    """
    Exact distance from point P to line segment AB.
    Used to buffer wall segments into the raster grid.
    """
    abx, aby = bx - ax, by - ay
    ab_sq = abx * abx + aby * aby
    if ab_sq == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_sq))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))
