"""
perception_map.py
-----------------
Standalone helper delivered by the mapping module to the perception module.

Drop this file alongside the _occupancy.json files.
No dependency on any mapping module internal code.

Dependencies:
    pip install shapely numpy

Usage:
    from perception_map import OccupancyMap

    # Load once at startup — one instance per floor
    occ_map = OccupancyMap("data/outputs/mall_L1_occupancy.json")

    # Check if a world position is walkable
    occ_map.is_walkable(14.5, 7.2)              # → True or False

    # Check if a motion segment crosses a wall
    occ_map.crosses_wall(14.5, 7.2, 14.8, 7.5) # → True or False

Coordinates:
    All coordinates are in metres in the IFC project coordinate system.
    See the coordinate_frame block in the occupancy JSON for axis directions
    and building extents. Align your sensor readings to this frame before
    calling any method.
"""

import json
import math
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from shapely.geometry import Point
from shapely import wkt as shapely_wkt


class OccupancyMap:
    """
    Loads an _occupancy.json produced by the mapping module and exposes
    walkability queries in world coordinates (metres).

    One instance per floor. Load at startup and keep in memory.
    The Shapely polygon is parsed once in __init__ and reused forever —
    never reload it per query.
    """

    # Cell value constants — same as occupancy_grid.py
    WALKABLE  = 0
    WALL      = 1
    DOOR      = 2
    OUTSIDE   = 3
    UNCERTAIN = 4

    def __init__(self, occ_path: str | Path):
        with open(occ_path, encoding="utf-8") as f:
            occ = json.load(f)

        self.floor_label = occ["floor_label"]
        self.resolution  = occ["resolution_m"]          # metres per cell
        self.origin_x    = occ["origin"]["x"]            # world x of col 0
        self.origin_y    = occ["origin"]["y"]            # world y of row 0
        self.width       = occ["width_cells"]
        self.height      = occ["height_cells"]
        self.bounding_box = occ.get("bounding_box", {})
        self.coordinate_frame = occ.get("coordinate_frame", {})

        # Grid as numpy array — uint8, shape (height, width)
        self.grid = np.array(occ["grid"], dtype=np.uint8)

        # Shapely walkable polygon — parsed ONCE, reused for all uncertain cells
        wkt = occ.get("walkable_wkt", "")
        self._walkable_poly = shapely_wkt.loads(wkt) if wkt else None

        print(
            f"[OccupancyMap] Loaded floor {self.floor_label} — "
            f"{self.width}×{self.height} cells @ {self.resolution} m/cell"
        )

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def world_to_cell(self, wx: float, wy: float) -> Tuple[int, int]:
        """
        Convert world coordinates (metres) to grid (row, col).
        Row 0 = minimum Y in the building.
        Col 0 = minimum X in the building.
        """
        col = math.floor((wx - self.origin_x) / self.resolution)
        row = math.floor((wy - self.origin_y) / self.resolution)
        return row, col

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Return the world coordinate of the centre of grid cell (row, col)."""
        wx = self.origin_x + (col + 0.5) * self.resolution
        wy = self.origin_y + (row + 0.5) * self.resolution
        return wx, wy

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    # ── Primary queries ───────────────────────────────────────────────────────

    def is_walkable(self, wx: float, wy: float) -> bool:
        """
        Returns True if the world coordinate (wx, wy) is in walkable space.

        Handles all five cell types:
          0 walkable          → True  (O(1))
          1 wall              → False (O(1))
          2 door threshold    → True  (O(1), doors are passable)
          3 outside building  → False (O(1))
          4 uncertain boundary→ exact Shapely point-in-polygon test

        Parameters
        ----------
        wx, wy : float
            World coordinates in metres, IFC project coordinate system.

        Returns
        -------
        bool
        """
        row, col = self.world_to_cell(wx, wy)

        if not self.in_bounds(row, col):
            return False

        cell = int(self.grid[row, col])

        if cell == self.WALKABLE:  return True
        if cell == self.WALL:      return False
        if cell == self.DOOR:      return True
        if cell == self.OUTSIDE:   return False

        # cell == UNCERTAIN (4) — raster cannot decide, use exact geometry
        if self._walkable_poly is None:
            return False
        return self._walkable_poly.contains(Point(wx, wy))

    def crosses_wall(self,
                     x1: float, y1: float,
                     x2: float, y2: float) -> bool:
        """
        Returns True if the straight-line motion from (x1,y1) to (x2,y2)
        passes through any wall or outside cell.

        Uses a Bresenham line walk — checks every grid cell the path
        crosses, not just the start and end positions. This correctly
        detects walls thinner than the motion step distance.

        For uncertain cells (4) encountered along the path, uses an exact
        Shapely test on the cell centre.

        Parameters
        ----------
        x1, y1 : float  start position in world metres
        x2, y2 : float  end position in world metres

        Returns
        -------
        bool
        """
        r1, c1 = self.world_to_cell(x1, y1)
        r2, c2 = self.world_to_cell(x2, y2)

        for r, c in self._bresenham(r1, c1, r2, c2):
            if not self.in_bounds(r, c):
                return True   # outside grid = outside building = wall

            cell = int(self.grid[r, c])

            if cell in (self.WALL, self.OUTSIDE):
                return True   # definite wall — stop immediately

            if cell == self.UNCERTAIN:
                # Raster cannot decide — test the cell centre exactly
                cx, cy = self.cell_to_world(r, c)
                if self._walkable_poly is None:
                    return True
                if not self._walkable_poly.contains(Point(cx, cy)):
                    return True

            # WALKABLE, DOOR → continue walking

        return False

    def cell_value(self, wx: float, wy: float) -> Optional[int]:
        """
        Return the raw cell value (0–4) at world position (wx, wy).
        Returns None if out of bounds.
        Useful for debugging — not needed for normal particle filter use.
        """
        row, col = self.world_to_cell(wx, wy)
        if not self.in_bounds(row, col):
            return None
        return int(self.grid[row, col])

    # ── Internal: Bresenham line walk ─────────────────────────────────────────

    @staticmethod
    def _bresenham(r1: int, c1: int, r2: int, c2: int):
        """
        Yields every (row, col) cell on the straight line from
        (r1, c1) to (r2, c2), inclusive of both endpoints.
        """
        dr = abs(r2 - r1);  sr = 1 if r2 > r1 else -1
        dc = abs(c2 - c1);  sc = 1 if c2 > c1 else -1
        err = dr - dc
        r, c = r1, c1
        while True:
            yield r, c
            if r == r2 and c == c2:
                break
            e2 = 2 * err
            if e2 > -dc: err -= dc; r += sr
            if e2 <  dr: err += dr; c += sc