"""
verify_door.py
-------------------
Quick check: do the 4 missed interior shop-front doors have
wall/outside cells within snapping distance on the occupancy grid?

Run from inside minus197_mapping/:
    python verify_door.py
"""

import json
import numpy as np
from pathlib import Path

OCC_PATH = Path("data/outputs/20201022mapping IFC4 Convenience store_occupancy.json")
SNAP_M   = 0.30   # same value as the proposed fix

with open(OCC_PATH) as f:
    occ = json.load(f)

grid       = np.array(occ["grid"], dtype=np.uint8)
resolution = occ["resolution_m"]   # 0.05
snap_cells = max(1, int(SNAP_M / resolution))

print(f"Grid shape      : {grid.shape}  (rows × cols)")
print(f"Resolution      : {resolution} m/cell")
print(f"Snap radius     : {SNAP_M} m = {snap_cells} cells")
print(f"Cell legend     : 0=walkable  1=wall  2=door  3=outside  4=uncertain")
print()

missed = [
    (132, 619, "door A — (30.287, 3.830)"),
    (166, 603, "door B — (29.470, 5.575)"),
    (140, 722, "door C — (35.445, 4.238)"),
    (102, 722, "door D — (35.445, 2.338)"),
]

all_pass = True

for r, c, name in missed:
    current_cell = int(grid[r, c])

    r0 = max(0, r - snap_cells)
    r1 = min(grid.shape[0] - 1, r + snap_cells)
    c0 = max(0, c - snap_cells)
    c1 = min(grid.shape[1] - 1, c + snap_cells)

    patch      = grid[r0:r1+1, c0:c1+1]
    wall_count = int(np.sum((patch == 1) | (patch == 3)))
    will_stamp = wall_count > 0

    status = "PASS — will stamp as door" if will_stamp else "FAIL — still missed"
    if not will_stamp:
        all_pass = False

    print(f"{name}")
    print(f"  Grid cell     : ({r}, {c})")
    print(f"  Current value : {current_cell} (walkable)")
    print(f"  Patch size    : {patch.shape}")
    print(f"  Wall/outside  : {wall_count} cells in neighbourhood")
    print(f"  Result        : {status}")
    print()

print("─" * 50)
if all_pass:
    print("All 4 doors PASS — the 0.30 m snap radius is sufficient.")
    print("Safe to apply the fix in occupancy_grid.py.")
else:
    print("Some doors FAIL — increase DOOR_SNAP_M before applying the fix.")
    print("Try 0.50 m (snap_cells = 10) and re-run.")