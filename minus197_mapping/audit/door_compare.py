from __future__ import annotations
import json, math
from pathlib import Path

import numpy as np
from skimage.draw import polygon as sk_polygon

# constants from occupancy_grid.py
CELL_WALKABLE, CELL_WALL, CELL_DOOR, CELL_OUTSIDE, CELL_UNCERTAIN = 0,1,2,3,4
WALL_THICKNESS_M = 0.15
WALKABLE_CATEGORIES = {
    "corridor","entrance","exit","shop","food_court","restroom","office","storage","unknown"
}
CELL_NAME = {0:"walkable",1:"wall",2:"door",3:"outside",4:"uncertain"}

def pick_latest(outputs: Path, suffix: str) -> Path:
    files = sorted(outputs.glob(f"*{suffix}"))
    if not files:
        raise FileNotFoundError(f"No *{suffix} in {outputs}")
    return max(files, key=lambda p: p.stat().st_mtime)

def pick_by_stem(outputs: Path, stem: str, suffix: str) -> Path:
    p = outputs / f"{stem}{suffix}"
    if not p.exists():
        raise FileNotFoundError(f"Expected file not found: {p}")
    return p

def as_xy(v):
    if isinstance(v, dict): return float(v["x"]), float(v["y"])
    return float(v[0]), float(v[1])

def to_row(y, oy, res): return int((y - oy) / res)
def to_col(x, ox, res): return int((x - ox) / res)

def cell_center(r,c,ox,oy,res): return ox+(c+0.5)*res, oy+(r+0.5)*res

def point_seg_dist(px,py,ax,ay,bx,by):
    abx, aby = bx-ax, by-ay
    ab2 = abx*abx + aby*aby
    if ab2 == 0: return math.hypot(px-ax, py-ay)
    t = max(0.0, min(1.0, ((px-ax)*abx + (py-ay)*aby)/ab2))
    qx, qy = ax + t*abx, ay + t*aby
    return math.hypot(px-qx, py-qy)

def get_ifc_type(f):
    for k in ("ifc_type","ifc_entity","source_ifc_type","entity_type","type"):
        if k in f: return str(f[k])
    return ""

def get_name(f):
    for k in ("name","label","id","guid","ifc_name"):
        if k in f: return str(f[k])
    return "(unnamed)"

def is_source_door(f):
    t = get_ifc_type(f).lower()
    if t == "ifcdoor": return True
    return str(f.get("feature_type","")).lower() == "door"

def rebuild_pre_door_grid(sfm, occ):
    rows, cols = int(occ["height_cells"]), int(occ["width_cells"])
    res = float(occ["resolution_m"])
    ox, oy = float(occ["origin"]["x"]), float(occ["origin"]["y"])
    g = np.full((rows, cols), CELL_OUTSIDE, dtype=np.uint8)

    # walkable zones
    for z in sfm.get("zones", []):
        cat = str(z.get("category","")).lower().strip()
        if cat not in WALKABLE_CATEGORIES: 
            continue
        poly = z.get("boundary_polygon") or z.get("polygon") or z.get("boundary") or []
        if len(poly) < 3: 
            continue
        pts = [as_xy(p) for p in poly]
        rr = [to_row(y, oy, res) for _,y in pts]
        cc = [to_col(x, ox, res) for x,_ in pts]
        r, c = sk_polygon(rr, cc, shape=g.shape)
        g[r, c] = CELL_WALKABLE

    # walls
    t = WALL_THICKNESS_M
    for w in sfm.get("walls", []):
        if "start" not in w or "end" not in w: 
            continue
        (sx,sy), (ex,ey) = as_xy(w["start"]), as_xy(w["end"])
        pad = int(t/res) + 1
        r0, r1 = max(0, to_row(min(sy,ey), oy, res)-pad), min(rows-1, to_row(max(sy,ey), oy, res)+pad)
        c0, c1 = max(0, to_col(min(sx,ex), ox, res)-pad), min(cols-1, to_col(max(sx,ex), ox, res)+pad)
        for r in range(r0, r1+1):
            for c in range(c0, c1+1):
                wx, wy = cell_center(r,c,ox,oy,res)
                if point_seg_dist(wx,wy,sx,sy,ex,ey) <= t:
                    g[r,c] = CELL_WALL
    return g

repo = Path(r"d:\Aca\L4S2\research\mapping-module-try-out-v1\minus197_mapping")
out = repo / "data" / "outputs"
target_ifc = repo / "data" / "ifc_files" / "all_names_translated_english.ifc"
sfm_path = pick_latest(out, "_sfm.json")
occ_path = pick_latest(out, "_occupancy.json")

sfm = json.loads(sfm_path.read_text(encoding="utf-8"))
occ = json.loads(occ_path.read_text(encoding="utf-8"))

final_grid = np.array(occ["grid"], dtype=np.uint8)
pre_grid = rebuild_pre_door_grid(sfm, occ)

rows, cols = int(occ["height_cells"]), int(occ["width_cells"])
res = float(occ["resolution_m"])
ox, oy = float(occ["origin"]["x"]), float(occ["origin"]["y"])

doors = [f for f in sfm.get("features", []) if is_source_door(f)]

print(f"SFM: {sfm_path.name}")
print(f"OCC: {occ_path.name}\n")
print("| Name | IFC Type | World Position | Grid Cell | Existing Tile | Rendered? |")
print("|---|---|---|---|---|---|")

for f in doors:
    name = get_name(f).replace("|","/")
    ifc_type = (get_ifc_type(f) or f.get("feature_type","")).replace("|","/")
    pos = f.get("position")
    if not pos:
        print(f"| {name} | {ifc_type} | (missing) | - | - | No |")
        continue
    x, y = as_xy(pos)
    r, c = to_row(y, oy, res), to_col(x, ox, res)

    if not (0 <= r < rows and 0 <= c < cols):
        print(f"| {name} | {ifc_type} | ({x:.3f}, {y:.3f}) | ({r}, {c}) | OOB | No |")
        continue

    existing = int(pre_grid[r,c])
    rendered = int(final_grid[r,c]) == CELL_DOOR
    print(f"| {name} | {ifc_type} | ({x:.3f}, {y:.3f}) | ({r}, {c}) | {existing} ({CELL_NAME[existing]}) | {'Yes' if rendered else 'No'} |")