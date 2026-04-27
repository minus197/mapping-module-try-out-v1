"""
semantic_floor_map.py  —  Sprint 2  (production build)
-------------------------------------------------------
Converts an IFCParseResult into the Semantic Floor Map Object (SFMO),
a JSON-serialisable structure consumed by all downstream modules.

    SemanticFloorMap
    ├── meta        : source, floor label, bounding box, counts
    ├── zones[]     : named spaces with real polygon, centroid, category
    ├── walls[]     : wall segments tagged shore-linable / non-shore-linable
    └── features[]  : landmark nodes with zone containment

Design notes
------------
Zone category classification
    Keyword matching on Name + LongName.  Text is normalised to ASCII
    where possible so both Cyrillic/Latin names work via the optional
    transliteration map below.  A real system would also use IFC
    property sets (Pset_SpaceCommon.Category) — easy to add here.

Shore-line tagging
    A wall is shore_linable when its axis start, midpoint, or end lies
    within SHORE_BUFFER metres of any corridor zone polygon.

Point-in-polygon
    Ray-casting algorithm supports arbitrary (non-convex) polygons
    extracted from real IFC files.

JSON safety
    All Point2D tuples are serialised as plain Python lists of plain
    Python floats to guarantee json.dumps compatibility.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from map_extraction.ifc_parser import IFCParseResult, ParsedFeature, ParsedSpace, ParsedWall, Point2D

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SHORE_BUFFER: float = 1.0   # metres — wall endpoint within this of a corridor → shore-linable

# Keyword → zone category.  Lower-case, space-stripped matching.
# Extended to recognise Russian transliterations common in IFC files.
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "corridor":   ["corridor", "hallway", "passage", "walkway", "aisle",
                   "koridory", "koridor", "коридор"],
    "shop":       ["shop", "store", "boutique", "retail", "торговый", "торг",
                   "nike", "zara", "h&m", "brand", "outlet", "магазин"],
    "food_court": ["food", "court", "cafe", "restaurant", "dining", "canteen",
                   "столовая", "кафе", "ресторан"],
    "restroom":   ["restroom", "toilet", "wc", "lavatory", "bathroom",
                   "washroom", "санузел", "туалет", "сан."],
    "entrance":   ["entrance", "entry", "lobby", "foyer", "reception",
                   "вход", "холл", "фойе", "лобби"],
    "exit":       ["exit", "emergency", "fire", "evacuation", "выход"],
    "storage":    ["storage", "warehouse", "склад", "холодильная", "кладовая",
                   "разделочная", "раздевалка"],
    "office":     ["office", "кабинет", "офис"],
}

_FEATURE_PRIORITY: Dict[str, int] = {
    "elevator":   3,
    "escalator":  3,
    "stair":      3,
    "door":       2,
    "info_desk":  2,
    "bench":      1,
    "furnishing": 1,
}


# ---------------------------------------------------------------------------
# Semantic data structures
# ---------------------------------------------------------------------------

@dataclass
class Zone:
    zone_id:          str
    ifc_guid:         str
    name:             str
    long_name:        str
    category:         str
    centroid:         Point2D
    boundary_polygon: List[Point2D]   # closed polygon, arbitrary shape
    width:            float           # bounding box width (m)
    depth:            float           # bounding box depth (m)
    area:             float           # shoelace area (m²)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone_id":          self.zone_id,
            "ifc_guid":         self.ifc_guid,
            "name":             self.name,
            "long_name":        self.long_name,
            "category":         self.category,
            "centroid":         [float(v) for v in self.centroid],
            "boundary_polygon": [[float(x), float(y)] for x, y in self.boundary_polygon],
            "width_m":          round(float(self.width), 4),
            "depth_m":          round(float(self.depth), 4),
            "area_m2":          round(float(self.area),  4),
        }


@dataclass
class WallSegment:
    wall_id:       str
    ifc_guid:      str
    name:          str
    start:         Point2D
    end:           Point2D
    length:        float
    shore_linable: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wall_id":       self.wall_id,
            "ifc_guid":      self.ifc_guid,
            "name":          self.name,
            "start":         [float(v) for v in self.start],
            "end":           [float(v) for v in self.end],
            "length_m":      round(float(self.length), 4),
            "shore_linable": bool(self.shore_linable),
        }


@dataclass
class Feature:
    feature_id:   str
    ifc_guid:     str
    name:         str
    feature_type: str
    position:     Point2D
    zone_id:      Optional[str]
    priority:     int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_id":   self.feature_id,
            "ifc_guid":     self.ifc_guid,
            "name":         self.name,
            "feature_type": self.feature_type,
            "position":     [float(v) for v in self.position],
            "zone_id":      self.zone_id,
            "priority":     int(self.priority),
        }


@dataclass
class SemanticFloorMap:
    """The complete Semantic Floor Map Object for one building floor."""
    source_file:  str
    floor_label:  str
    bounding_box: Dict[str, float]
    zones:        List[Zone]        = field(default_factory=list)
    walls:        List[WallSegment] = field(default_factory=list)
    features:     List[Feature]     = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": {
                "source_file":   self.source_file,
                "floor_label":   self.floor_label,
                "zone_count":    len(self.zones),
                "wall_count":    len(self.walls),
                "feature_count": len(self.features),
            },
            "bounding_box": {k: float(v) for k, v in self.bounding_box.items()},
            "zones":    [z.to_dict() for z in self.zones],
            "walls":    [w.to_dict() for w in self.walls],
            "features": [f.to_dict() for f in self.features],
        }

    def save(self, path: str | Path = "semantic_floor_map.json") -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                     encoding="utf-8")
        print(f"[SemanticFloorMap] Saved → {p.resolve()}")
        return p

    def summary(self) -> str:
        zone_cats: Dict[str, int] = {}
        for z in self.zones:
            zone_cats[z.category] = zone_cats.get(z.category, 0) + 1
        feat_types: Dict[str, int] = {}
        for f in self.features:
            feat_types[f.feature_type] = feat_types.get(f.feature_type, 0) + 1
        shore_n = sum(1 for w in self.walls if w.shore_linable)
        bb = self.bounding_box
        return (
            f"SemanticFloorMap — {self.floor_label}\n"
            f"  Source    : {self.source_file}\n"
            f"  BBox      : x=[{bb['min_x']:.2f}, {bb['max_x']:.2f}]"
            f"  y=[{bb['min_y']:.2f}, {bb['max_y']:.2f}]\n"
            f"  Zones     : {len(self.zones)}  {dict(sorted(zone_cats.items()))}\n"
            f"  Walls     : {len(self.walls)}  ({shore_n} shore-linable)\n"
            f"  Features  : {len(self.features)}  {dict(sorted(feat_types.items()))}\n"
        )


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def _classify_zone(name: str, long_name: str) -> str:
    combined = f"{name} {long_name}".lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return category
    return "unknown"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _point_in_polygon(px: float, py: float, polygon: List[Point2D]) -> bool:
    """
    Ray-casting point-in-polygon test. Handles non-convex polygons.
    Returns True if (px, py) is inside the polygon.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _point_near_polygon(px: float, py: float,
                        polygon: List[Point2D],
                        buffer: float) -> bool:
    """
    True if (px, py) is inside the polygon OR within `buffer` metres
    of any edge of the polygon.
    """
    if _point_in_polygon(px, py, polygon):
        return True
    # Check distance to each edge
    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n
        ax, ay = polygon[i]
        bx, by = polygon[j]
        # Vector AB
        abx, aby = bx - ax, by - ay
        ab_len_sq = abx * abx + aby * aby
        if ab_len_sq == 0:
            dist = math.hypot(px - ax, py - ay)
        else:
            t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_len_sq))
            nearest_x = ax + t * abx
            nearest_y = ay + t * aby
            dist = math.hypot(px - nearest_x, py - nearest_y)
        if dist <= buffer:
            return True
    return False


def _wall_near_corridor(wall: ParsedWall,
                        corridor_polys: List[List[Point2D]],
                        buffer: float = SHORE_BUFFER) -> bool:
    """True if any of the wall's key points is within buffer of a corridor polygon."""
    test_pts = [wall.start, wall.midpoint, wall.end]
    for poly in corridor_polys:
        for px, py in test_pts:
            if _point_near_polygon(px, py, poly, buffer):
                return True
    return False


def _find_containing_zone(pos: Point2D,
                          raw_spaces: List[ParsedSpace]) -> Optional[str]:
    """Return the GUID of the first zone whose polygon contains the position."""
    px, py = pos
    # First pass: strict containment
    for s in raw_spaces:
        if s.polygon and _point_in_polygon(px, py, s.polygon):
            return s.guid
    # Second pass: 0.5 m buffer (for door/feature positions on zone boundary)
    for s in raw_spaces:
        if s.polygon and _point_near_polygon(px, py, s.polygon, 0.5):
            return s.guid
    return None


def _compute_bounding_box(spaces: List[ParsedSpace],
                          walls:  List[ParsedWall]) -> Dict[str, float]:
    xs: List[float] = []
    ys: List[float] = []
    for s in spaces:
        for x, y in s.polygon:
            xs.append(x); ys.append(y)
    for w in walls:
        xs += [w.start[0], w.end[0]]
        ys += [w.start[1], w.end[1]]
    if not xs:
        return {"min_x": 0.0, "min_y": 0.0, "max_x": 0.0, "max_y": 0.0}
    return {
        "min_x": round(float(min(xs)), 4),
        "min_y": round(float(min(ys)), 4),
        "max_x": round(float(max(xs)), 4),
        "max_y": round(float(max(ys)), 4),
    }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class SemanticFloorMapBuilder:
    """
    Transforms an IFCParseResult into a SemanticFloorMap.

    Parameters
    ----------
    parse_result : IFCParseResult
    floor_label  : str  e.g. 'L1', 'Ground', 'B1'
    """

    def __init__(self, parse_result: IFCParseResult, floor_label: str = "L1"):
        self.pr    = parse_result
        self.label = floor_label

    def build(self) -> SemanticFloorMap:
        zones    = self._build_zones()
        walls    = self._build_walls()
        features = self._build_features()
        bbox     = _compute_bounding_box(self.pr.spaces, self.pr.walls)

        return SemanticFloorMap(
            source_file  = self.pr.source_file,
            floor_label  = self.label,
            bounding_box = bbox,
            zones        = zones,
            walls        = walls,
            features     = features,
        )

    # ------------------------------------------------------------------

    def _build_zones(self) -> List[Zone]:
        zones = []
        for ps in self.pr.spaces:
            category = _classify_zone(ps.name, ps.long_name)
            # Use real polygon if available; fallback to AABB as polygon
            if ps.polygon:
                poly = ps.polygon
            else:
                x0, y0, x1, y1 = ps.bbox
                poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

            zones.append(Zone(
                zone_id          = ps.guid or f"ZONE_{ps.ifc_id}",
                ifc_guid         = ps.guid,
                name             = ps.name,
                long_name        = ps.long_name,
                category         = category,
                centroid         = ps.centroid,
                boundary_polygon = poly,
                width            = ps.width,
                depth            = ps.depth,
                area             = ps.area,
            ))
        return zones

    def _build_walls(self) -> List[WallSegment]:
        # Collect corridor polygons for shore-line tagging
        corridor_polys = [
            ps.polygon for ps in self.pr.spaces
            if ps.polygon and _classify_zone(ps.name, ps.long_name) == "corridor"
        ]

        walls = []
        for idx, pw in enumerate(self.pr.walls):
            shore = _wall_near_corridor(pw, corridor_polys) if corridor_polys else False
            walls.append(WallSegment(
                wall_id       = pw.guid or f"WALL_{idx:04d}",
                ifc_guid      = pw.guid,
                name          = pw.name,
                start         = pw.start,
                end           = pw.end,
                length        = pw.length,
                shore_linable = shore,
            ))
        return walls

    def _build_features(self) -> List[Feature]:
        raw_spaces = self.pr.spaces
        features = []
        for idx, pf in enumerate(self.pr.features):
            zone_id = _find_containing_zone(pf.position, raw_spaces)
            features.append(Feature(
                feature_id   = pf.guid or f"FEAT_{idx:04d}",
                ifc_guid     = pf.guid,
                name         = pf.name,
                feature_type = pf.feature_type,
                position     = pf.position,
                zone_id      = zone_id,
                priority     = _FEATURE_PRIORITY.get(pf.feature_type, 1),
            ))
        return features