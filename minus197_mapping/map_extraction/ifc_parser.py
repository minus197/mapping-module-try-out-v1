"""
ifc_parser.py  —  Sprint 1  (production build)
------------------------------------------------
Parses a real-world IFC 2x3 / IFC4 file and returns a structured
IFCParseResult consumed by Sprint 2's SemanticFloorMapBuilder.

Geometry strategy (derived from real-file inspection)
------------------------------------------------------
  Spaces : footprint polygon from IfcExtrudedAreaSolid → SweptArea
             IfcArbitraryClosedProfileDef → IfcIndexedPolyCurve  (IFC4 typical)
             IfcArbitraryClosedProfileDef → IfcPolyline           (IFC2x3 fallback)
             IfcRectangleProfileDef                                (simple fallback)
           All local coordinates are transformed to world coords via
           ifcopenshell.util.placement.get_local_placement() matrix.

  Walls  : axis centreline from Axis representation
             IfcIndexedPolyCurve  (IFC4 typical)
             IfcPolyline          (IFC2x3 fallback)
           First and last coord-list points give start / end in world space.

  Units  : auto-detected from IfcSIUnit (MILLIMETRE → scale 1/1000, METRE → 1).
           All output coordinates are in metres.

  Landmarks: IfcTransportElement, IfcFurnishingElement, IfcDoor, IfcStair
           Position = ObjectPlacement world origin (mm → m).

Usage
-----
  from ifc_parser import IFCParser
  result = IFCParser("building.ifc").parse()
  print(result.summary())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import ifcopenshell
import ifcopenshell.util.placement

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

Point2D = Tuple[float, float]


@dataclass
class ParsedSpace:
    """One IfcSpace — becomes a Zone in the Semantic Floor Map Object."""
    ifc_id:    int
    guid:      str
    name:      str
    long_name: str
    polygon:   List[Point2D]   # world-space footprint polygon (metres, closed)

    # Derived — computed in __post_init__
    centroid:  Point2D                           = field(init=False)
    bbox:      Tuple[float, float, float, float] = field(init=False)  # x0,y0,x1,y1
    area:      float                             = field(init=False)
    width:     float                             = field(init=False)
    depth:     float                             = field(init=False)

    def __post_init__(self):
        if self.polygon:
            xs = [p[0] for p in self.polygon]
            ys = [p[1] for p in self.polygon]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            self.bbox     = (x0, y0, x1, y1)
            self.centroid = ((x0 + x1) / 2, (y0 + y1) / 2)
            self.width    = x1 - x0
            self.depth    = y1 - y0
            # Shoelace formula for polygon area
            n = len(self.polygon)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += self.polygon[i][0] * self.polygon[j][1]
                area -= self.polygon[j][0] * self.polygon[i][1]
            self.area = abs(area) / 2.0
        else:
            self.bbox     = (0.0, 0.0, 0.0, 0.0)
            self.centroid = (0.0, 0.0)
            self.width    = 0.0
            self.depth    = 0.0
            self.area     = 0.0


@dataclass
class ParsedWall:
    """One IfcWall — axis centreline segment in world metres."""
    ifc_id: int
    guid:   str
    name:   str
    start:  Point2D
    end:    Point2D

    @property
    def length(self) -> float:
        return math.hypot(self.end[0] - self.start[0],
                          self.end[1] - self.start[1])

    @property
    def midpoint(self) -> Point2D:
        return ((self.start[0] + self.end[0]) / 2,
                (self.start[1] + self.end[1]) / 2)


@dataclass
class ParsedFeature:
    """A landmark or passage point extracted from the IFC model."""
    ifc_id:       int
    guid:         str
    name:         str
    position:     Point2D
    feature_type: str   # elevator|escalator|stair|door|info_desk|bench|furnishing
    raw_class:    str


@dataclass
class IFCParseResult:
    """Everything extracted from one IFC file."""
    source_file: str
    schema:      str
    unit_scale:  float                 # multiplier: IFC raw coords → metres
    spaces:      List[ParsedSpace]     = field(default_factory=list)
    walls:       List[ParsedWall]      = field(default_factory=list)
    features:    List[ParsedFeature]   = field(default_factory=list)

    def summary(self) -> str:
        feat_types: Dict[str, int] = {}
        for f in self.features:
            feat_types[f.feature_type] = feat_types.get(f.feature_type, 0) + 1
        spaces_with_poly = sum(1 for s in self.spaces if s.polygon)
        return (
            f"IFCParseResult\n"
            f"  file        : {self.source_file}\n"
            f"  schema      : {self.schema}\n"
            f"  unit scale  : ×{self.unit_scale} (→ metres)\n"
            f"  spaces      : {len(self.spaces)}  ({spaces_with_poly} with geometry)\n"
            f"  walls       : {len(self.walls)}\n"
            f"  features    : {len(self.features)}  {dict(sorted(feat_types.items()))}\n"
        )


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

_TRANSPORT_KIND: Dict[str, str] = {
    "ELEVATOR":       "elevator",
    "ESCALATOR":      "escalator",
    "MOVING_WALKWAY": "escalator",
    "VEHICLE":        "furnishing",
    "CRANE":          "furnishing",
    "USERDEFINED":    "elevator",
    "NOTDEFINED":     "elevator",
}

_FURNISHING_KEYWORDS: Dict[str, str] = {
    "info":      "info_desk",
    "desk":      "info_desk",
    "reception": "info_desk",
    "bench":     "bench",
    "seat":      "bench",
    "chair":     "bench",
    "sofa":      "bench",
}


def _classify_furnishing(name: str) -> str:
    low = (name or "").lower()
    for kw, label in _FURNISHING_KEYWORDS.items():
        if kw in low:
            return label
    return "furnishing"


# ---------------------------------------------------------------------------
# Unit detection
# ---------------------------------------------------------------------------

_UNIT_SCALE_MAP: Dict[Tuple[Optional[str], str], float] = {
    ("MILLI", "METRE"): 1 / 1000,
    ("CENTI", "METRE"): 1 / 100,
    (None,    "METRE"): 1.0,
    ("",      "METRE"): 1.0,
    ("DECI",  "METRE"): 1 / 10,
    ("KILO",  "METRE"): 1000.0,
}


def _detect_unit_scale(model) -> float:
    """Return multiplier to convert IFC length units to metres."""
    for unit in model.by_type("IfcSIUnit"):
        try:
            if unit.UnitType == "LENGTHUNIT":
                prefix = unit.Prefix or None
                name   = unit.Name   or "METRE"
                key    = (prefix, name)
                if key in _UNIT_SCALE_MAP:
                    return _UNIT_SCALE_MAP[key]
        except Exception:
            pass
    return 1.0  # assume metres if unknown


# ---------------------------------------------------------------------------
# Coordinate transformation helpers
# ---------------------------------------------------------------------------

def _placement_matrix(entity) -> Optional[np.ndarray]:
    """Return the 4×4 world placement matrix for any IfcProduct."""
    try:
        lp = entity.ObjectPlacement
        if lp is None:
            return None
        return ifcopenshell.util.placement.get_local_placement(lp)
    except Exception:
        return None


def _world_xy(entity, scale: float) -> Optional[Point2D]:
    """Return world (x, y) origin of an IfcProduct, converted to metres."""
    mat = _placement_matrix(entity)
    if mat is None:
        return None
    return (float(mat[0, 3]) * scale, float(mat[1, 3]) * scale)


def _apply_mat_2d(local_x: float, local_y: float,
                  mat: np.ndarray, scale: float) -> Point2D:
    """Transform a 2-D local point through a 4×4 matrix to world metres."""
    v = np.array([local_x, local_y, 0.0, 1.0], dtype=float)
    w = mat @ v
    return (float(w[0]) * scale, float(w[1]) * scale)


# ---------------------------------------------------------------------------
# Space polygon extraction
# ---------------------------------------------------------------------------

def _poly_from_indexed_poly_curve(curve, mat: np.ndarray,
                                  scale: float) -> Optional[List[Point2D]]:
    """IfcIndexedPolyCurve (IFC4) → world polygon."""
    try:
        coord_list = curve.Points.CoordList
        return [_apply_mat_2d(c[0], c[1], mat, scale) for c in coord_list]
    except Exception:
        return None


def _poly_from_polyline(polyline, mat: np.ndarray,
                        scale: float) -> Optional[List[Point2D]]:
    """IfcPolyline (IFC2x3) → world polygon."""
    try:
        return [_apply_mat_2d(p.Coordinates[0], p.Coordinates[1], mat, scale)
                for p in polyline.Points]
    except Exception:
        return None


def _rect_polygon(x_dim: float, y_dim: float,
                  mat: np.ndarray, scale: float) -> List[Point2D]:
    """IfcRectangleProfileDef → axis-aligned world rectangle (4 vertices)."""
    w2, d2 = x_dim / 2, y_dim / 2
    corners = [(-w2, -d2), (w2, -d2), (w2, d2), (-w2, d2)]
    return [_apply_mat_2d(cx, cy, mat, scale) for cx, cy in corners]


def _extract_space_polygon(space, scale: float) -> Optional[List[Point2D]]:
    """
    Extract the world-space floor footprint polygon for an IfcSpace.

    Search order:
      1. Body rep  — ExtrudedAreaSolid → ArbitraryClosedProfileDef
                                        → IndexedPolyCurve / Polyline
      2. Body rep  — ExtrudedAreaSolid → RectangleProfileDef  (synthetic rect)
      3. FootPrint rep — bare IndexedPolyCurve / Polyline
    """
    if space.Representation is None:
        return None
    mat = _placement_matrix(space)
    if mat is None:
        return None

    rep_map: Dict[str, list] = {}
    for rep in space.Representation.Representations:
        rid = rep.RepresentationIdentifier or ""
        rep_map.setdefault(rid, []).append(rep)

    for pid in ("Body", "FootPrint", "Axis", "Reference"):
        for rep in rep_map.get(pid, []):
            for item in rep.Items:

                # Case A — ExtrudedAreaSolid
                if item.is_a("IfcExtrudedAreaSolid"):
                    sw = item.SweptArea
                    if sw.is_a("IfcArbitraryClosedProfileDef"):
                        curve = sw.OuterCurve
                        if curve.is_a("IfcIndexedPolyCurve"):
                            poly = _poly_from_indexed_poly_curve(curve, mat, scale)
                        else:
                            poly = _poly_from_polyline(curve, mat, scale)
                        if poly and len(poly) >= 3:
                            return poly
                    elif sw.is_a("IfcRectangleProfileDef"):
                        return _rect_polygon(float(sw.XDim), float(sw.YDim), mat, scale)

                # Case B — bare IndexedPolyCurve (FootPrint rep)
                elif item.is_a("IfcIndexedPolyCurve"):
                    poly = _poly_from_indexed_poly_curve(item, mat, scale)
                    if poly and len(poly) >= 3:
                        return poly

                # Case C — bare Polyline (FootPrint rep)
                elif item.is_a("IfcPolyline"):
                    poly = _poly_from_polyline(item, mat, scale)
                    if poly and len(poly) >= 3:
                        return poly

    return None


# ---------------------------------------------------------------------------
# Wall axis extraction
# ---------------------------------------------------------------------------

def _extract_wall_axis(wall, scale: float) -> Optional[Tuple[Point2D, Point2D]]:
    """
    Extract start/end world coordinates of a wall centreline axis.
    Looks for the 'Axis' representation containing IndexedPolyCurve or Polyline.
    Falls back to a zero-length stub at the ObjectPlacement origin.
    """
    if wall.Representation is None:
        return None
    mat = _placement_matrix(wall)
    if mat is None:
        return None

    for rep in wall.Representation.Representations:
        if (rep.RepresentationIdentifier or "") != "Axis":
            continue
        for item in rep.Items:
            pts_local: Optional[List[Tuple[float, float]]] = None

            if item.is_a("IfcIndexedPolyCurve"):
                cl = item.Points.CoordList
                if len(cl) >= 2:
                    pts_local = [(cl[0][0], cl[0][1]), (cl[-1][0], cl[-1][1])]

            elif item.is_a("IfcPolyline") and hasattr(item, "Points"):
                raw = item.Points
                if len(raw) >= 2:
                    pts_local = [
                        (raw[0].Coordinates[0],  raw[0].Coordinates[1]),
                        (raw[-1].Coordinates[0], raw[-1].Coordinates[1]),
                    ]

            if pts_local:
                p1 = _apply_mat_2d(pts_local[0][0], pts_local[0][1], mat, scale)
                p2 = _apply_mat_2d(pts_local[1][0], pts_local[1][1], mat, scale)
                return p1, p2

    # Fallback — placement origin stub
    origin = _world_xy(wall, scale)
    if origin:
        return origin, origin
    return None


# ---------------------------------------------------------------------------
# Main parser class
# ---------------------------------------------------------------------------

class IFCParser:
    """
    Parses an IFC 2x3 or IFC4 file.

    Parameters
    ----------
    ifc_path : str | Path
        Path to the .ifc file.
    """

    def __init__(self, ifc_path: str | Path):
        self.ifc_path = str(ifc_path)
        self._model   = None
        self._scale   = 1.0

    def parse(self) -> IFCParseResult:
        self._model = ifcopenshell.open(self.ifc_path)
        self._scale = _detect_unit_scale(self._model)

        result = IFCParseResult(
            source_file = self.ifc_path,
            schema      = self._model.schema,
            unit_scale  = self._scale,
        )
        result.spaces   = self._extract_spaces()
        result.walls    = self._extract_walls()
        result.features = self._extract_features()
        return result

    # ------------------------------------------------------------------
    def _extract_spaces(self) -> List[ParsedSpace]:
        spaces = []
        for s in self._model.by_type("IfcSpace"):
            poly = _extract_space_polygon(s, self._scale) or []
            spaces.append(ParsedSpace(
                ifc_id    = s.id(),
                guid      = s.GlobalId or "",
                name      = s.Name      or f"Space_{s.id()}",
                long_name = s.LongName  or s.Name or "",
                polygon   = poly,
            ))
        return spaces

    def _extract_walls(self) -> List[ParsedWall]:
        walls = []
        for w in self._model.by_type("IfcWall"):
            axis = _extract_wall_axis(w, self._scale)
            if axis is None:
                continue
            walls.append(ParsedWall(
                ifc_id = w.id(),
                guid   = w.GlobalId or "",
                name   = w.Name     or f"Wall_{w.id()}",
                start  = axis[0],
                end    = axis[1],
            ))
        return walls

    def _extract_features(self) -> List[ParsedFeature]:
        feats = []

        for t in self._model.by_type("IfcTransportElement"):
            pos = _world_xy(t, self._scale)
            if pos is None:
                continue
            kind_raw = str(getattr(t, "PredefinedType", None) or "NOTDEFINED")
            feats.append(ParsedFeature(
                ifc_id       = t.id(),
                guid         = t.GlobalId or "",
                name         = t.Name     or f"Transport_{t.id()}",
                position     = pos,
                feature_type = _TRANSPORT_KIND.get(kind_raw, "elevator"),
                raw_class    = "IfcTransportElement",
            ))

        for f in self._model.by_type("IfcFurnishingElement"):
            pos = _world_xy(f, self._scale)
            if pos is None:
                continue
            feats.append(ParsedFeature(
                ifc_id       = f.id(),
                guid         = f.GlobalId or "",
                name         = f.Name     or f"Furnishing_{f.id()}",
                position     = pos,
                feature_type = _classify_furnishing(f.Name or ""),
                raw_class    = "IfcFurnishingElement",
            ))

        for d in self._model.by_type("IfcDoor"):
            pos = _world_xy(d, self._scale)
            if pos is None:
                continue
            feats.append(ParsedFeature(
                ifc_id       = d.id(),
                guid         = d.GlobalId or "",
                name         = d.Name     or f"Door_{d.id()}",
                position     = pos,
                feature_type = "door",
                raw_class    = "IfcDoor",
            ))

        for st in self._model.by_type("IfcStair"):
            pos = _world_xy(st, self._scale)
            if pos is None:
                continue
            feats.append(ParsedFeature(
                ifc_id       = st.id(),
                guid         = st.GlobalId or "",
                name         = st.Name     or f"Stair_{st.id()}",
                position     = pos,
                feature_type = "stair",
                raw_class    = "IfcStair",
            ))

        return feats
