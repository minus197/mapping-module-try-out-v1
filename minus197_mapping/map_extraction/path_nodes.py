"""
map_extraction/path_nodes.py
----------------------------
Builds the *path nodes* layer and exports it as <stem>_path_nodes.json.

Path nodes are cane-trailing waypoints placed ALONG the physical walls that
line the walkable circulation area (the corridor). They give a visually
impaired user a dense set of positions to shore-line against while walking a
corridor, distinct from the sparse navigation-graph junction nodes.

Definition (as specified by the requirement)
---------------------------------------------
1. A path node sits ~PATH_GAP_M (0.3 m) away from an actual wall of the map,
   on the corridor-facing side of that wall.
2. Consecutive path nodes along a wall are spaced ~NODE_SPACING m apart
   (SPACING_MIN .. SPACING_MAX, i.e. 5.5 m .. 6.0 m).
3. Path nodes are placed only along walls that face the walkable circulation
   network (corridor / entrance / exit zones). This covers:
       - the physical walls of the corridor,
       - the corridor-facing (outer) wall of each shop,
       - interior walls inside the building that border the corridor.
   They are NOT placed along:
       - the inside faces of shop walls (those face a shop interior, not the
         corridor — the offset point lands inside the shop, so it is rejected),
       - the building's outer perimeter walls (the offset point lands outside
         the building / not inside a corridor, so it is rejected).
4. No edges are produced — this layer is purely a set of node positions.

Per-node search circle (added on top of the basic layer)
--------------------------------------------------------
For every path node two extra pieces of information are computed:

  * ``nearest_wall_dist_m`` (x) — the TRUE distance from the node's position
    to the nearest wall, recomputed by geometry rather than assumed equal to
    the node's stored ``gap_m``::

        x(node) = min over all walls in sfm.walls of
                  distance(node.position, wall_segment)

    In most cases x ≈ PATH_GAP_M (the node's own gap). Only near a dead end /
    corner — where the wall capping the corridor in front of the node is
    closer than the wall it shore-lines along — does x come out smaller.

  * ``search_circle`` — a circle of radius 2·x centred on the node. Every
    non-circulation zone (rooms / shops) and every feature (doors + landmarks
    such as staircases / elevators / washrooms) whose boundary falls inside
    this circle is reported with:
        - ``node_id``          — the graph node id (ZONE-<id> / FEAT-<id>),
        - ``perp_dist_m``      — perpendicular distance from the path node to
                                 the target's boundary (for a point-only
                                 target this equals the centroid distance),
        - ``centroid_dist_m``  — distance from the path node to the target's
                                 centroid.
    A target without a boundary polygon (doors and point-only landmarks) is
    matched on, and measured to, its centroid.

Method
------
For every wall segment:
  * Sample points every ~PATH_GAP_M along the wall axis, spaced later to the
    5.5-6 m target (see _place_along_wall).
  * At each sample, offset PATH_GAP_M along each wall normal (both sides).
  * Keep the offset that lands inside the corridor union; that is the
    corridor-facing side. If neither side is inside the corridor, the wall
    does not border the corridor at that point and no node is placed there.
A final de-duplication pass drops nodes closer than DEDUP_RADIUS to an
already-kept node, so wall corners are not double-noded.

This module depends only on the SemanticFloorMap (zones + walls) and Shapely —
the same inputs the occupancy grid and graph builder already use.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from map_extraction.semantic_floor_map import SemanticFloorMap
from shared.types import Point2D

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

PATH_GAP_M     = 2.0   # m — gap between a wall and its path node
SPACING_MIN    = 5.5    # m — minimum spacing between consecutive path nodes
SPACING_MAX    = 6.0    # m — maximum (target) spacing between path nodes
DEDUP_RADIUS   = 2.0    # m — drop a node this close to an already-kept node
MIN_WALL_LEN   = 0.50   # m — ignore wall stubs shorter than this
CORRIDOR_BUFFER = 0.05  # m — small outward buffer on the corridor union so a
                        # point sitting exactly on the corridor edge counts as
                        # inside (absorbs floating-point boundary cases)
# A candidate node landing within this distance of ANY wall centreline is
# rejected, so a node is never placed on (or hard against) a wall. This is what
# skips a node that, while being laid along one wall, happens to land on a
# perpendicular wall bounding the corridor (e.g. at a corridor corner).
WALL_CLEARANCE_M = 0.50  # m — minimum clearance a node must keep from any wall

# Zones whose interior is the walkable *circulation network*. Path nodes are
# kept only where the wall's 0.3 m offset lands inside one of these — matching
# the graph builder's CIRCULATION_CATEGORIES so "corridor side" means the same
# thing across the pipeline.
CIRCULATION_CATEGORIES = {"corridor", "entrance", "exit"}


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class SearchHit:
    """
    One zone / feature whose boundary falls inside a path node's search circle.

    A target without a boundary polygon (doors, point-only landmarks) is
    matched on its centroid, and ``perp_dist_m`` then equals ``centroid_dist_m``.
    """
    node_id:         str    # graph node id of the target (ZONE-<id> / FEAT-<id>)
    kind:            str    # "zone" | "feature"
    perp_dist_m:     float  # perpendicular distance node → target boundary
    centroid_dist_m: float  # distance node → target centroid

    def to_dict(self) -> Dict[str, object]:
        return {
            "node_id":         self.node_id,
            "kind":            self.kind,
            "perp_dist_m":     round(float(self.perp_dist_m), 4),
            "centroid_dist_m": round(float(self.centroid_dist_m), 4),
        }


@dataclass
class PathNode:
    """One cane-trailing waypoint offset PATH_GAP_M from a wall."""
    node_id:   str
    position:  Point2D
    wall_id:   str        # the wall this node shore-lines against
    gap_m:     float      # actual gap to the wall (≈ PATH_GAP_M)
    # Populated by PathNodeBuilder._annotate() after de-duplication.
    nearest_wall_dist_m: float = 0.0          # x — true nearest-wall distance
    search_hits: List["SearchHit"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        # x is the true nearest-wall distance; the search circle has radius 2·x.
        x = self.nearest_wall_dist_m
        return {
            "node_id":  self.node_id,
            "position": [round(float(self.position[0]), 4),
                         round(float(self.position[1]), 4)],
            "wall_id":  self.wall_id,
            "gap_m":    round(float(self.gap_m), 4),
            "nearest_wall_dist_m": round(float(x), 4),
            "search_circle": {
                "radius_m": round(float(2.0 * x), 4),
                "hits":     [h.to_dict() for h in self.search_hits],
            },
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class PathNodeBuilder:
    """
    Produces the path-node layer from a SemanticFloorMap.

    Parameters
    ----------
    sfm : SemanticFloorMap  — the already-built floor map (zones + walls)

    Usage
    -----
        builder = PathNodeBuilder(sfm)
        builder.build().save("data/outputs/<stem>_path_nodes.json")
    """

    def __init__(self, sfm: SemanticFloorMap):
        self.sfm = sfm
        self._nodes: List[PathNode] = []
        self._corridor: Optional[object] = None   # Shapely corridor union
        self._wall_obst: Optional[object] = None   # all wall lines buffered
        self._wall_lines: List[object] = []        # per-wall LineStrings (for x)
        self._targets: List["_Target"] = []        # search-circle candidates

    # ── Public build / save ───────────────────────────────────────────────────

    def build(self) -> "PathNodeBuilder":
        print("[PathNodes] Building corridor geometry ...")
        self._corridor = self._build_corridor_union()
        self._wall_obst = self._build_wall_obstacle()
        if self._corridor is None:
            print("[PathNodes]   WARNING: no circulation "
                  f"({'/'.join(sorted(CIRCULATION_CATEGORIES))}) zone found — "
                  "no path nodes can be placed.")
            return self

        print("[PathNodes] Placing path nodes along corridor-facing walls ...")
        raw: List[PathNode] = []
        for idx, wall in enumerate(self.sfm.walls):
            raw.extend(self._place_along_wall(wall, idx))

        self._nodes = _dedup(raw, DEDUP_RADIUS)
        print(f"[PathNodes] Done: {len(self._nodes)} path nodes "
              f"(from {len(raw)} pre-dedup) along "
              f"{len({n.wall_id for n in self._nodes})} walls")

        print("[PathNodes] Computing nearest-wall distance (x) and search "
              "circles ...")
        self._targets = self._build_targets()
        self._annotate()
        n_hits = sum(len(n.search_hits) for n in self._nodes)
        print(f"[PathNodes]   {n_hits} search-circle hits across "
              f"{len(self._nodes)} nodes over {len(self._targets)} targets")
        return self

    def save(self, path: str | Path = "path_nodes.json") -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._to_dict(), indent=2, ensure_ascii=False),
                     encoding="utf-8")
        print(f"[PathNodes] Saved → {p.resolve()}")
        return p

    @property
    def nodes(self) -> List[PathNode]:
        return self._nodes

    # ── Corridor geometry ─────────────────────────────────────────────────────

    def _build_corridor_union(self) -> Optional[object]:
        """
        Union of all circulation-zone polygons, buffered outward by a hair so
        boundary points count as inside. This is the region a path node's
        0.3 m offset must fall into for the node to be kept — i.e. the wall's
        corridor-facing side.
        """
        polys: List[object] = []
        for zone in self.sfm.zones:
            if zone.category not in CIRCULATION_CATEGORIES:
                continue
            if len(zone.boundary_polygon) < 3:
                continue
            try:
                sp = Polygon(zone.boundary_polygon).buffer(0)
                if sp.is_valid and sp.area > 0:
                    polys.append(sp)
            except Exception:
                pass
        if not polys:
            return None
        return unary_union(polys).buffer(CORRIDOR_BUFFER)

    def _build_wall_obstacle(self) -> Optional[object]:
        """
        Union of every wall centreline, buffered by WALL_CLEARANCE_M. A
        candidate node whose position lands inside this region sits on (or too
        close to) a wall and is rejected. This is what stops a node laid along
        one wall from landing on a perpendicular wall bounding the corridor.
        """
        lines: List[object] = []
        for wall in self.sfm.walls:
            try:
                seg = LineString([wall.start, wall.end])
                if seg.length > 1e-6:
                    lines.append(seg)
            except Exception:
                pass
        # Keep the individual segments so nearest-wall distance (x) can be
        # measured against each wall; the buffered union below is only used for
        # the on-wall rejection test.
        self._wall_lines = lines
        if not lines:
            return None
        return unary_union(lines).buffer(WALL_CLEARANCE_M)

    # ── Node placement along one wall ─────────────────────────────────────────

    def _place_along_wall(self, wall, wall_idx: int) -> List[PathNode]:
        """
        Place path nodes along one wall, PATH_GAP_M off its corridor-facing
        side(s), spaced SPACING_MIN..SPACING_MAX apart.

        Two behaviours added on top of the basic offset:
          * If BOTH sides of the wall face the corridor (a free-standing wall
            with corridor on either side), nodes are placed on BOTH sides so a
            visually impaired user gets a consistent line of nodes whichever
            side they are trailing.
          * A candidate node that lands on (or within WALL_CLEARANCE_M of) any
            wall is skipped — the rest of the nodes are still placed. This is
            what drops a node laid along a horizontal wall when it would fall
            on a perpendicular wall bounding the corridor (a corner).

        Returns [] when the wall never borders the corridor (both offset sides
        land outside the corridor union along its whole length) — this rejects
        building-perimeter walls and shop-interior faces automatically.
        """
        s = np.array(wall.start, dtype=float)
        e = np.array(wall.end,   dtype=float)
        seg_len = float(np.hypot(*(e - s)))
        if seg_len < MIN_WALL_LEN:
            return []

        axis   = (e - s) / seg_len
        normal = np.array([-axis[1], axis[0]])   # unit normal to the wall

        # Distances along the wall at which to try placing a node. Start half a
        # spacing in from the corner and step by the target spacing, so nodes
        # sit roughly centred on the wall rather than right on its corners.
        spacing   = SPACING_MAX
        positions = _spacing_positions(seg_len, spacing)

        out: List[PathNode] = []
        for t in positions:
            base = s + axis * t
            for side in self._corridor_sides(base, normal):
                pos = base + normal * (PATH_GAP_M * side)
                # Skip a node that lands on / hard against any wall (e.g. a node
                # from a horizontal wall landing on a perpendicular wall at a
                # corridor corner). Keep placing the remaining nodes.
                if self._on_wall(pos):
                    continue
                out.append(PathNode(
                    node_id  = f"PATH-{wall_idx:04d}-{len(out):03d}",
                    position = (float(pos[0]), float(pos[1])),
                    wall_id  = wall.wall_id,
                    gap_m    = PATH_GAP_M,
                ))
        return out

    def _corridor_sides(self, base: np.ndarray, normal: np.ndarray) -> List[int]:
        """
        Which side(s) of the wall (at point `base`) face the corridor.

        Returns a list containing +1 (the +normal side), -1 (the -normal side),
        or both when the wall has corridor on either side, or [] when neither
        side is inside the corridor (a building-exterior wall or a shop-interior
        face). Returning both sides is what lets a free-standing wall carry a
        line of nodes on each of its faces.
        """
        sides: List[int] = []
        if self._corridor.contains(Point(*(base + normal * PATH_GAP_M))):
            sides.append(+1)
        if self._corridor.contains(Point(*(base - normal * PATH_GAP_M))):
            sides.append(-1)
        return sides

    def _on_wall(self, pos: np.ndarray) -> bool:
        """
        True if `pos` lands on (or within WALL_CLEARANCE_M of) any wall — i.e.
        the node would sit on top of a wall and must be skipped.
        """
        if self._wall_obst is None:
            return False
        return self._wall_obst.contains(Point(float(pos[0]), float(pos[1])))

    # ── Nearest-wall distance (x) + per-node search circle ────────────────────

    def _build_targets(self) -> List["_Target"]:
        """
        The zones and features a search circle can match.

        Included:
          * every zone that is NOT a circulation zone (rooms / shops), carried
            as its boundary polygon — path nodes already live in the corridor,
            so matching the corridor/entrance/exit zones back to themselves
            would only add noise;
          * every feature (doors + landmarks such as staircases / elevators /
            washrooms), carried as a point (features have no boundary polygon).

        The ``node_id`` mirrors the graph builder's ids: ``ZONE-<zone_id>`` for a
        zone centroid and ``FEAT-<feature_id>`` for a feature, so a consumer can
        join these hits straight onto the navigation graph.
        """
        targets: List["_Target"] = []

        for zone in self.sfm.zones:
            if zone.category in CIRCULATION_CATEGORIES:
                continue
            poly = None
            if len(zone.boundary_polygon) >= 3:
                try:
                    sp = Polygon(zone.boundary_polygon).buffer(0)
                    if sp.is_valid and sp.area > 0:
                        poly = sp
                except Exception:
                    poly = None
            centroid = (float(zone.centroid[0]), float(zone.centroid[1]))
            targets.append(_Target(
                node_id  = f"ZONE-{zone.zone_id}",
                kind     = "zone",
                polygon  = poly,
                centroid = centroid,
            ))

        for feat in self.sfm.features:
            centroid = (float(feat.position[0]), float(feat.position[1]))
            poly = None
            boundary = getattr(feat, "boundary_polygon", None)
            if boundary and len(boundary) >= 3:
                try:
                    sp = Polygon(boundary).buffer(0)
                    if sp.is_valid and sp.area > 0:
                        poly = sp
                except Exception:
                    poly = None
            targets.append(_Target(
                node_id  = f"FEAT-{feat.feature_id}",
                kind     = "feature",
                polygon  = poly,           # None for point-only features
                centroid = centroid,
            ))

        return targets

    def _annotate(self) -> None:
        """
        Fill in each node's nearest-wall distance x and its search-circle hits.

        x = min distance from the node to any wall segment (recomputed by
        geometry). The search circle has radius 2·x; a target is a hit when its
        boundary is within 2·x of the node (or, for a point-only target, when
        its centroid is). ``perp_dist_m`` is the perpendicular distance to the
        boundary — for a point-only target this collapses to the centroid
        distance.
        """
        for node in self._nodes:
            p = Point(node.position[0], node.position[1])

            x = self._nearest_wall_dist(p)
            node.nearest_wall_dist_m = x
            radius = 2.0 * x

            hits: List[SearchHit] = []
            for tgt in self._targets:
                cx, cy = tgt.centroid
                centroid_dist = math.hypot(node.position[0] - cx,
                                           node.position[1] - cy)
                if tgt.polygon is not None:
                    perp_dist = tgt.polygon.distance(p)      # 0 if inside
                    inside = perp_dist <= radius or centroid_dist <= radius
                else:
                    perp_dist = centroid_dist                # point-only target
                    inside = centroid_dist <= radius
                if not inside:
                    continue
                hits.append(SearchHit(
                    node_id         = tgt.node_id,
                    kind            = tgt.kind,
                    perp_dist_m     = perp_dist,
                    centroid_dist_m = centroid_dist,
                ))

            hits.sort(key=lambda h: h.perp_dist_m)
            node.search_hits = hits

    def _nearest_wall_dist(self, p: object) -> float:
        """
        True distance from point ``p`` to the nearest wall segment. Falls back
        to the node's gap (PATH_GAP_M) if there are no walls to measure against.
        """
        if not self._wall_lines:
            return PATH_GAP_M
        return min(line.distance(p) for line in self._wall_lines)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def _to_dict(self) -> Dict[str, object]:
        bb = self.sfm.bounding_box
        return {
            "floor_label": self.sfm.floor_label,
            "source_file": self.sfm.source_file,

            "meta": {
                "path_node_count": len(self._nodes),
                "gap_to_wall_m":   PATH_GAP_M,
                "spacing_min_m":   SPACING_MIN,
                "spacing_max_m":   SPACING_MAX,
                "wall_clearance_m": WALL_CLEARANCE_M,
                "search_circle_radius": "2 * nearest_wall_dist_m (x) per node",
                "description": (
                    f"Cane-trailing waypoints ~{PATH_GAP_M} m off the "
                    "corridor-facing side of walls that line the walkable "
                    "circulation area, spaced ~5.5-6 m apart. No edges. "
                    "Building-perimeter walls and shop-interior wall faces are "
                    "excluded. Walls with corridor on both sides carry nodes on "
                    "both sides. Nodes that would land on a wall (e.g. at a "
                    "corridor corner) are skipped. Each node also carries "
                    "nearest_wall_dist_m (x, the true geometric distance to the "
                    "closest wall) and a search_circle of radius 2*x listing "
                    "non-circulation zones (rooms/shops) and features "
                    "(doors/landmarks) whose boundary falls inside it, each with "
                    "perpendicular distance to the boundary and distance to the "
                    "centroid. Point-only targets are matched on their centroid."
                ),
            },

            # Same IFC project coordinate frame as every other mapping output.
            "coordinate_frame": {
                "units": "metres",
                "source": "IFC project coordinate system",
                "x_axis": "IFC project X axis",
                "y_axis": "IFC project Y axis",
                "origin_description": (
                    "IFC project origin — positions are exact Shapely "
                    "coordinates in this frame"
                ),
            },
            "bounding_box": {k: float(v) for k, v in bb.items()},

            "path_nodes": [n.to_dict() for n in self._nodes],
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

@dataclass
class _Target:
    """A zone / feature the search circle can match (internal, not serialised)."""
    node_id:  str               # graph node id (ZONE-<id> / FEAT-<id>)
    kind:     str               # "zone" | "feature"
    polygon:  Optional[object]  # Shapely polygon, or None for point-only targets
    centroid: Point2D


def _spacing_positions(seg_len: float, spacing: float) -> List[float]:
    """
    Distances along a wall of length seg_len at which to place nodes.

    The requirement is that the gap between CONSECUTIVE nodes is ~6 m, anywhere
    in [SPACING_MIN, SPACING_MAX]; the corner-to-first-node distance is not
    constrained. So we lay nodes with a fixed in-range step and let the wall's
    leftover length become symmetric end margins.

    Node count is chosen so the even step seg_len / n_gaps stays ≥ SPACING_MIN
    (never clamped UP, which would push the end nodes off the wall). If that
    step exceeds SPACING_MAX (a wall too short to split evenly in-band, e.g.
    ~6-11 m), it is clamped DOWN to SPACING_MAX and the node run is centred,
    leaving equal end margins — every inter-node gap still lands in the band.

    Short walls (seg_len < SPACING_MIN) get a single node at the midpoint.
    """
    if seg_len < SPACING_MIN:
        return [seg_len / 2.0]

    # Most gaps that keep the even step ≥ SPACING_MIN. floor(L / SPACING_MIN)
    # is the largest gap count whose even division does not fall below the min,
    # so the end nodes never overshoot the wall.
    n_gaps = max(1, int(seg_len // SPACING_MIN))
    step   = seg_len / n_gaps
    if step > SPACING_MAX:
        # Even division is wider than the band (short wall). Clamp the step to
        # the max and centre the shorter node run; span < seg_len so the end
        # nodes stay comfortably inside the wall.
        step = SPACING_MAX

    n_nodes = n_gaps + 1
    span    = step * n_gaps                          # length covered by nodes
    start   = (seg_len - span) / 2.0                 # symmetric end margins
    return [start + i * step for i in range(n_nodes)]


def _dedup(nodes: List[PathNode], radius: float) -> List[PathNode]:
    """
    Greedy de-duplication: keep a node only if no already-kept node lies within
    `radius`. Stops two walls meeting at a corner from producing two nodes a
    few centimetres apart. Kept nodes are renumbered so ids stay contiguous.
    """
    kept: List[PathNode] = []
    kept_xy: List[Tuple[float, float]] = []
    for n in nodes:
        x, y = n.position
        if any(math.hypot(x - kx, y - ky) < radius for kx, ky in kept_xy):
            continue
        kept.append(n)
        kept_xy.append((x, y))

    # Renumber ids contiguously so consumers get a clean PATH-0000.. sequence.
    for i, n in enumerate(kept):
        n.node_id = f"PATH-{i:04d}"
    return kept
