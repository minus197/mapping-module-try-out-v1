
## `_sfm.json` — Semantic Floor Map

The base geometric/semantic model everything else is derived from.

- **`zones[]`** — rooms/shops/corridor, each with `category` (corridor, shop, restroom, unknown, ...), `centroid`, and a full `boundary_polygon`.
- **`walls[]`** — wall segments (`start`/`end`), flagged `shore_linable` if they border a corridor.
- **`features[]`** — doors, elevators, stairs, etc. Always have a `position`. Elevators/stairs additionally carry a `boundary_polygon` (their real footprint) when extractable from the IFC body geometry; doors/furnishing stay point-only.

## `_graph.json` — Navigation Graph

The routable graph used for pathfinding.

- **`nodes[]`** — one per zone centroid (`ZONE-<id>`) and per feature (`FEAT-<id>`), plus `junction` nodes from corridor skeletonisation. Each has `position`, `node_type` (door/elevator/stair/zone_centroid/junction/landmark), and `tags`.
- **`edges[]`** — connections between nodes, with `distance`, `shore_linable`, `safety_score`, and `landmark_score`.

## `_occupancy.json` — Occupancy Grid

A 5-state raster of the floor, for grid-based queries/collision checks.

- **`grid`** — 2-D array of cell values: `0` walkable, `1` wall, `2` door_threshold, `3` outside_building, `4` uncertain_boundary (see `cell_legend`).
- **`resolution_m`**, **`origin`**, **`width_cells`/`height_cells`** — how to map grid indices to world coordinates.
- **`walkable_wkt`** — the walkable area as a WKT MULTIPOLYGON (for exact geometric queries instead of the raster).

## `_path_nodes.json` — Path Nodes

Cane-trailing waypoints ~0.3–2 m off corridor-facing walls, spaced ~5.5–6 m apart, for tactile/shore-line navigation. No edges — positions only.

Each node in **`path_nodes[]`** has:
- `position`, `wall_id` — which wall it shore-lines against, `gap_m` — its intended offset.
- `nearest_wall_dist_m` (**x**) — the *true* distance to the closest wall (usually ≈ `gap_m`, but smaller near dead ends/corners).
- `search_circle` — radius `2·x` around the node, with `hits[]`: every non-corridor zone/room and every feature (door/landmark) whose boundary falls inside, each with `node_id` (joins to `_graph.json`), `perp_dist_m` (distance to its boundary), and `centroid_dist_m` (distance to its centroid).


