# Minus197 — Indoor Navigation System for Visually Impaired Users

**Group 38 Final-Year Research Project**
University of Moratuwa, Department of Information Technology

---

## Overall Project

Minus197 is an indoor navigation system designed to help visually impaired
people move independently through complex public buildings — initially shopping
malls. Sighted users navigate large indoor environments by reading signs,
following visual landmarks, and absorbing spatial layout at a glance. Visually
impaired users have none of these cues, and existing indoor navigation systems
built for sighted users do not translate well to their needs. A path that is
"shortest" for a sighted person may cross open halls with no tactile reference
points, leaving a visually impaired user disoriented and unable to confirm
their position.

Minus197 addresses this gap by building a complete navigation pipeline that
takes the architect's building model as ground truth, tracks the user's
position in real time, computes routes optimised for accessible navigation
behaviour, and delivers turn-by-turn guidance through voice and haptic
feedback.

The system is structured into four cooperating modules:

| Module | Responsibility |
|---|---|
| **Map Extraction** | Convert architectural IFC files into a navigation graph and an occupancy grid |
| **Perception** | Track the user's real-time position using a particle filter over sensor data |
| **Path Finding** | Compute optimal routes balancing distance, shore-lining, and landmark density |
| **Feedback** | Convert navigation instructions into voice prompts, spatial audio, and haptic cues |

The modules communicate through well-defined data contracts. Map Extraction
produces JSON outputs in the IFC project coordinate system. Perception aligns
its sensor frame to this same coordinate system through a one-time calibration.
Path Finding receives world coordinates from Perception and a destination node
ID, then returns a sequence of human-readable instructions. Feedback consumes
those instructions and produces the final user-facing guidance.

---

## My Contribution

My contribution covers two of the four modules: **Map Extraction** and
**Path Finding**. Together they form the spatial reasoning backbone of the
navigation system — without them, Perception has no map to localise against
and Feedback has no routes to deliver.

---

## Project Layout

```
minus197_mapping/
├── map_extraction/
│   ├── ifc_parser.py          # IFC entity extraction (Sprint 1)
│   ├── semantic_floor_map.py  # Zone classification and enrichment (Sprint 2)
│   ├── graph_builder.py       # Node placement and edge weighting (Sprints 3–4)
│   ├── occupancy_grid.py      # 5-state hybrid occupancy grid (Sprint 4)
│   ├── inter_floor_linker.py  # Multi-floor merging and admin tags (Sprint 7)
│   └── pipeline.py            # Orchestrates the full pipeline (Sprints 2–7)
├── pathfinding/
│   ├── engine.py              # PathfindingEngine — five-stage pipeline (Sprints 5–6)
│   ├── scorer.py              # score_path() and select_best() (Sprint 6)
│   └── node_resolver.py       # Destination resolution (Sprint 5)
├── shared/
│   └── types.py               # Core dataclasses shared across modules
├── perception_map.py           # Standalone helper delivered to Perception team
├── data/
│   ├── ifc_files/             # Input IFC files
│   └── outputs/               # Generated JSON outputs
├── tests/                     # pytest unit tests
├── audit/                     # Diagnostic scripts
├── main.py                    # CLI entry point
├── conftest.py                # pytest path setup
└── requirements.txt
```

---

## Tools and Libraries

| Library | Used in | Purpose |
|---|---|---|
| `ifcopenshell >= 0.8.0` | Map Extraction | IFC parsing and geometry extraction |
| `shapely >= 2.0` | Map Extraction, Perception helper | Exact 2D geometry operations |
| `scikit-image >= 0.21` | Map Extraction | Medial-axis skeletonisation |
| `scipy >= 1.11` | Both modules | KD-tree spatial indexing |
| `numpy >= 1.24` | Both modules | Numerical operations |
| `networkx >= 3.1` | Path Finding | Graph traversal and shortest paths |
| `sentence-transformers >= 2.2` | Path Finding | MiniLM embedding-based destination resolution (Tier 1, stub) |
| `rapidfuzz >= 3.0` | Path Finding | Fuzzy string matching for destination resolution (Tier 2) |
| `pytest >= 7.4` | Both modules | Unit tests |

---

## Map Extraction Module

### Purpose

Converts an architect's IFC building file into three structured digital
artefacts consumed by downstream modules. Geometric ground truth is extracted
exactly once from the authoritative source, then exposed to each consumer in
the format it actually needs.

### Inputs

A single IFC 2x3 or IFC4 file containing full 3D building geometry with
semantic information. Relevant entity types:

- `IfcSpace` — rooms and corridors
- `IfcWall` / `IfcWallStandardCase` — walls
- `IfcDoor` — doorways
- `IfcStair` / `IfcTransportElement` — vertical connectors (stairs, elevators, escalators)
- `IfcFurnishingElement` — landmarks (info desks, benches, etc.)

### Pipeline Stages

The pipeline runs in seven sequential stages, each in its own source file.

---

#### Stage 1 — IFC Parsing (`ifc_parser.py`)

Walks the IFC project hierarchy from `IfcProject` → `IfcSite` → `IfcBuilding`
→ `IfcBuildingStorey`, placing every entity into a single shared coordinate
frame and flattening from 3D to 2D.

**Geometry extraction** uses multiple fallback strategies per entity type:

- **Spaces** — `IfcExtrudedAreaSolid` → `ArbitraryClosedProfileDef`
  (`IndexedPolyCurve` or `Polyline`) or `RectangleProfileDef`
- **Walls** — Axis representation containing `IndexedPolyCurve` or `Polyline`;
  first and last coordinates become the wall axis start/end
- **Features** — World origin from `ObjectPlacement`

**Unit detection** reads `IfcSIUnit` entities and maps prefix + name to a
scale factor (e.g. `MILLI` + `METRE` → 0.001). Defaults to 1.0 (assumes
metres) when no unit declaration is found.

**Coordinate transformation** uses `ifcopenshell.util.placement.get_local_placement()`
to obtain the 4×4 world matrix, then applies it to each local 2D point.

**Output dataclasses**: `ParsedSpace`, `ParsedWall`, `ParsedFeature`,
`IFCParseResult`.

**Feature classification**:

| IFC type | Classification |
|---|---|
| `IfcTransportElement` | elevator / escalator by `PredefinedType` |
| `IfcStair` | stair |
| `IfcDoor` | door |
| `IfcFurnishingElement` | info\_desk / bench / furnishing by keyword match |

---

#### Stage 2 — Semantic Floor Map (`semantic_floor_map.py`)

Converts `IFCParseResult` into a `SemanticFloorMap` containing typed zones,
shore-tagged walls, and prioritised features.

**Zone classification** matches keywords against the architect-supplied
`Name` and `LongName` fields (case-insensitive, ASCII-normalised, includes
Russian transliteration):

| Category | Example keywords |
|---|---|
| `corridor` | corridor, hall, коридор |
| `shop` | shop, store, retail |
| `food_court` | food, restaurant, café |
| `restroom` | toilet, restroom, туалет |
| `entrance` | entrance, lobby |
| `exit` | exit |
| `storage` | storage, warehouse |
| `office` | office |
| `unknown` | (fallback) |

**Shore-linable tagging** — a wall is tagged `shore_linable = True` when any
of its start, midpoint, or end points falls within `SHORE_BUFFER = 1.0 m` of
a corridor polygon.

**Feature zone containment** — two-pass lookup: strict point-in-polygon first
(ray-casting), then 0.5 m buffer fallback.

**Feature priorities**:

| Priority | Types |
|---|---|
| 3 | elevator, escalator, stair |
| 2 | door, info\_desk |
| 1 | bench, furnishing |

**Output dataclasses**: `Zone`, `WallSegment`, `Feature`, `SemanticFloorMap`.

---

#### Stage 3 — Graph Construction (`graph_builder.py`)

Converts `SemanticFloorMap` into a `FloorGraph`. Uses a hybrid strategy:
rasterisation and medial-axis skeletonisation for junction *positions only*;
exact Shapely geometry for all *measurements* (distance, clearance, shore,
landmark density). Zero pixel error on any scored quantity.

**Node placement**:

- One node per feature (door, elevator, escalator, stair, landmark)
- One zone-centroid node per walkable zone
- Junction nodes at skeleton branch and endpoint positions after pruning

**Junction extraction**:

1. Rasterise walkable zones to a binary grid at `GRID_RES = 0.10 m`
2. Compute medial-axis skeleton with distance transform via `scikit-image`
3. Identify pixels of degree 1 (dead ends) or ≥ 3 (branch points)
4. Convert pixel positions to world coordinates
5. Prune to one junction per `MIN_NODE_SEP = 1.50 m` cluster (greedy KD-tree)

**Edge scoring** — all computed via Shapely on the actual walkable polygon:

| Weight | Implementation |
|---|---|
| `distance` | `LineString.intersection(walkable_polygon)` length; handles `MultiLineString` and `GeometryCollection` results |
| `safety_score` | Mean clearance from N sample points to walkable boundary, normalised by `MAX_CLEARANCE = 2.0 m` |
| `shore_linable` | True when ≥ `SHORE_FRACTION = 0.40` of N sample points fall within `SHORE_BUFFER = 0.80 m` of any zone perimeter |
| `landmark_score` | Count of feature nodes (door, elevator, escalator, stair, landmark) whose position lies within `LANDMARK_RADIUS = 3.0 m` of *any* of the N sample points along the edge, normalised as `min(count / 3, 1.0)` (3+ nearby features saturate to 1.0) |

> **What `landmark_score` measures.** It is the *density of landmarks along an
> edge* — how many navigable reference features an edge passes close to — not
> the number of graph nodes clustered *in front of* a single landmark. The
> direction is edge → nearby landmarks, not landmark → nearby nodes. Only the
> five feature node types above count as landmarks; junction nodes, zone
> centroids, and path nodes are never counted.

**Edge limits**: max 6 edges per node, 4 for zone-centroid nodes. Edges are
deduplicated by canonical sorted node-ID pair.

**Constants summary**:

| Constant | Value | Effect |
|---|---|---|
| `GRID_RES` | 0.10 m | Skeleton topology resolution |
| `MIN_NODE_SEP` | 1.50 m | Minimum pruned junction spacing |
| `SHORE_BUFFER` | 0.80 m | Wall-contact zone for shore scoring |
| `SHORE_FRACTION` | 0.40 | Fraction of edge samples needed for shore_linable |
| `LANDMARK_RADIUS` | 3.0 m | Feature influence radius |
| `MAX_CLEARANCE` | 2.0 m | Clearance above which safety_score = 1.0 |
| `MAX_EDGES` | 6 (4 centroids) | Neighbour limit per node |

---

#### Stage 4 — Occupancy Grid (`occupancy_grid.py`)

Generates a hybrid five-state occupancy grid for the Perception module.

**Cell states**:

| Value | Name | Meaning |
|---|---|---|
| 0 | `CELL_WALKABLE` | Definitely walkable — O(1) lookup |
| 1 | `CELL_WALL` | Definitely blocked — O(1) lookup |
| 2 | `CELL_DOOR` | Door threshold, passable — O(1) lookup |
| 3 | `CELL_OUTSIDE` | Outside building footprint — O(1) lookup |
| 4 | `CELL_UNCERTAIN` | Rasterisation boundary — requires exact Shapely test |

**Build pipeline**:

1. Allocate grid filled with `CELL_OUTSIDE`
2. Fill walkable zone polygons with `CELL_WALKABLE`
3. Buffer each wall segment by `WALL_THICKNESS_M = 0.15 m` and stamp `CELL_WALL`
4. Stamp door positions; snap to nearby wall if centroid falls inside a shop
   polygon (`DOOR_SNAP_M = 0.30 m`)
5. Mark uncertain boundary layer: any walkable cell with ≥ 1 wall neighbour
   (8-connected), or any wall cell touching a walkable cell
6. Cache a Shapely union of all walkable polygons as WKT in the output

**Resolution**: `DEFAULT_RESOLUTION_M = 0.05 m/cell`. For a 40 m × 20 m
floor this produces an 800 × 400 grid.

**JSON output** includes: floor metadata, cell type legend, per-type cell
counts, 2D row-major grid, WKT walkable polygon, coordinate-frame declaration,
and inline code hints for the Perception team.

---

#### Stage 5 — Inter-Floor Linking (`inter_floor_linker.py`)

Combines per-floor `FloorGraph` instances into a `BuildingGraph` with
bidirectional inter-floor edges between matched vertical connectors.

**Admin config injection** — an optional per-floor JSON block allows an
administrator to attach labels and accessibility flags to specific nodes
before the graph is saved:

```json
{
  "floor_label": "L1",
  "floor_height_m": 4.0,
  "nodes": {
    "FEAT-<guid>": {
      "admin_label": "Main Escalator",
      "is_accessible": true,
      "connects_to": ["L1", "L2"]
    }
  }
}
```

**Connector matching** — two vertical connectors of the same type on adjacent
floors are linked when their XY distance is ≤ `CONNECTOR_MATCH_RADIUS = 1.5 m`.
Bidirectional UP / DOWN edges are created with connector-dependent properties:

| Connector | safety_score |
|---|---|
| elevator | 1.0 |
| escalator | 0.8 |
| stair | 0.6 |

Inter-floor edge distance defaults to `DEFAULT_FLOOR_HEIGHT = 4.0 m` when not
supplied by admin config. `shore_linable` is always `False` for vertical
travel; `landmark_score` is always `1.0`.

---

#### Stage 6 — Pipeline Orchestration (`pipeline.py`)

`MapExtractionPipeline` orchestrates all previous stages for single-floor and
multi-floor runs.

**Single-floor workflow**:
1. `IFCParser.parse()` → `IFCParseResult`
2. `SemanticFloorMapBuilder.build()` → `SemanticFloorMap`
3. `GraphBuilder.build()` → `FloorGraph`
4. Optionally inject admin tags
5. Save `_graph.json`, `_sfm.json`, `_occupancy.json`

**Multi-floor workflow** (factory method `multi_floor()`):
1. Run single-floor pipeline for each IFC file sequentially
2. Register each `FloorGraph` with `InterFloorLinker`
3. `InterFloorLinker.build()` → `BuildingGraph`
4. Save `building_graph.json` + one `_occupancy.json` per floor

The graph JSON includes a `spatial_meta` block declaring units, axis
directions, bounding box, and skeleton grid resolution so downstream modules
can align their data correctly.

### Outputs

Three JSON files per floor, plus a standalone Python helper:

| File | Consumer | Contents |
|---|---|---|
| `<floor>_sfm.json` | Intermediate cache | Zones, walls, features in canonical form |
| `<floor>_graph.json` | Path Finding | Navigation graph with all nodes, edges, weights, and `spatial_meta` |
| `<floor>_occupancy.json` | Perception | Five-state grid, walkable WKT polygon, coordinate frame metadata |
| `perception_map.py` | Perception team | Standalone helper — `is_walkable(wx, wy)` and `crosses_wall(x1, y1, x2, y2)` |

### Key Design Decisions

**One coordinate system everywhere.** All outputs are in the IFC project
coordinate frame in metres, eliminating integration burden across modules.

**Hybrid occupancy resolution.** O(1) raster lookup for the ~97% of cells
that are unambiguously walkable or blocked; exact Shapely point-in-polygon
test for the thin uncertain boundary layer. Speed and geometric correctness
without compromise.

**Two-stage geometry in graph building.** Rasterisation and skeletonisation
are used only to locate junction positions. All scored quantities — distance,
clearance, shore fraction, landmark density — are computed from exact Shapely
operations on the original polygon. No pixel-level measurement error.

**Walkable distance, not Euclidean.** Edge distances are the actual
`LineString.intersection(walkable_polygon)` length, correctly accounting for
obstacles a straight line would cross.

**Clean perception interface.** `perception_map.py` depends only on `numpy`
and `shapely`. The Perception team installs it as a standalone file.

---

## Shared Types (`shared/types.py`)

Lightweight cross-module data contracts with no internal imports:

| Type | Description |
|---|---|
| `Point2D` | `Tuple[float, float]` in world metres |
| `NavigationNode` | `node_id`, `label`, `position`, `node_type`, `zone_id`, `tags` |
| `NavigationEdge` | `edge_id`, `source_id`, `target_id`, `distance`, `shore_linable`, `safety_score`, `landmark_score`, `tags` |
| `FloorGraph` | `floor_label`, `source_file`, `nodes`, `edges` + node index + lookup helpers |
| `BuildingGraph` | `building_name`, `floors`, `inter_floor_edges` + flat node/floor indexes + `all_nodes` property |

---

## Perception Helper (`perception_map.py`)

Standalone class delivered directly to the Perception team. No dependency on
any other mapping module file.

```python
occ = OccupancyMap("mall_L1_occupancy.json")

occ.is_walkable(wx=12.4, wy=8.1)          # True / False
occ.crosses_wall(x1, y1, x2, y2)          # True if segment crosses a wall
```

**`is_walkable(wx, wy)`** — converts world coordinates to grid cell, performs
O(1) lookup for hard cell types (0/1/2/3), and falls back to exact Shapely
`Point.within(walkable_polygon)` for uncertain boundary cells (type 4).

**`crosses_wall(x1, y1, x2, y2)`** — Bresenham line walk over the grid;
O(1) for hard cells, Shapely intersection for uncertain cells.

---

## Path Finding Module

### Purpose

Computes walking routes genuinely usable by a visually impaired person. A
standard shortest-path algorithm optimises only for distance, which often
produces routes that are fast for a sighted user but disorienting for someone
relying on cane trailing and verbal landmarks. This module treats navigation
as a multi-objective problem.

### Inputs

| Input | Source | Format |
|---|---|---|
| Navigation graph | `_graph.json` from Map Extraction | `FloorGraph` loaded once at startup |
| Current location | Perception module | `(wx, wy)` world coordinates in metres |
| Destination | Spoken query, resolved upstream or by NodeResolver | String query or node ID |

### Pipeline Stages

#### Stage 1 — Start Node Resolution (`engine.py`, `node_resolver.py`)

A KD-tree is built once at startup over all navigable node positions
(junctions, doors, stairs, elevators, escalators). Zone centroids are excluded
— they are destinations, not waypoints. Nearest-neighbour lookup resolves the
user's exact world position to a graph node in microseconds. The user's exact
position is preserved separately so the first instruction guides them from
their actual location to the resolved start node.

#### Stage 2 — Destination Resolution (`node_resolver.py`)

Three-tier fallback resolution, stopping at the first confident match:

| Tier | Method | Threshold | Status |
|---|---|---|---|
| 1 | Sentence embedding cosine similarity (MiniLM `all-MiniLM-L6-v2`) | ≥ 0.65 | Stub — returns `None` |
| 2 | Fuzzy string match (`rapidfuzz.WRatio`, `difflib` fallback) | ≥ 75 / 0.5 | Implemented |
| 3 | Category keyword match with node-type priority | substring | Implemented |

A pre-processing step strips navigation intent preamble ("I want to go to",
"take me to", "navigate to", "find", etc.) before matching.

**Tier 3 category aliases**:

| Query keyword | Target category |
|---|---|
| shop, store | shop |
| food, restaurant, café | food\_court |
| toilet, restroom, bathroom | restroom |
| entrance, lobby | entrance |
| exit | exit |
| corridor, hall | corridor |

When multiple nodes match, Tier 3 picks the highest node-type priority
(elevator/escalator/stair = 3, door/landmark = 2, zone\_centroid = 1,
junction = 0).

**`rapidfuzz`** is the primary Tier 2 library; the resolver gracefully falls
back to `difflib.get_close_matches` if `rapidfuzz` is not installed.
**`sentence-transformers`** is lazily loaded and skipped without error if
unavailable.

#### Stage 3 — K Shortest Paths (`engine.py`)

Generates the top `k = 15` candidate routes between start and destination,
weighted by walking distance. The current implementation uses a single greedy
Dijkstra path (stub). The planned implementation uses `networkx.shortest_simple_paths()`
(Yen's K-shortest loopless paths algorithm).

#### Stage 4 — Scoring and Selection (`scorer.py`)

Each candidate path receives three distance-weighted scores:

```
safety_score  = Σ(edge.safety_score  × edge.distance) / Σ(edge.distance)
shore_score   = Σ(float(edge.shore_linable) × edge.distance) / Σ(edge.distance)
landmark_score = Σ(edge.landmark_score × edge.distance) / Σ(edge.distance)
```

A composite score combines them:

```
composite = 0.40 × safety + 0.35 × shore + 0.25 × landmark
```

**Selection** (two-step):

1. Filter to paths with `composite ≥ TOP_TIER_THRESHOLD = 0.55` (the top tier)
2. Among the top tier, choose the path with the shortest total distance
3. Return up to `MAX_ALTERNATIVES = 3` runner-up paths

The current stub returns the first scored path directly; the top-tier
filtering step is implemented in `scorer.py` and wired in as the planned
`_select_path()` replacement.

**Scoring constants** (Sprint 8 calibration targets):

| Constant | Value |
|---|---|
| `W_SAFETY` | 0.40 |
| `W_SHORE` | 0.35 |
| `W_LANDMARK` | 0.25 |
| `TOP_TIER_THRESHOLD` | 0.55 |
| `MAX_ALTERNATIVES` | 3 |

#### Stage 5 — Instruction Generation (`engine.py`)

Each consecutive pair of edges produces a bearing change mapped to a
natural language instruction:

| Bearing change | Instruction |
|---|---|
| −20° to +20° | "Continue straight" |
| +20° to +100° | "Bear right" |
| +100° to +170° | "Turn right" |
| Beyond ±170° | "Turn around" |
| −100° to −20° | "Bear left" |
| −170° to −100° | "Turn left" |

A landmark hint ("You will reach {label}") is appended when the next node
is a door, stair, elevator, escalator, or any node carrying a non-empty
`admin_label` tag.

Bearing is computed clockwise from north: `atan2(dx, dy)` converted to
degrees in [0, 360).

### Outputs

A `PathResult` object containing:

- Resolved start and destination nodes
- Ordered list of `PathStep` objects (`from_node`, `to_node`, `bearing`,
  `distance`, `instruction`)
- Total walking distance
- Three composite quality scores (safety, shore, landmark)
- Up to three alternative paths from the top tier
- `found` boolean — `False` if no path exists

### Key Design Decisions

**Multi-objective optimisation, not just shortest path.** The system
explicitly models trade-offs that matter for visually impaired navigation
rather than treating distance as the only objective.

**Top-tier then shortest.** Selection cannot pick a marginally longer path
purely because it scored fractionally higher on quality, nor can it pick a
much shorter path that sacrifices safety, shore, and landmark properties.

**Tunable composite weights.** The 0.40 / 0.35 / 0.25 weights are constants
in the current build, designed to be calibrated through user trials with
visually impaired participants.

**Node ID contract, not free-text.** Destination resolution from spoken
phrases to node IDs is a contained responsibility in `NodeResolver`.
`PathfindingEngine.find_path()` accepts either a pre-resolved node ID or a
natural language query string.

---

## CLI Entry Point (`main.py`)

```bash
# Single floor
python main.py --ifc data/ifc_files/mall_L1.ifc --floor L1

# Single floor with pathfinding demo
python main.py --ifc data/ifc_files/mall_L1.ifc --floor L1 \
               --query "food court"

# Multi-floor
python main.py --ifc data/ifc_files/mall_L1.ifc --floor L1 \
               --ifc data/ifc_files/mall_L2.ifc --floor L2
```

Outputs are written to `data/outputs/`.

---

## Implementation Status

| Component | Status |
|---|---|
| IFC parsing (Sprint 1) | Complete — validated on real IFC files |
| Semantic floor map (Sprint 2) | Complete |
| Graph construction with skeletonisation (Sprints 3–4) | Complete |
| Edge weighting — all four weights (Sprint 4) | Complete |
| Occupancy grid with hybrid resolution (Sprint 4) | Complete |
| Inter-floor linking with admin tags (Sprint 7) | Complete |
| `perception_map.py` standalone helper | Complete and validated |
| Path scoring and top-tier selection (Sprint 6) | Complete (`scorer.py`) |
| Destination resolution — Tier 2 fuzzy, Tier 3 keyword (Sprint 5) | Complete (`node_resolver.py`) |
| Instruction generation with bearing and landmark hints (Sprint 5) | Complete (`engine.py`) |
| Start node KD-tree resolution | Complete (`engine.py`) |
| K shortest paths via Yen's algorithm (Sprint 5) | Stub — single greedy Dijkstra path |
| Top-tier path selection wired into engine (Sprint 6) | Stub — returns first scored path |
| Destination resolution — Tier 1 embedding similarity | Stub — `sentence-transformers` model loaded but cosine lookup not implemented |
| Merchant-name admin labelling | Designed, not yet implemented |
| Multi-floor `spatial_meta` propagation | Known limitation |

---

*Project: Minus197 — Group 38*
*University of Moratuwa — Final Year Research Project*
