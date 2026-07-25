# Accepting a Path Node ID as the Current (Start) Node

## 1. What we need to do

### Goal
Today, `PathfindingEngine.find_path(current_node_id, destination_node_id)` only
works when `current_node_id` is a **graph node** (junction / door / stair / lift /
zone_centroid). If a **path node** id is passed as the current node, the call
returns an empty result.

We want `find_path` to accept **either kind of id** as the current node,
transparently, through the same call:

- a **non-path (graph) node id** → behaves exactly as it does today (unchanged);
- a **path node id** → resolves to the nearby graph route, then delivers a route
  that starts exactly at that path node.

An **unknown id** (neither) still returns an empty result, as today.

### Why it doesn't work today
Path nodes are **not part of the `FloorGraph`** and have **no edges**. They live
in a separate `*_path_nodes.json` file and are only loaded *inside* the adjuster
(phase 2). So the very first lookup in `find_path`:

```python
start_node = self.graph.node(start_node_id)   # None for a path node id
if start_node is None:
    return _empty_result()                     # <-- bails out here
```

...returns `None` for a path node id, before any search runs. Path nodes also
never appear in the networkx search graph (`build_nx_graph`), so there is nothing
to search from either.

### The approach (why this one)
A path node is fundamentally a **world position**. We already own a component
whose entire job is *"given a world position, find the graph node to start
from"*: `StartNodeResolver` (`node_resolver.py`). So instead of injecting path
nodes into the `FloorGraph` (which would change every route's cost/scoring and
bloat the search graph), we:

1. recognise a path node id and look up its position;
2. resolve that position to the nearest wall-clear graph node (phase 1 routing);
3. pin the path node as the true start in phase 2 (the adjuster), so the final
   turn-by-turn route begins exactly on it.

This keeps the two-graph separation intact and leaves all existing non-path-node
routes **byte-for-byte identical**.

---

## 2. The changes needed, and in which files

Three changes across two source files, plus one caller-wiring update.

### Change 1 — recognise & resolve a path node id in `find_path`
**File:** `pathfinding/engine.py` — method `PathfindingEngine.find_path`
(around lines 103–111).

**Now:**
```python
start_node = self.graph.node(start_node_id)
if start_node is None:
    return _empty_result()
user_xy: Point2D = start_node.position
```

**Change to:**
```python
start_node = self.graph.node(start_node_id)
if start_node is not None:
    # Non-path (graph) node — unchanged behaviour.
    user_xy: Point2D = start_node.position
else:
    # Not a graph node — try to treat it as a path node.
    pn = self._path_node_index.get(start_node_id)   # supplied by Change 2
    if pn is None:
        return _empty_result()                       # genuinely unknown id
    # Resolve the path node's world position to a real graph start node,
    # exactly as we resolve a raw user position. user_xy stays the path
    # node's true position so build_steps' first step starts on it.
    start_node, user_xy = self._resolver.resolve(pn.position[0], pn.position[1])
```

**Effect:** phase 1 (graph route) now runs from the nearest wall-clear graph node
to the path node, while `user_xy` remains the path node's true coordinates.

---

### Change 2 — give the engine access to path nodes (lookup index)
**File:** `pathfinding/engine.py` — `PathfindingEngine.__init__`
(constructor, around lines 57–82).

Add an optional `path_nodes` parameter and build an id → PathNode index:

```python
def __init__(
    self,
    graph:        FloorGraph,
    wall_checker: WallChecker,
    weights:      CostWeights = None,
    landmark_max: Optional[float] = None,
    path_nodes:   Optional[List["PathNode"]] = None,   # NEW
) -> None:
    ...
    # NEW — read-only lookup so find_path can resolve a path-node id.
    self._path_node_index = {pn.node_id: pn for pn in (path_nodes or [])}
```

Add the import for the `PathNode` type at the top of `engine.py`:
```python
from pathfinding.path_node_adjuster import PathNode
```

**Effect:** pure plumbing — a read-only dictionary. Adds **no** nodes/edges to
any graph. Non-path-node routes are completely unchanged. This is what makes a
path node id *recognisable* in Change 1 (without it, a path node id looks like an
unknown id).

---

### Change 3 — pin the path node as the adjuster's start (accuracy)
**File:** `pathfinding/path_node_adjuster.py` — function `adjust_with_path_nodes`
and helper `_mandatory_nodes` (around lines 439 and 597).

Pass the input path node id through to the adjuster and add it to the mandatory
anchor set so the mixed-graph Dijkstra's `start_id` is the path node itself:

- add an optional `start_path_node_id: Optional[str] = None` parameter to
  `adjust_with_path_nodes`;
- when set, ensure that path node is included in `_mandatory_nodes(...)` and used
  as `start_id` for `_dijkstra_mixed(...)` (instead of `route_nodes[0]`).

**Effect:** the final walking geometry begins **exactly on the input path node**,
eliminating the only accuracy risk from Change 1 (a possible short backtrack when
the path node sits just past a junction). Phase 1's junction snap only decides
*which corridors*, which a one-junction offset does not change.

> Change 3 is **required, not optional**, if path-node starts must be accurate.
> Change 1+2 alone let you *pass* a path node id; Change 3 makes the result
> trustworthy.

---

### Caller wiring — pass `path_nodes` into the engine
**File:** `run_pathfinding.py` (and any other caller that constructs the engine),
around lines 198 and 207–211.

The path nodes are already loaded for the adjuster; they now also need to reach
the engine constructor. Load them **before** building the engine:

**Now:**
```python
engine = PathfindingEngine(graph, _no_wall_checker)
result = engine.find_path(start_id, dest_id)
...
if args.path_nodes_path and args.sfm_path:
    path_nodes = load_path_nodes(args.path_nodes_path)
    walls = load_corridor_walls(args.sfm_path)
    zones = load_zones(args.sfm_path)
    result = adjust_with_path_nodes(result, path_nodes, walls, zones=zones)
```

**Change to:**
```python
path_nodes = load_path_nodes(args.path_nodes_path) if args.path_nodes_path else None
engine = PathfindingEngine(graph, _no_wall_checker, path_nodes=path_nodes)   # Change 2 wiring
result = engine.find_path(start_id, dest_id)
...
if args.path_nodes_path and args.sfm_path:
    walls = load_corridor_walls(args.sfm_path)
    zones = load_zones(args.sfm_path)
    # Change 3: pin the start path node when the current node IS a path node.
    result = adjust_with_path_nodes(
        result, path_nodes, walls, zones=zones,
        start_path_node_id=(start_id if start_id in {p.node_id for p in path_nodes} else None),
    )
```

> If `path_nodes` is not wired in, a path-node id silently falls through to an
> empty result (looks like "unknown id"). Consider a one-line warning at
> construction when `_path_node_index` is empty, to make a misconfigured caller
> obvious.

---

## Summary table

| # | Change | File | Accuracy impact |
|---|--------|------|-----------------|
| 1 | Resolve a path-node id to a graph start via `StartNodeResolver` | `pathfinding/engine.py` (`find_path`) | The only risk: possible short backtrack if the path node is just past a junction — fixed by Change 3 |
| 2 | Add `path_nodes` param + `_path_node_index` lookup | `pathfinding/engine.py` (`__init__`) | **None** — read-only lookup, graphs unchanged, non-path routes identical |
| 3 | Pin the input path node as the adjuster's start | `pathfinding/path_node_adjuster.py` (`adjust_with_path_nodes`, `_mandatory_nodes`) | **Improves** accuracy — route starts exactly on the path node |
| — | Pass `path_nodes` into the engine constructor | `run_pathfinding.py` (+ any other caller) | Wiring only — required for Changes 1–3 to activate |

### End state
After all three changes, a single `find_path(current_node_id, destination_node_id)`
call accepts:

- **any graph node id** (junction/door/stair/lift/zone) → unchanged;
- **any path node id** → resolved and pinned, route starts exactly on it;
- **an unknown id** → empty result.
