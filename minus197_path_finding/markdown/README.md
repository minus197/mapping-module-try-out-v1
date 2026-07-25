# minus197_path_finding

Route generation for the Minus197 indoor navigation system for visually impaired
users. Takes a navigation graph produced by `minus197_mapping`, finds a route
between two nodes, then rewrites the walking geometry to hug the cane-trailing
path-node layer and emits turn-by-turn instructions.

**Group 38 — University of Moratuwa**

---

## Contents

- [Pipeline](#pipeline)
- [Module map](#module-map)
- [Quick start](#quick-start)
- [Current findings — open defects](#current-findings--open-defects)
- [Documents](#documents)

---

## Pipeline

A route is produced in two distinct stages. Keeping them separate matters for
reading any of the findings below.

**Stage 1 — graph routing** (`pathfinding/engine.py`)
Dijkstra/Yen's over the `FloorGraph` from `minus197_mapping`. Operates on
junctions, doors and zone centroids. Produces the *topological* route: which
corridors and doorways the walk passes through.

**Stage 2 — path-node adjustment** (`pathfinding/path_node_adjuster.py`)
Discards stage 1's geometry and re-derives the walking line by a second
shortest-path search over a mixed graph of cane-trailing path nodes plus the
mandatory anchors (start, destination, every door on the route). Costs:

| hop kind | cost |
|---|---|
| same-face hop (along one wall) | real Euclidean distance |
| corner bridge (mutual-NN endpoints of different walls) | real Euclidean distance |
| **crossing** (leaving one wall face for another) | **flat `BETA = 8.0`** |
| original-route fallback edge | distance × `ORIGINAL_EDGE_PENALTY_FACTOR` (5.0) |

Stage 2 replaces only `steps`. All scores/costs on the returned `PathResult`
still describe the stage-1 graph route.

---

## Module map

| File | Role |
|---|---|
| `pathfinding/engine.py` | Stage-1 integrator; `find_path()`, `to_feedback_json()` |
| `pathfinding/path_node_adjuster.py` | Stage-2 geometry rewrite over the path-node layer |
| `pathfinding/search.py` | Dijkstra / Yen's k-best over the networkx graph |
| `pathfinding/cost.py` | Edge cost model (λ/μ/ν weights) |
| `pathfinding/scorer.py` | Safety / shoreline / landmark path scores |
| `pathfinding/instructions.py` | `build_steps()` — bearings, turn phrases, spoken text |
| `pathfinding/node_resolver.py` | KD-tree resolution of a raw position to a graph node |
| `pathfinding/multi_floor.py` | Inter-floor routing, stitched at the elevator |
| `shared/types.py` | `NavigationNode` / `NavigationEdge` / `FloorGraph` / `PathResult` |
| `evaluation/` | Endpoint checks over a produced route |
| `visualizer/` | Read-only PNG / interactive renderers |

---

## Quick start

Run from the `minus197_path_finding/` directory. Full command reference in
[RUN_COMMANDS_PATHFINDING.md](RUN_COMMANDS_PATHFINDING.md).

```bash
# Route with the path-node adjustment (both flags required together)
python run_pathfinding.py \
  --graph      ../minus197_mapping/data/outputs/itfac_mall_floors_3-4_combined_L4_graph.json \
  --path-nodes ../minus197_mapping/data/outputs/itfac_mall_floors_3-4_combined_L4_path_nodes.json \
  --sfm        ../minus197_mapping/data/outputs/itfac_mall_floors_3-4_combined_L4_sfm.json \
  --from ZONE-2aQOv7Awv2XAeFI_6R_KbE --to ZONE-233bRpVtr7PADnPMIOrRa2
```

The **reference route** used throughout the findings below is that command —
ODEL → Popeyes on `itfac_mall_floors_3-4_combined_L4`.

---

## Current findings — open defects

Measured 2026-07-26 on the reference route.

> **Findings 2 and 3 are now fixed** (implemented 2026-07-26); finding 4 is
> partially addressed and findings 1 and 5 stand. See
> [LAST_MILE_FIX.md §9](LAST_MILE_FIX.md) for results and the factor sweep.
> The reference route now walks **135.28 m with 0.00 m unreferenced** (was
> 130.91 m with 11.25 m), and floor-wide unreferenced distance fell from
> 18.28 % to 15.13 % across all 90 L4 centroid pairs.
>
> The status lines below are kept as the *diagnosis of record*; each says what
> was found, not what remains outstanding.

> **Method note.** These were found by dumping the adjuster's *internal winning
> chain*, not by reading the emitted feedback JSON. The original diagnosis was
> written from the JSON and was wrong for that reason — see finding 1.

### 1. `to_feedback_json()` hides the final door — output-format bug

**Status:** confirmed · **Severity:** high (user-visible) · `engine.py:282`

`to_feedback_json()` anchors each action at `step.from_node`, except the last,
which anchors at `step.to_node`:

```python
anchor = step.to_node if is_last else step.from_node
```

On a route ending `… → PATH-0126 → DOOR → CENTROID`, this emits the
`PATH-0126 → DOOR` hop under the label `PATH-0126`, then consumes the door's own
row as the anchor of the terminal `stop`. **The door never appears in the
output**, even though the route walks to it.

This is what made the route look like it terminated on a bare zone centroid. It
does not. The centroid is reached via the door, correctly.

Actual chain tail on the reference route:

```
PATH-0129 → PATH-0126                      11.25 m
PATH-0126 → FEAT-0NrdkfQ69FhA8iVi0WZ06i     3.17 m   ← door, traversed
FEAT-0Nrd… → ZONE-233bRpVtr7PADnPMIOrRa2    5.80 m
```

**Implication:** `_mandatory_nodes()` and `_reconstruct_route_nodes()` both work
as intended. There is no missing-portal defect, and no "19.1 m two-legged
diagonal" — the real tail is 14.42 m and terminates at a physical doorway.

### 2. Flat-`BETA` crossing edges short-circuit connected wall routes

**Status:** confirmed · **FIXED 2026-07-26** · **Severity:** high (the real
routing defect) · `path_node_adjuster.py:393`, `BETA` at `:97`

`_build_crossing_edges()` prices every cross-face hop at a flat `BETA = 8.0`
**without checking whether a wall-hugging route between the same two nodes
already exists.** When one does and it is longer than 8.0, the search takes the
open-floor diagonal.

On the reference route:

```
chosen:    PATH-0129 → PATH-0126      11.25 m straight, cost 8.0 (flat BETA)
           midpoint 4.92 m from the nearest wall — genuinely unreferenced

available: PATH-0129 → PATH-0128       5.65 m  same face
           PATH-0128 → PATH-0127       3.97 m  corner bridge
           PATH-0127 → PATH-0126       6.00 m  same face
                                     ────────
                                      15.62 m  all at wall_d ≈ 1.8–2.0 m
```

8.0 < 15.62, so the diagonal wins. The crossing edge is **not bridging a gap** —
it short-circuits a corner that is already fully connected in the path graph.

**Scale across L4** — of 186 crossing edges generated:

| | count | meaning |
|---|---|---|
| genuinely bridge a disconnected pair | 140 | BETA doing its intended job |
| **short-circuit an already-connected route** | **38** | **the defect** |
| connected route already cheaper than BETA | 8 | harmless |

**Do not simply raise or remove `BETA`** — 140 edges depend on it to bridge
faces that have no wall route at all. The fix must distinguish *"no wall route
exists"* from *"a wall route exists but is longer"*: e.g. suppress the crossing
edge when a path-only route between its endpoints already exists within some
multiple of BETA.

The reference route's instance is mid-severity — 15th of 38 by distance saved.
The worst is `PATH-0120 → PATH-0135`, short-circuiting a **62.57 m** wall route
for a flat 8.0.

### 3. Synthetic edges falsely claim `shore_linable=True`

**Status:** confirmed · **FIXED 2026-07-26** · **Severity:** high (corrupts all
measurement) · `path_node_adjuster.py:733-749`

`_add_synthetic_edge()` stamps `shore_linable=True` and `safety_score=1.0` on
**every** synthetic hop unconditionally, including open-floor diagonals.

Measured on the reference route: **21 of 21 hops report `shore_linable=True`** —
including the 11.25 m diagonal sitting 4.92 m from any wall.

**Any shoreline or safety metric computed from adjusted-route edges is therefore
falsified.** Fix this before recording any baseline, or before/after comparisons
mean nothing.

Once computed honestly (taking "midpoint > 4 m from nearest wall" as
unreferenced), the true baseline is:

```
total walked          130.91 m
genuinely unreferenced 11.25 m   (one hop — finding 2)
```

That is a *better* baseline than previously assumed, which shrinks the
improvement headroom any evaluation can claim.

### 4. `PATH_GAP_M` docstring is stale — says 0.3 m, is 2.0 m

**Status:** confirmed · **Severity:** medium (misleads every reader) ·
`../minus197_mapping/map_extraction/path_nodes.py:92`

```python
PATH_GAP_M = 2.0   # m — gap between a wall and its path node
```

The module docstring states 0.3 m at **lines 13, 107 and 242**. Emitted L4 data
confirms 2.0 m (nodes measure 1.8–2.0 m off their wall).

2.0 m is a wide offset for cane trailing — a cane user tracking a wall is
typically much closer. Whether 2.0 m is intentional or a regression from 0.3 m
is **an open question worth resolving**, since it directly sets how far every
route sits from its reference surface.

### 5. Portal machinery is not justified by current data

**Status:** assessed, not a defect · **Severity:** n/a

[LAST_MILE_FIX.md](LAST_MILE_FIX.md) §3.1/§3.2 propose a `portal_resolver.py`
plus a two-segment Dijkstra to force door termination. Measurement says this
solves nothing currently broken:

- 10 of 11 L4 zone centroids already have **exactly one** door edge (the 11th
  has none — the genuine fallback case);
- the router already traverses that door (finding 1).

It would convert an accident into a guarantee, which has some value, but it is
not the fix for the observed geometry.

**Also: §3.1's rule 2 can never fire.** It selects doors "whose `zone_id`
matches the centroid's zone". All 10 L4 door nodes carry
`zone_id = "1i$2JbPrrAMOUePJPoCKpR"` — the storey/corridor zone — while each
centroid carries its own shop zone id. No door's `zone_id` will ever match a
shop centroid's. If built anyway, drop rule 2 or re-source it from the door's
`connects_to` tag.

### Build order — status

1. ~~**Honest synthetic edges** (finding 3)~~ — **done.** `shore_linable` is now
   measured by sampling along each hop; `shore_fraction` recorded in
   `edge.tags`.
2. ~~**Crossing-edge short-circuit suppression** (finding 2)~~ — **done.**
   `CROSSING_SHORTCUT_FACTOR = 3.0`; 34 of 38 short-circuits suppressed, all 140
   legitimate bridges retained, connectivity intact (146/146).
3. **Arrival phrasing / feedback anchoring** (finding 1) — *outstanding.*
   Largely a `to_feedback_json()` fix, not a routing one. Note it changes the
   output contract: one extra action per route plus a door row.
4. **Resolve `PATH_GAP_M`** (finding 4) — *outstanding.* `SHORE_MAX_D` is now
   derived from it (`1.25 × PATH_GAP_M`), so the decision propagates
   automatically once made.
5. Portal machinery (finding 5) — only if a floor appears where a *shop*
   centroid has multiple or zero door edges.

---

## Documents

| File | What it covers |
|---|---|
| [RUN_COMMANDS_PATHFINDING.md](RUN_COMMANDS_PATHFINDING.md) | Every CLI invocation — single floor, multi-floor, visualizers |
| [ACCEPT_PATH_NODE_AS_CURRENT_NODE.md](ACCEPT_PATH_NODE_AS_CURRENT_NODE.md) | Using a `PATH-xxxx` id as the current node |
| [LAST_MILE_FIX.md](LAST_MILE_FIX.md) | Door-terminated routing proposal. **§0–§8 are superseded** — read §−1 first; §0–§8 retained for audit |

### Caveats on the reference docs

- **`ROUTE_QUALITY_FACTORS.md` is not in this repository.** `LAST_MILE_FIX.md`
  cites it throughout (§5, §7.1, §7.2, §8, §9, §10) as if present. Those
  cross-references cannot currently be checked.
- **`instruction_compressor.py` does not exist.** `LAST_MILE_FIX.md` §6 build
  item 3 refers to it as though it were an existing file.

---

## Tests

```bash
python -m pytest                       # whole suite
python -m pytest tests/test_path_node_adjuster.py
```

**Suite is green: 191 passed** (was 23 failed, 162 passed).

The 23 failures were all in `tests/test_wp6_engine.py`, calling the retired
positional signature `find_path(x, y, dest_id)` against the current
`find_path(start_node_id, destination_node_id)` (`engine.py:98`). Fixed by
passing node ids. One test —
`test_first_step_is_user_segment_when_far` — relied on an off-node user position
that the node-id API cannot express; it was replaced by
`test_first_step_is_user_segment_when_starting_on_a_path_node`, which exercises
the same `USER` virtual-first-step behaviour through the path-node entry point.

Findings 2 and 3 are now covered by `TestCrossingShortCircuit` and
`TestSyntheticEdgeShoreHonesty` in `tests/test_path_node_adjuster.py` (6 tests).
Both previously passed silently.
