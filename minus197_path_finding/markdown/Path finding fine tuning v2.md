# Path Finding Fine Tuning v2

**Part of:** Minus197 — Indoor Navigation System for Visually Impaired Users
**Module:** `minus197_path_finding`
**Branch:** `path-finding-v3`
**Date:** 2026-07-26
**Status:** implemented, uncommitted (working tree)

---

## 0. What this round changed

Two routing defects were fixed in `pathfinding/path_node_adjuster.py`, both
found while investigating why the ODEL → Popeyes route on
`itfac_mall_floors_3-4_combined_L4` cut diagonally across open floor instead of
following a wall.

| # | Change | Kind |
|---|---|---|
| 1 | Synthetic edges now **measure** `shore_linable` instead of asserting it | measurement integrity |
| 2 | Crossing edges are **suppressed** when a wall route already connects the pair | routing |
| 3 | 23 stale `find_path()` calls migrated to the node-id API | test maintenance |

Change 1 had to land first: every metric used to validate change 2 was being
read off edge data that was falsified by default.

The full diagnosis, including the measurement method and a superseded earlier
analysis, is in [`LAST_MILE_FIX.md`](LAST_MILE_FIX.md). This document covers
what was built and what it did to the numbers.

---

## 1. Honest `shore_linable` on synthetic edges

**File:** `pathfinding/path_node_adjuster.py` — `_add_synthetic_edge()`, `_shore_fraction()`

`_add_synthetic_edge()` previously stamped `shore_linable=True` on every
synthetic hop unconditionally. On the reference route that meant **21 of 21 hops
claimed to be wall-hugging**, including an 11.25 m diagonal whose midpoint sat
4.92 m from the nearest wall.

Shore-linability is now sampled along the hop:

```python
frac = _shore_fraction(pn_graph, src.position, tgt.position)
edge = NavigationEdge(
    ...,
    shore_linable=frac >= SHORE_FRACTION,
    tags={"shore_fraction": str(round(frac, 3))},
)
```

`_add_synthetic_edge()` gained a `pn_graph` parameter to reach the wall geometry;
the single call site in `adjust_with_path_nodes()` was updated to pass it.

### New constants

```python
PATH_GAP_M     = 2.0                 # mirrors map_extraction/path_nodes.py
SHORE_MAX_D    = 1.25 * PATH_GAP_M   # 2.5 m
SHORE_FRACTION = 0.40                # mirrors graph_builder.SHORE_FRACTION
SHORE_SAMPLES  = 9                   # endpoints included
```

Two deliberate choices here:

- **`SHORE_MAX_D` is derived from `PATH_GAP_M`, not a literal.** Path nodes sit
  2.0 m off their wall, so the threshold has to track that constant if it is
  ever retuned. Upstream `graph_builder.SHORE_BUFFER` (0.80 m) could not be
  reused — at a 2.0 m gap it would report *every* hop as unshored, trading a
  uniformly optimistic metric for a uniformly pessimistic one.
- **Sampling, not a midpoint test.** A hop that starts hard against a wall and
  ends in open floor has a midpoint that can land either side of the threshold;
  one sample cannot separate that case from a genuine wall run.

`shore_fraction` is also written into `edge.tags`, mirroring what
`graph_builder` does for real edges, so the boolean is auditable.

---

## 2. Crossing-edge short-circuit suppression

**File:** `pathfinding/path_node_adjuster.py` — `_build_crossing_edges()`, `_path_only_distance()`

`_build_crossing_edges()` priced every cross-face hop at a flat `BETA = 8.0`
without checking whether a wall route between the same two nodes already
existed. When one did and it was longer than 8.0, the search took the diagonal.

On the reference route:

```
chosen:    PATH-0129 → PATH-0126     11.25 m straight, charged 8.0 (flat BETA)
available: PATH-0129 → PATH-0128      5.65 m   same face
           PATH-0128 → PATH-0127      3.97 m   corner bridge
           PATH-0127 → PATH-0126      6.00 m   same face
                                    ─────────
                                     15.62 m   all at wall_d ≈ 1.8–2.0 m
```

`8.0 < 15.62`, so the diagonal won. The crossing was not bridging a gap — it was
short-circuiting a corner that was already fully connected. Across L4, **38 of
186 crossing edges did this**; the worst short-circuited a 62.57 m wall route.

### The fix

`BETA` could not simply be raised or removed: 140 of those 186 edges bridge
faces with **no** wall route at all, and are what keeps the graph connected. The
fix has to separate *"no wall route exists"* from *"a wall route exists but is
longer"*:

```python
if _path_only_distance(
    pn_graph, a.node_id, b.node_id,
    limit=CROSSING_SHORTCUT_FACTOR * BETA,
) is not None:
    continue        # the wall already connects these — don't offer the diagonal
```

`_path_only_distance()` is a new bounded Dijkstra over `kind='path'` adjacency
only — same-face runs and corner bridges, never crossings. It abandons any
frontier beyond `limit`, so the per-candidate cost stays low even though it runs
once per candidate crossing. `CROSSING_SHORTCUT_FACTOR = 3.0` gives a 24 m
cutoff.

### Effect on L4 crossing edges — 186 → 152

| | before | after |
|---|---|---|
| bridge a disconnected pair | 140 | **140** (all retained) |
| short-circuit a connected route | 38 | **4** |
| connected route cheaper than `BETA` | 8 | 8 |

Connectivity is preserved — 146/146 path nodes remain reachable. The 4 survivors
have wall routes beyond the 24 m cutoff and are kept deliberately: past that
distance the crossing is a genuine shortcut rather than a short-circuit.

---

## 3. Test suite repair

`tests/test_wp6_engine.py` was **23 failed, 162 passed** before this round — all
23 calling the retired positional signature `find_path(x, y, dest_id)` against
the current `find_path(start_node_id, destination_node_id)` (`engine.py:98`).

Most were a mechanical swap. Two needed more:

- `test_distance_biased_weights_select_route_x` was using the wrong graph's
  start id (`SKE-START` instead of `DB-START`);
- `test_first_step_is_user_segment_when_far` relied on an off-node user
  position, which the node-id API cannot express. It is replaced by
  `test_first_step_is_user_segment_when_starting_on_a_path_node`, which reaches
  the same `USER` virtual-first-step behaviour through the path-node entry
  point — where `user_xy` legitimately differs from the resolved graph node.

Six new regression tests cover the two defects, which previously passed
silently:

| Class | Tests | Covers |
|---|---|---|
| `TestCrossingShortCircuit` | 3 | suppression fires on a connected corner, does **not** fire on a genuinely disconnected pair, and the bounded search respects its limit |
| `TestSyntheticEdgeShoreHonesty` | 3 | wall-hugging hops report `True`, an open-floor hop reports `False`, and `shore_fraction` is recorded on every synthetic edge |

**Suite is now 191 passed, 0 failed** (verified 2026-07-26).

---

## 4. Results

### Reference route — ODEL → Popeyes, L4

| | before | after |
|---|---|---|
| hops | 21 | 23 |
| total walked | 130.91 m | 135.28 m |
| shore-linable hops | 20 / 21 | **23 / 23** |
| unreferenced distance | 11.25 m | **0.00 m** |
| longest unreferenced run | 11.25 m | **0.00 m** |
| detour ratio | — | 1.033 |

The tail now runs
`PATH-0131 → PATH-0129 → PATH-0128 → PATH-0127 → PATH-0126 → door → Popeyes`:
the user follows the west wall down to the corner, turns along the south wall,
and reaches the doorway. No open-floor traverse at any point.

### Floor-wide — all 90 routable L4 centroid pairs

| | before | after |
|---|---|---|
| routes computed | 90 | 90 (0 exceptions) |
| total walked | 10 459.5 m | 10 572.5 m |
| unreferenced | 1 911.5 m (18.28 %) | **1 599.9 m (15.13 %)** |

**311.6 m of unreferenced walking removed for 113 m of extra distance** — routes
are 1.1 % longer on average. That is the trade this module exists to make.

### `CROSSING_SHORTCUT_FACTOR` sweep

Swept over all 90 L4 pairs. `∞` means "suppress whenever any wall route exists".

| factor | cutoff | crossings | suppressed | walked | unreferenced |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0 m | 186 | 0 | 10 457.1 m | 18.03 % |
| 1.0 | 8 m | 178 | 8 | 10 457.1 m | 18.03 % |
| **2.0** | **16 m** | **153** | **33** | **10 572.5 m** | **15.13 %** |
| 3.0 | 24 m | 152 | 34 | 10 572.5 m | 15.13 % |
| 5.0 | 40 m | 147 | 39 | 10 572.5 m | 15.13 % |
| 8.0 | 64 m | 140 | 46 | 10 555.5 m | 15.16 % |
| ∞ | ∞ | 140 | 46 | 10 555.5 m | 15.16 % |

Three things worth reporting:

1. **The benefit is a step, not a gradient.** Everything useful happens between
   1.0 and 2.0. Above 2.0 the unreferenced fraction is flat to within 0.03 pp
   even as suppression keeps climbing — the extra suppressed edges were on
   routes nobody takes.
2. **3.0 is inside the plateau but is not optimal.** 2.0 achieves the identical
   15.13 % with a 16 m cutoff. 3.0 is retained for headroom above the reference
   route's 15.62 m wall chain; at 2.0 the margin is 0.4 m and would flip on
   small geometry changes.
3. **Unbounded suppression is slightly worse** (15.16 % vs 15.13 %). Removing
   every crossing with any wall alternative forces some routes onto longer
   detours that themselves contain unreferenced stretches. This is the empirical
   case for having a cutoff at all, rather than the simpler rule "never cross
   when the wall connects".

---

## 5. Files changed

| File | Change |
|---|---|
| `pathfinding/path_node_adjuster.py` | `_shore_fraction()`, measured `_add_synthetic_edge()`, `_path_only_distance()`, suppression in `_build_crossing_edges()`, five new constants (+117 lines) |
| `tests/test_path_node_adjuster.py` | `TestCrossingShortCircuit`, `TestSyntheticEdgeShoreHonesty` (+142 lines) |
| `tests/test_wp6_engine.py` | 23 stale `find_path()` calls migrated to the node-id API |

Supporting artefacts (untracked): `LAST_MILE_FIX.md`, `README.md`, and the
regenerated route `data/outputs/itfac_mall_floors_3-4_combined_ODEL-POPEYES_Route_v3.json`.

---

## 6. Verifying the result visually

The v3 route renders with:

```bash
cd minus197_path_finding
python visualizer/path_node_visualizer.py itfac_mall_floors_3-4_combined_L4 \
    --path data/outputs/itfac_mall_floors_3-4_combined_ODEL-POPEYES_Route_v3.json \
    --save data/outputs/odel-popeyes_route_v3.png
```

Note the **`_L4` suffix on the prefix**. The bare
`itfac_mall_floors_3-4_combined` only auto-resolves per-floor artefacts when the
route contains a `take_elevator` action — `visualize()` in
`visualizer/path_node_visualizer.py` only calls `_floor_prefix()` for legs that
come back with a floor label. ODEL and Popeyes are both on L4, so this route has
no elevator step, takes the `floor_label is None` branch, and uses the prefix
verbatim — looking for a `itfac_mall_floors_3-4_combined_sfm.json` that does not
exist. Only `_L3_` and `_L4_` artefacts are emitted by the mapping module.

The render shows 23 steps, 20 of them path-node hops, against the 146-node
background layer: a long wall-hugging run west along the north corridor, a turn
south at x ≈ 44, and a descent along the corridor's east wall into Popeyes.

---

## 7. What is still outstanding

Carried over from `LAST_MILE_FIX.md` §6 — neither is a routing defect:

- **Feedback anchoring and arrival phrasing** (§1 there). `to_feedback_json()`
  anchors every action at `from_node` except the terminal `stop`, which is
  anchored at `to_node` — so the final door row is consumed as the stop's anchor
  and **the door never appears in the output**, though the route walks to it.
  This is why routes must not be diagnosed from the emitted JSON. Fixing it
  changes the output contract (one extra action per route), so it needs
  coordinating with the feedback module rather than shipping silently.
- **Resolve `PATH_GAP_M`** (§4 there). The constant is 2.0 m; the
  `map_extraction/path_nodes.py` docstring says 0.3 m in three places. Emitted
  L4 data confirms 2.0. 2.0 m is a wide offset for cane trailing. `SHORE_MAX_D`
  now derives from this constant, so the decision propagates automatically.

### Caveats on the numbers above

- The 4 surviving short-circuits are a **choice, not a residue**. A claim that
  "crossing short-circuits were eliminated" would be wrong; the claim is
  "reduced from 38 to 4, with the remainder retained deliberately above a 24 m
  threshold".
- **All floor-wide figures are L4 only.** L3 is unmeasured, and so is the factor
  sweep.
- **Multi-floor routes are unexamined.** `multi_floor.py` stitches legs at the
  elevator and may reintroduce centroid-terminated segments at each boundary.
- The sweep's `walked` column is **not monotone** in the factor, so total
  distance is not a clean function of suppression strength — report the
  unreferenced fraction as the headline metric, not route length.
- The 140/38/8 classification depends on the current path-node layer. Re-run it
  after any change to `PATH_GAP_M`, `SPACING_MAX`, or `MAX_CHAIN_GAP`.

---

*Module owner: Minus197 mapping and pathfinding*
*Group 38 — University of Moratuwa*
