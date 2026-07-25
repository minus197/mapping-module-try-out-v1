# Last-Mile Approach Fix — Crossing-Edge Short-Circuit

**Part of:** Minus197 — Indoor Navigation System for Visually Impaired Users
**Module:** `minus197_path_finding`
**Reference route:** `ZONE-2aQOv7Awv2XAeFI_6R_KbE` → `ZONE-233bRpVtr7PADnPMIOrRa2`
(ODEL → Popeyes) on `itfac_mall_floors_3-4_combined_L4`
**Status:** items 1–3 **implemented** 2026-07-26 (see §9); items 4–6 outstanding

> **Revision history.** The first version of this document diagnosed a
> *missing-portal* defect: it claimed the router terminated on a bare zone
> centroid and never traversed the Popeyes door, producing a 19.14 m two-legged
> diagonal. That diagnosis was written from the emitted feedback JSON and **was
> wrong**. The door is traversed; the output format hides it. The superseded
> analysis is preserved in [§A](#appendix-a--superseded-diagnosis-retained-for-audit)
> for audit. Everything in §0–§8 below reflects the re-verified position.

---

## 0. Summary

Three real defects, none of which is the one originally diagnosed.

| # | Defect | Severity | Kind |
|---|---|---|---|
| 1 | `to_feedback_json()` drops the final door from the output | high, user-visible | output format |
| 2 | Flat-`BETA` crossing edges short-circuit connected wall routes | high | routing |
| 3 | Synthetic edges falsely claim `shore_linable=True` | high | measurement integrity |

Defect 2 is the actual cause of the unreferenced diagonal. Defect 3 must be
fixed *first*, because every metric used to validate the others is currently
read off falsified edge data.

The proposed `portal_resolver.py` from the earlier revision is **not needed**.
See §5.

### Verification method

All figures below were reproduced by instrumenting `_dijkstra_mixed()` to dump
the adjuster's internal winning chain, then re-deriving each measurement from
the emitted graph and path-node artefacts. **Do not diagnose this pipeline from
the feedback JSON** — defect 1 makes that output structurally misleading.

---

## 1. Defect 1 — `to_feedback_json()` hides the final door

**Confirmed** · `pathfinding/engine.py:282`

```python
anchor = step.to_node if is_last else step.from_node
```

Every action is anchored at the node the user is *standing on* (`from_node`),
except the terminal `stop`, which is anchored at the node *reached*
(`to_node`). On a route ending `… → PATH-0126 → DOOR → CENTROID`:

- the `PATH-0126 → DOOR` hop is emitted under the label `PATH-0126`;
- the door's own row is consumed as the anchor of the terminal `stop`.

**The door never appears in the output**, though the route walks to it.

### The actual chain tail

Dumped from `adjust_with_path_nodes()`:

```
PATH-0131 → PATH-0129                       5.65 m
PATH-0129 → PATH-0126                      11.25 m   ← the real defect (§2)
PATH-0126 → FEAT-0NrdkfQ69FhA8iVi0WZ06i     3.17 m   ← door, traversed
FEAT-0Nrd… → ZONE-233bRpVtr7PADnPMIOrRa2    5.80 m
```

`'FEAT-0NrdkfQ69FhA8iVi0WZ06i' in chain` → `True`.

**Implication.** `_mandatory_nodes()` and `_reconstruct_route_nodes()` both work
as intended. There is no missing-portal defect and no 19.14 m two-legged
diagonal. The real tail is 14.42 m and terminates at a physical doorway.

### Fix

Emit the door as its own row. The cleanest form is to stop special-casing the
last step's anchor and instead append a distinct terminal action:

- every movement action stays anchored at `from_node` — including the
  `PATH-0126 → DOOR` hop, which becomes a normal 3 m approach step;
- the terminal `stop` is appended as an extra action anchored at the
  destination, rather than consuming a movement row.

Arrival phrasing should then name the portal and a side:

| Step | Emit |
|---|---|
| approach | "Follow the wall on your right for 6 metres" |
| arrival | "The entrance to Popeyes is on your right in 3 metres" |

Side identification comes from the sign of the cross product between the final
approach bearing and the vector to the door. Unit-test both signs.

The same treatment fixes the start: `ZONE-2aQOv7…` → `FEAT-26jjhu0…` is 6.09 m
of diagonal *inside the ODEL shop*, currently instructed as a walking leg. It
should read "Leave ODEL through the entrance."

---

## 2. Defect 2 — flat-`BETA` crossing edges short-circuit connected routes

**Confirmed** · `path_node_adjuster.py:393` (`_build_crossing_edges`), `BETA` at `:97`

`_build_crossing_edges()` prices every cross-face hop at a flat `BETA = 8.0`
**without checking whether a wall-hugging route between the same two nodes
already exists.** When one does and it is longer than 8.0, the search takes the
open-floor diagonal.

### On the reference route

```
chosen:    PATH-0129 → PATH-0126     11.25 m straight, charged 8.0 (flat BETA)
           midpoint 4.92 m from the nearest wall — genuinely unreferenced

available: PATH-0129 → PATH-0128      5.65 m   same face
           PATH-0128 → PATH-0127      3.97 m   corner bridge
           PATH-0127 → PATH-0126      6.00 m   same face
                                    ─────────
                                     15.62 m   all at wall_d ≈ 1.8–2.0 m
```

All three links verified present in `_GlobalPathGraph.adjacency` with
`kind='path'`. Path-only shortest distance `PATH-0129 → PATH-0126` = 15.62 m.

8.0 < 15.62, so the diagonal wins. **The crossing edge is not bridging a gap —
it short-circuits a corner that is already fully connected.**

Note this is a `BETA` under-pricing defect specifically, not a plain-distance
one: under raw distance the diagonal (11.25 m) also beats the chain (15.62 m).
Raising the cost of unreferenced traverses is the fix; the flat charge is what
makes it structurally impossible.

### Scale across L4

Of 186 crossing edges generated (classified by whether a `kind='path'`-only
route exists between the endpoints, and how long it is):

| | count | meaning |
|---|---|---|
| genuinely bridge a disconnected pair | 140 | `BETA` doing its intended job |
| **short-circuit an already-connected route** | **38** | **the defect** |
| connected route already cheaper than `BETA` | 8 | harmless |

The reference route's instance ranks **15th of 38** by distance saved (7.62 m).
The worst is `PATH-0120 → PATH-0135`, short-circuiting a **62.57 m** wall route
for a flat 8.0 — followed by four more in the 40–52 m range.

### Fix

**Do not simply raise or delete `BETA`.** 140 edges depend on it to bridge faces
with no wall route at all; removing it disconnects the graph, and raising it
uniformly makes those legitimate bridges unaffordable.

The fix must distinguish *"no wall route exists"* from *"a wall route exists but
is longer"*. Suppress or surcharge the crossing edge when a path-only route
between its endpoints already exists within some multiple of `BETA`:

**Implemented** as `CROSSING_SHORTCUT_FACTOR = 3.0` plus `_path_only_distance()`
in `path_node_adjuster.py`; the check sits in `_build_crossing_edges()` after
the existing geometric tests:

```python
if _path_only_distance(
    pn_graph, a.node_id, b.node_id,
    limit=CROSSING_SHORTCUT_FACTOR * BETA,
) is not None:
    continue        # the wall already connects these — don't offer the diagonal
edges.append((a.node_id, b.node_id, BETA, "crossing"))
```

`_path_only_distance()` is a bounded Dijkstra over `kind='path'` adjacency only
(same-face runs and corner bridges, never crossings), abandoning any frontier
beyond `limit` so the per-candidate cost stays low.

**Measured effect on L4** — crossing edges 186 → 152:

| | before | after |
|---|---|---|
| bridge a disconnected pair | 140 | **140** (all retained) |
| short-circuit a connected route | 38 | **4** |
| connected route cheaper than `BETA` | 8 | 8 |

Connectivity is preserved: 146/146 path nodes remain reachable. The 4 survivors
have wall routes beyond the 24 m cutoff and are kept deliberately — past that
distance the crossing is a genuine shortcut, not a short-circuit.

**Acceptance — met.** The winning chain now runs
`PATH-0129 → PATH-0128 → PATH-0127 → PATH-0126`, and every hop reports
`shore_linable=True`.

A softer alternative remains untested: instead of suppressing, charge
`min(BETA, wall_route_distance)` so the diagonal can never undercut the wall
while still existing as a fallback. Worth trying if a floor appears where
suppression removes a bridge that is genuinely needed.

---

## 3. Defect 3 — synthetic edges falsely claim `shore_linable=True`

**Confirmed** · `path_node_adjuster.py:733-749`

`_add_synthetic_edge()` stamps `shore_linable=True` and `safety_score=1.0` on
**every** synthetic hop unconditionally, including open-floor diagonals.

Measured on the reference route: **21 of 21 hops report `shore_linable=True`** —
including the 11.25 m diagonal sitting 4.92 m from any wall.

Any shoreline or safety metric computed from adjusted-route edges is therefore
falsified. **Fix this before recording any baseline**, or before/after
comparison means nothing.

### Fix — implemented

`_add_synthetic_edge()` now takes `pn_graph` (the call site passes it) and
measures shore-linability by **sampling along the hop**:

```python
frac = _shore_fraction(pn_graph, src.position, tgt.position)
edge = NavigationEdge(
    ...,
    shore_linable=frac >= SHORE_FRACTION,
    tags={"shore_fraction": str(round(frac, 3))},
)
```

Three constants were added rather than reusing the document's `SHORE_MAX_D`
placeholder, which did not exist in this module:

```python
PATH_GAP_M     = 2.0                 # mirrors map_extraction/path_nodes.py
SHORE_MAX_D    = 1.25 * PATH_GAP_M   # 2.5 m
SHORE_FRACTION = 0.40                # mirrors graph_builder.SHORE_FRACTION
SHORE_SAMPLES  = 9
```

`SHORE_MAX_D` is expressed as a **multiple of `PATH_GAP_M`**, not a literal, so
it tracks §4 if that constant is retuned. Upstream `graph_builder.SHORE_BUFFER`
(0.80 m) could not be reused directly: path nodes sit 2.0 m off their wall, so a
0.80 m threshold would report *every* hop as unshored — swapping a uniformly
optimistic metric for a uniformly pessimistic one.

Sampling rather than a midpoint test matters: a hop starting hard against a wall
and ending in open floor has a midpoint that can fall either side of the
threshold, and one sample cannot separate those cases. `shore_fraction` is also
recorded in `edge.tags`, mirroring what `graph_builder` does for real edges, so
the decision is auditable rather than just a boolean.

Anchor hops to a door or centroid are measured on the same basis — reaching a
doorway along a wall is genuinely shored; cutting to a centroid is not.

### The honest baseline — measured

With honest edges in place but **before** the §2 fix, the reference route
measures:

```
hops                   21
total walked      130.91 m
shore_linable      20 / 21
unreferenced       11.25 m   (the single diagonal — defect 2)
longest unref run  11.25 m
```

The diagonal reports `shore_fraction = 0.22`; every wall run reports 1.00. The
two anchor hops (shop centroid → door) report 0.44 — above the 0.40 threshold,
so they count as shored, which is correct: they run alongside the shopfront.

Not 19.14 m across two legs. This is a **better** baseline than previously
assumed, which shrinks the improvement headroom any evaluation can claim. State
the real figure; a 130.91 m route with one 11.25 m unreferenced hop is a
defensible starting point, and overstating the before-picture is the kind of
thing an examiner tests.

---

## 4. Defect 4 — `PATH_GAP_M` docstring is stale

**Confirmed** · `../minus197_mapping/map_extraction/path_nodes.py:92`

```python
PATH_GAP_M = 2.0   # m — gap between a wall and its path node
```

The module docstring states 0.3 m in three places (lines 13, 20, 64 in the
current file). Emitted L4 data confirms 2.0 m — nodes measure 1.8–2.0 m off
their wall.

2.0 m is a wide offset for cane trailing; a user tracking a wall is typically
much closer. Whether 2.0 m is intentional or a regression from 0.3 m is **an
open question worth resolving**, since it sets how far every route sits from its
reference surface — and it interacts directly with the `wall_d ≤ SHORE_MAX_D`
test in §3.

Resolve the value first, then correct either the constant or the docstring.

---

## 5. Assessed and rejected — portal machinery

**Not a defect.** The earlier revision proposed `portal_resolver.py` plus a
two-segment Dijkstra to force door termination. Measurement says this solves
nothing currently broken:

- 10 of the 11 L4 zone centroids already have **exactly one** door edge;
- the router already traverses that door (§1);
- the 11th centroid with no door edge is `ZONE-1i$2JbPrrAMOUePJPoCKpR` — the
  4240 m² **corridor** zone itself, not a shop. It is a routing origin/waypoint,
  not a destination needing a portal.

It would convert an accident into a guarantee, which has some value if a floor
later appears where a shop centroid has multiple doors. It is not the fix for
the observed geometry, and should not be built now.

**Rule 2 of the old §3.1 could never have worked.** It selected doors "whose
`zone_id` matches the centroid's zone". All 10 L4 door nodes carry
`zone_id = "1i$2JbPrrAMOUePJPoCKpR"` — the corridor zone — because they are
*hosted* in the corridor, not in the shop they serve. The only centroid sharing
that id is the corridor's own. So rule 2 matches **all 10 doors to the corridor
centroid and no shop centroid ever**: it is not merely dead code, it is
actively wrong where it fires. If portal resolution is ever built, drop rule 2
or re-source it from a door's `connects_to` relationship, which the graph does
not currently emit.

---

## 6. Build order

1. ~~**Honest synthetic edges** (§3)~~ — **done.** The 11.25 m diagonal reports
   `shore_linable=False` (`shore_fraction` 0.22).
2. ~~**Re-measure the baseline**~~ — **done.** See §3; 130.91 m walked, 11.25 m
   unreferenced.
3. ~~**Crossing-edge short-circuit suppression** (§2)~~ — **done.** The winning
   chain contains `PATH-0128` and `PATH-0127`; no unreferenced hop remains.
4. **Feedback anchoring and arrival phrasing** (§1) — *outstanding.* Surfaces
   the door already being walked to. Largely a `to_feedback_json()` fix, not a
   routing one. Changes the output contract (§8) — coordinate with the feedback
   module before shipping.
5. **Resolve `PATH_GAP_M`** (§4) — *outstanding.* Decide 0.3 vs 2.0, then
   correct the constant or the docstring. Note `SHORE_MAX_D` is now derived from
   it, so this decision propagates automatically.
6. Portal machinery (§5) — only if a floor appears where a *shop* centroid has
   multiple or zero door edges.

### Prerequisite — the test suite (resolved)

Was **23 failed, 162 passed**, all in `tests/test_wp6_engine.py`, calling the
retired positional signature `find_path(x, y, dest_id)` against the current
`find_path(start_node_id, destination_node_id)` (`engine.py:98`).

Fixed by passing node ids. Two tests needed more than a mechanical swap:

- `test_distance_biased_weights_select_route_x` used the wrong graph's start id
  (`SKE-START` instead of `DB-START`);
- `test_first_step_is_user_segment_when_far` relied on an off-node user
  position, which the node-id API cannot express. It is replaced by
  `test_first_step_is_user_segment_when_starting_on_a_path_node`, which reaches
  the same `USER` virtual-first-step behaviour through the path-node entry
  point — where `user_xy` legitimately differs from the resolved graph node.

Suite is now **191 passed, 0 failed**, including six new regression tests:
`TestCrossingShortCircuit` (3) and `TestSyntheticEdgeShoreHonesty` (3), covering
defects 2 and 3 which previously passed silently.

---

## 7. Acceptance test

Reference route:

- [x] the 11.25 m `PATH-0129 → PATH-0126` hop reports `shore_linable=False`
      before the §2 fix, and is absent from the chain after it
- [x] the winning chain contains `PATH-0128` and `PATH-0127`
- [x] no hop reports `shore_linable=False` (unreferenced distance 0.00 m)
- [x] total walked distance stays within 1.4 × the 130.91 m baseline
      — 135.28 m, a detour ratio of **1.033**
- [x] no exceptions across all 90 routable L4 centroid pairs
- [ ] `FEAT-0NrdkfQ69FhA8iVi0WZ06i` appears as its own row in the emitted
      feedback JSON *(item 4, outstanding)*
- [ ] the arrival action names a side: "entrance to Popeyes is on your **right**"
      *(item 4)*
- [ ] the ODEL exit leg is phrased as a doorway exit, not a 6 m diagonal
      *(item 4)*

Regression: a destination zone with no door edge must still route, terminating
at the centroid with one logged warning and no exception.

> The old criterion *"the winning chain contains `FEAT-0Nrd…`"* is **not**
> discriminating — it already passes at baseline (§1).

---

## 8. Known limitations

- ~~**`CROSSING_SHORTCUT_FACTOR` is unmeasured.**~~ Swept on L4 — see §9. The
  metric plateaus above factor 2.0, so 3.0 sits on a flat region rather than at
  a tuned optimum. Unmeasured on L3.
- **The 4 m unreferenced threshold is a convention**, chosen against
  `PATH_GAP_M = 2.0` so a hop hugging either of two opposite walls stays under
  it. It moves if §4 resolves the gap to 0.3 m.
- **The 140/38/8 classification depends on the current path-node layer.**
  Re-run it after any change to `PATH_GAP_M`, `SPACING_MAX`, or `MAX_CHAIN_GAP`;
  the counts are not stable constants.
- **Multi-floor routes are unexamined here.** All measurements are single-floor
  L4. `multi_floor.py` stitches legs at the elevator and may reintroduce
  centroid-terminated segments at each floor boundary.
- **Defect 1's fix changes the output contract.** Any consumer of
  `to_feedback_json()` — including the feedback module — will see one extra
  action per route and a door row that was never there before. Coordinate the
  change rather than shipping it silently.

---

## 9. Implementation results (2026-07-26)

Items 1–3 are built. Item 4 (output contract) and item 5 (`PATH_GAP_M`) remain.

### Files changed

| File | Change |
|---|---|
| `pathfinding/path_node_adjuster.py` | `_shore_fraction()`, honest `_add_synthetic_edge()`, `_path_only_distance()`, suppression in `_build_crossing_edges()`, five new constants |
| `tests/test_path_node_adjuster.py` | `TestCrossingShortCircuit`, `TestSyntheticEdgeShoreHonesty` (6 tests) |
| `tests/test_wp6_engine.py` | 23 stale `find_path()` calls migrated to the node-id API |

### Reference route — before vs after

| | before | after |
|---|---|---|
| hops | 21 | 23 |
| total walked | 130.91 m | 135.28 m |
| shore-linable hops | 20 / 21 | **23 / 23** |
| unreferenced distance | 11.25 m | **0.00 m** |
| longest unreferenced run | 11.25 m | **0.00 m** |
| detour ratio | — | 1.033 |

The tail is now
`PATH-0131 → PATH-0129 → PATH-0128 → PATH-0127 → PATH-0126 → door → Popeyes`:
the user follows the west wall down to the corner, turns along the south wall,
and reaches the doorway — no open-floor traverse at any point.

### Floor-wide — all 90 routable L4 centroid pairs

| | before | after |
|---|---|---|
| routes computed | 90 | 90 (0 exceptions) |
| total walked | 10 459.5 m | 10 572.5 m |
| unreferenced | 1 911.5 m (18.28 %) | **1 599.9 m (15.13 %)** |

**311.6 m of unreferenced walking removed for 113 m of extra distance** — routes
are 1.1 % longer on average. This is the trade the module exists to make.

### Where the remaining 1 599.9 m sits

| source | hops | distance |
|---|---|---|
| path-node ↔ path-node | 106 | 1 343.6 m |
| anchor hops (door / centroid) | 18 | 256.8 m |

The path-node share is dominated by the 4 crossings whose wall route exceeds the
24 m cutoff, plus long same-face hops where the path-node layer has coverage
gaps. Neither is addressable by this fix: the first is a deliberate policy
choice, the second is a mapping-side density question tied to §4. **This is the
next thing worth measuring**, and it is a better-defined target than any weight
sweep.

### Factor sweep — `CROSSING_SHORTCUT_FACTOR`

Swept over all 90 L4 centroid pairs. `inf` means "suppress whenever any wall
route exists at all", i.e. an unbounded search:

| factor | cutoff | crossings | suppressed | walked | unreferenced |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0 m | 186 | 0 | 10 457.1 m | 18.03 % |
| 1.0 | 8 m | 178 | 8 | 10 457.1 m | 18.03 % |
| **2.0** | **16 m** | **153** | **33** | **10 572.5 m** | **15.13 %** |
| 3.0 | 24 m | 152 | 34 | 10 572.5 m | 15.13 % |
| 5.0 | 40 m | 147 | 39 | 10 572.5 m | 15.13 % |
| 8.0 | 64 m | 140 | 46 | 10 555.5 m | 15.16 % |
| 12.0 | 96 m | 140 | 46 | 10 555.5 m | 15.16 % |
| ∞ | ∞ | 140 | 46 | 10 555.5 m | 15.16 % |

Three things worth reporting:

1. **The benefit is a step, not a gradient.** Everything useful happens between
   factors 1.0 and 2.0. Above 2.0 the unreferenced fraction is flat to within
   0.03 pp, even though suppression keeps climbing (34 → 46 edges). The extra
   suppressed edges were on routes nobody takes.
2. **The chosen 3.0 is safely inside the plateau but is not special.** 2.0
   achieves the identical 15.13 % with a 16 m cutoff. 3.0 is retained because it
   has more headroom above the reference route's 15.62 m wall chain — at 2.0 the
   margin is only 0.4 m, which would flip on small geometry changes.
3. **Unbounded suppression is slightly *worse*** (15.16 % vs 15.13 %). Removing
   every crossing that has any wall alternative forces some routes onto longer
   detours that themselves contain unreferenced stretches. This is the empirical
   justification for having a cutoff at all, rather than the simpler rule
   "never cross when the wall connects".

The sweep is cheap to re-run (`CROSSING_SHORTCUT_FACTOR` is a module constant)
and should be repeated on L3 and after any change to §4.

### Caveats

- The 4 surviving short-circuits are a *choice*, not a residue. If the
  dissertation claims "crossing short-circuits eliminated", it is wrong — the
  claim is "reduced from 38 to 4, with the remainder retained deliberately
  above a 24 m threshold".
- Floor-wide figures cover L4 only. L3 is unmeasured.
- The sweep's `walked` column is not monotone in the factor, so total distance
  is not a clean function of suppression strength — another reason to report the
  unreferenced fraction as the headline metric rather than route length.

---

## Appendix A — superseded diagnosis (retained for audit)

The original §0–§8 claimed a *missing-portal* defect. Its central assertions and
what measurement showed:

| Original claim | Verdict |
|---|---|
| The route terminates on a bare zone centroid | **False.** The door is in the winning chain; `to_feedback_json()` hides it (§1). |
| "19.14 m two-legged unreferenced diagonal" | **False.** `PATH-0126 → GOAL 7.89 m` was a straight-line measurement of a segment never walked. Real tail: 14.42 m. |
| `_mandatory_nodes()` drops the door | **False.** It retains it; the door is reached as a `mandatory` hop. |
| Centroid snap-edge creates a bypass shortcut | **False.** No bypass occurs. |
| `_add_synthetic_edge()` falsifies `shore_linable` | **True.** Re-verified — now §3. |
| Root cause is termination/connectivity | **False.** Root cause is flat-`BETA` crossing pricing (§2). |
| Fix requires `portal_resolver.py` | **Rejected.** Not justified by data (§5). |

The methodological lesson is worth recording: the original analysis was derived
from `data/outputs/L4_ODEL-POPEYES_route.json`, and defect 1 makes that file an
unreliable witness to its own route. Diagnose from the adjuster's internal
chain.

---

*Module owner: Minus197 mapping and pathfinding*
*Group 38 — University of Moratuwa*
