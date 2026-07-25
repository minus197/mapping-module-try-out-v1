# Feedback-Contract Fix — `to_feedback_json()`

**Part of:** Minus197 — Indoor Navigation System for Visually Impaired Users
**Commit:** `a69b4bf` on `path-finding-v3`
**Date:** 2026-07-26
**Scope:** the pathfinding → feedback-module boundary only. **No routing logic
was changed.** The path a user walks is byte-identical to v3.
**Supersedes:** `V3_DEFECT_REGISTER.md` D-01, D-02, D-05, D-06, D-08, D-09

---

## 0. Summary

`to_feedback_json()` in `pathfinding/engine.py` was discarding most of the
`PathStep` contract on its way out of the module. Five of the ten defects in
`V3_DEFECT_REGISTER.md` trace to this one function.

The important finding is **where** the defects were. Every field the register
reported as missing was already being computed correctly upstream —
`shared/types.PathStep` declares them and `pathfinding/instructions.py`
populates them. They were being thrown away at emission. Nothing was missing
from the pipeline; one function at the boundary was lossy.

This has a direct consequence for the register: its headline defect (D-01,
"portal resolution missing, requires `portal_resolver.py`") is **false**. The
Popeyes door was in the winning chain the whole time and the serialiser was
hiding it. `LAST_MILE_FIX.md` Appendix A had already tested and rejected that
exact diagnosis. See §5.

---

## 1. Files changed

| File | Change | Lines |
|---|---|---|
| `pathfinding/engine.py` | `to_feedback_json()` rewritten; `_stop_action()`, `_approach_side()`, `_turn_band()` added; `import math` | +130 / −27 |
| `run_pathfinding.py` | console printer updated for the new keys | +6 / −2 |
| `tests/test_feedback_json.py` | **new** — 18 tests; the function had no coverage | +262 |
| `data/outputs/…_ODEL-POPEYES_Route_v4.json` | **new** — regenerated reference route | +397 |

`v3.json` was deliberately **not** overwritten, so the before/after comparison
survives for the dissertation.

---

## 2. The five defects, and what each fix was

### 2.1 The door was consumed as the stop's anchor — **the headline bug**

**Was** (`engine.py:282` *as of `a69b4bf^` — all line numbers in this section
refer to the pre-fix file*):

```python
anchor = step.to_node if is_last else step.from_node
```

Every action anchored at the node the user stands on (`from_node`), *except*
the terminal `stop`, which anchored at the node reached (`to_node`). On a route
ending `… → PATH-0126 → DOOR → CENTROID`:

- the `PATH-0126 → DOOR` hop was emitted under the label `PATH-0126`;
- the door's own row was consumed as the anchor of the terminal `stop`.

**The door never appeared in the output, though the route walked to it.**

**Now:** every `PathStep` emits exactly one movement action anchored at
`from_node`, and the `stop` is *appended* as an extra action rather than
substituting for a movement row.

**Why it matters beyond tidiness:** it made the output an unreliable witness to
its own route. Consecutive rows were not consecutive hops, so anyone
reconstructing the path from the file measured segments that were never walked.
This produced two successive wrong root-cause diagnoses (§5). The new
`to_node_id` field makes hops reconstructable so this cannot recur.

---

### 2.2 Turn bands collapsed to sign-of-angle

**Was** — `_turn_direction()` grepped the instruction string for "left"/"right"
and discarded everything else. `turn_phrase()` in `instructions.py` had
correctly resolved a band table (`≤20°` straight, `≤100°` bear, `≤170°` turn,
else around), and the serialiser reduced it to a sign.

A 29° bear and a 141° near-reversal both emitted `{"direction": "right"}`.

**Now:** movement actions carry `band` (`bear` | `turn` | `around`) alongside
`direction`.

**Why it matters:** for a user executing a turn by proprioception with no visual
confirmation, magnitude *is* the instruction. "Bear right" and "turn right" are
different physical acts. This is the single most user-facing item in the set.

---

### 2.3 Distances rounded to whole metres, per hop

**Was** — `dist_m = round(step.distance)` at emission, while
`PathStep.distance` was already a float.

Because each hop rounded independently and this route's hops sit just above
`x.5`, the error accumulated in one direction.

| | v3 (announced) | v4 (walked) | error |
|---|---|---|---|
| Whole ODEL→Popeyes route | 141 m | 135.31 m | **+5.69 m** |

The register estimated +3.15 m from one corridor leg; measured route-wide it is
**+5.69 m**.

**Now:** floats, rounded to 2 dp for readability.

**Why it matters:** for a user pacing distance, a cumulative 5.7 m overstatement
is a wrong-place arrival. It also made any metric computed from the emitted
field unreliable.

**Downstream note:** round at the **speech layer**, and round the *cumulative*
figure rather than each hop independently — otherwise this defect returns one
module over.

---

### 2.4 `bearing` and `instruction` were never emitted

**Was** — neither field appeared in the output, despite both being populated on
every `PathStep` by `instructions.py:160-167`.

The register concluded from this that "no human-readable string is produced
anywhere in the pipeline" and that instruction generation must live in the
feedback module. **That inference was wrong.** `instructions.py` produces full
instruction text — `"Turn right. Walk 3 m. You will reach …"` — and the
serialiser read it only to extract the first sentence for turn classification,
then dropped it.

**Now:** both emitted verbatim.

**Why it matters:** the feedback module had no heading for spatial-audio cueing
and was presumably re-synthesising text it was already being sent. This is the
likeliest contributor to the reported feedback-module variance — not a
shared-ownership problem, but a one-line omission at the boundary.

---

### 2.5 Edge attributes were absent

**Was** — `edge_id`, `shore_linable`, `safety_score`, `landmark_score` never
emitted, so **no `ROUTE_QUALITY_FACTORS.md` §8 metric could be computed from the
output**. This is what put §5 of the register ("cannot be assessed") in the
position it was in.

**Now:** all four emitted on every movement action.

**See §4 — these are now computable but not yet trustworthy.**

---

## 3. Arrival phrasing

The terminal `stop` now names a side:

```json
{
  "action": "stop",
  "landmark": "pop eyes",
  "side": "left",
  "instruction": "The entrance to pop eyes is on your left in 6 metres."
}
```

`side` comes from the sign of the cross product between the heading the user is
already travelling (the previous step) and the vector to the destination, with a
≈5° dead band collapsing to `"ahead"`. Both signs are unit-tested
(`test_side_flips_with_approach_geometry`), as `LAST_MILE_FIX.md` §1 required.

Returns `"ahead"` when there is no preceding step to establish a heading — a
single-step route has no approach direction to judge from.

---

## 4. Verification

```bash
cd minus197_path_finding
python run_pathfinding.py \
  --building-graph ../minus197_mapping/data/outputs/itfac_mall_floors_3-4_combined_full_building_graph.json \
  --from ZONE-2aQOv7Awv2XAeFI_6R_KbE --to ZONE-233bRpVtr7PADnPMIOrRa2 \
  --save-feedback data/outputs/itfac_mall_floors_3-4_combined_ODEL-POPEYES_Route_v4.json
```

**Tests:** 18 new, 209 existing — all pass.

**Output shape:**

| | v3 | v4 |
|---|---|---|
| actions | 23 | 24 |
| keys per action | `action`, `distance`, `node_id`, `position` | + `bearing`, `instruction`, `to_node_id`, `edge_id`, `shore_linable`, `safety_score`, `landmark_score`, `band`, `side` |

**The door is present**, with its real identity and the coordinates the register
cited as missing:

```json
{ "action": "turn", "direction": "right", "band": "turn", "distance": 3.17,
  "node_id": "PATH-0126", "to_node_id": "FEAT-0NrdkfQ69FhA8iVi0WZ06i",
  "bearing": 237.91, "shore_linable": true }
```

`FEAT-0NrdkfQ69FhA8iVi0WZ06i` at (55.32, −33.75) — matching the chain
`LAST_MILE_FIX.md` §1 dumped from `adjust_with_path_nodes()`.

### ⚠ Metrics are computable but not yet trustworthy

`shore_linable` is `true` on **every** edge in v4, so unreferenced distance
computes to **0.00 m**. That is almost certainly false — it cannot be reconciled
with D-10's 9.38 m open-floor corridor crossing.

Either the `_add_synthetic_edge()` honesty fix (`LAST_MILE_FIX.md` §3) does not
cover path-node edges (`PN-EDGE-*`), or it covers them and stamps `true`
regardless.

**Do not report "unreferenced distance: 0 m" anywhere.** The field is now
present and appears to be lying. Verifying it is the natural next task.

---

## 5. Corrections to `V3_DEFECT_REGISTER.md`

| Defect | Register's claim | Actual |
|---|---|---|
| **D-01, D-02, D-09** | Critical — portal resolution missing; build `pathfinding/portal_resolver.py` | **False.** Door is in the winning chain and always was. Already tested and rejected in `LAST_MILE_FIX.md` Appendix A |
| **D-05** | Six-band turn table not implemented | **Implemented** in `turn_phrase()`; discarded by `_turn_direction()`. Right symptom, wrong layer |
| **D-06** | `PathStep` lacks `bearing`/`instruction`; no instruction text produced anywhere | **Present** on `PathStep` and produced by `instructions.py`; dropped by the serialiser |
| **D-08** | Distances rounded upstream | **Floats** upstream; rounded at `engine.py:277`. Magnitude understated (+5.69 m, not +3.15 m) |
| **D-04** | Arrival/departure phrasing, "blocked on D-01" | **Real**, and was never blocked — D-01 does not exist. Arrival side shipped here; departure phrasing still open |
| **D-03, D-07, D-10** | compression, start-side snap, corridor crossing | **Unverified.** Need checking against `path_node_adjuster.py` |

### The methodological point

`LAST_MILE_FIX.md` Appendix A already recorded that the missing-portal
diagnosis was derived from the feedback JSON and was wrong. The v3 register
re-derived the same rejected conclusion from the same misleading file and
promoted it to "the headline fix."

**Both wrong diagnoses share one cause: they were written from the emitted JSON
rather than from the code or the adjuster's internal chain.** That file hid the
door and dropped the edge attributes, so any reconstruction from it measured
segments that were never walked.

Standing rule, now enforced by `to_node_id` and the edge attributes:

> Diagnose this pipeline from the source, or from an instrumented dump of
> `_dijkstra_mixed()`'s winning chain. Route JSONs generated before `a69b4bf`
> (`*_Route.json`, `*_Route_v3.json`) remain structurally misleading and should
> not be used as evidence for anything.

---

## 6. What this does and does not improve

**Does not:** change the route. Same nodes, same metres, same turns as v3. No
routing, cost, or adjuster code was touched.

**Does:** improve what the user is *told* — the entrance is announced instead of
a bare "stop"; turn magnitude survives to the user; distances stop overstating;
the feedback module receives headings and instruction text it was previously
denied.

**Enables:** the §8 route-quality metrics to be computed at all, which every
before/after claim in the dissertation depends on — subject to the
`shore_linable` caveat in §4.

For the write-up, the accurate framing is **"fixed the instruction contract at
the pathfinding/feedback boundary"**, not "fixed the last-mile routing." Whether
the routing needs fixing at all is now an open question — D-03, D-07 and D-10
are the remaining candidates and none has yet been verified against source.

---

## 7. Revised next steps

| # | Item | Why |
|---|---|---|
| 1 | Verify `shore_linable` on `PN-EDGE-*` edges | §4 — metrics are untrustworthy until this is settled |
| 2 | Re-measure the true baseline; correct `ROUTE_QUALITY_FACTORS.md` §8 | Depends on 1 |
| 3 | Verify D-07 (start-side snap) and D-10 (corridor crossing) against `path_node_adjuster.py` | The only remaining candidates for genuine *route* defects |
| 4 | `compress_steps()` — D-03 | Independent; 13 zero-turn hops emitted as separate steps is real cognitive load |
| 5 | Departure phrasing — D-04 | "Leave ODEL through the entrance"; unblocked, ships independently |

Items previously ranked 1 and 6 in the register (`PathStep` field emission and
portal resolution) are **closed and withdrawn** respectively.

---

*Module owner: Minus197 mapping and pathfinding*
*Group 38 — University of Moratuwa*
