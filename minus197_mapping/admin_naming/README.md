# Admin Shop-Naming UI

## What this does

When an architect creates an IFC building model, spaces (shops, food court units,
kiosks) are annotated with short internal codes — `Z1`, `Z2`, `Торговый зал` —
not the real tenant names that a mall user would recognise. The pathfinding
feedback module produces a stop instruction that names the destination:

```json
{ "action": "stop", "landmark": "Starbucks" }
```

That real name has to come from somewhere. This module provides the bridge: an
admin tool that reads the generated JSON outputs, presents the architect's
annotations with geometry context, and lets a mall administrator type in the
actual shop names. It then patches the outputs so every downstream consumer
(pathfinding engine, feedback module) uses the correct name automatically.

---

## Where it sits in the pipeline

```
IFC file
   │
   ▼
[IFCParser]               ParsedSpace.name = "Z1"
                          ParsedSpace.guid = "0BGjw438P6JAPe2g$sDgcT"
   │
   ▼
[SemanticFloorMapBuilder] {stem}_sfm.json      zones[].name = "Z1"
[GraphBuilder]            {stem}_graph.json    nodes[].label = "Z1"
[OccupancyGridExporter]   {stem}_occupancy.json
   │
   ▼ ◀── THIS MODULE ──▶
   │
[shop_name_ui.py]         reads  _sfm.json
                          writes {stem}_shop_names.json
   │
   ▼
[shop_name_patcher.py]    patches _sfm.json      zones[].admin_name = "Starbucks"
                          patches _graph.json     nodes[].tags.admin_label = "Starbucks"
                          patches _occupancy.json zones[].admin_name = "Starbucks"
   │
   ▼
[PathfindingEngine]       reads tags["admin_label"] first → uses "Starbucks"
[to_feedback_json()]      { "action": "stop", "landmark": "Starbucks" }
```

---

## Files

```
minus197_mapping/
└── admin_naming/
    ├── README.md               this file
    ├── __init__.py
    ├── shop_name_ui.py         interactive CLI — reads SFM, writes _shop_names.json
    └── shop_name_patcher.py    patches the three JSON outputs in-place
```

---

## How to use

### Step 1 — Run the mapping pipeline first

```bash
cd minus197_mapping
python main.py --ifc data/ifc_files/mall_L1.ifc --floor L1
```

This produces:

```
data/outputs/
├── mall_L1_sfm.json
├── mall_L1_graph.json
└── mall_L1_occupancy.json
```

### Step 2 — Launch the naming UI

```bash
python main.py --ifc data/ifc_files/mall_L1.ifc --floor L1 --admin-name
```

Or run it standalone against already-generated outputs:

```bash
python -m admin_naming.shop_name_ui \
    --sfm data/outputs/mall_L1_sfm.json \
    --output data/outputs/
```

### Step 3 — The UI session

The UI lists only nameable zones — shops, food court units, and spaces the
classifier could not categorise (category `unknown`). Corridors, restrooms,
storage rooms, and offices are filtered out automatically.

```
──────────────────────────────────────────────────────────────
 Shop Naming — mall_L1  (floor L1)
 12 zones total · 8 need names · press Enter to skip a zone
──────────────────────────────────────────────────────────────
 #  │ IFC name (architect) │ Category   │ Area   │ Status
────┼──────────────────────┼────────────┼────────┼──────────
  1 │ Z1                   │ shop       │  45 m² │ (empty)
  2 │ Z2                   │ shop       │  62 m² │ (empty)
  3 │ Торговый зал         │ unknown    │  88 m² │ (empty)
  4 │ F1                   │ food_court │  34 m² │ (empty)
──────────────────────────────────────────────────────────────

[1/4] Z1 (shop, 45 m²)
  Enter shop name [skip=Enter]: Starbucks

[2/4] Z2 (shop, 62 m²)
  Enter shop name [skip=Enter]: Zara

[3/4] Торговый зал (unknown, 88 m²)
  Enter shop name [skip=Enter]: Food Court

[4/4] F1 (food_court, 34 m²)
  Enter shop name [skip=Enter]:

──────────────────────────────────────────────────────────────
 3 names entered · 1 skipped
 Saved → data/outputs/mall_L1_shop_names.json
 Run patcher? [Y/n]: Y
 Patched _sfm.json, _graph.json, _occupancy.json
──────────────────────────────────────────────────────────────
```

### Step 4 — Re-running to update names

If `{stem}_shop_names.json` already exists, the UI pre-fills every prompt with
the existing name. Press Enter to keep it, or type a new name to replace it.

```
[1/4] Z1 (shop, 45 m²)
  Enter shop name [current: Starbucks]:
```

### Step 5 — Re-patching without the UI

If you edit `_shop_names.json` by hand, re-apply it without going through the
UI:

```bash
python -m admin_naming.shop_name_patcher \
    --names data/outputs/mall_L1_shop_names.json \
    --output data/outputs/
```

---

## The shop names file

The UI writes a single file per floor:

```json
{
  "source_stem": "mall_L1",
  "floor_label": "L1",
  "generated_at": "2026-06-26T10:30:00",
  "mappings": [
    {
      "zone_id":    "0BGjw438P6JAPe2g$sDgcT",
      "ifc_name":   "Z1",
      "admin_name": "Starbucks"
    },
    {
      "zone_id":    "2lyI2yiub9QBhgywjO7xcX",
      "ifc_name":   "Z2",
      "admin_name": "Zara"
    }
  ]
}
```

`zone_id` is the IFC **GlobalId** (the 22-character GUID on every `IfcSpace`).
This is the globally unique, stable identifier assigned by the BIM tool. The
mapping survives IFC re-exports as long as the architect did not recreate the
space entity from scratch. `ifc_name` is stored for human readability only —
the patcher always keys on `zone_id`.

Zones that were skipped (no name entered) are not written to this file. The
patcher leaves those zones unchanged.

---

## What gets patched

### `{stem}_sfm.json`

Each zone that has a mapping gains an `admin_name` field:

```json
{
  "zone_id":    "0BGjw438P6JAPe2g$sDgcT",
  "ifc_name":   "Z1",
  "name":       "Z1",
  "long_name":  "Z1",
  "admin_name": "Starbucks",
  ...
}
```

### `{stem}_graph.json`

Zone centroid nodes (node type `zone_centroid`) whose `zone_id` matches get
two tag fields updated:

```json
{
  "node_id":   "ZONE-0BGjw438P6JAPe2g$sDgcT",
  "label":     "Starbucks",
  "node_type": "zone_centroid",
  "tags": {
    "admin_label": "Starbucks",
    "admin_name":  "Starbucks",
    ...
  }
}
```

`label` is updated so logs and debug output show the real name.
`tags["admin_label"]` is the key the pathfinding engine already reads when
building the `"stop"` feedback action.

### `{stem}_occupancy.json`

Any zone metadata block in the occupancy grid that carries a `zone_id` gains
`admin_name` in the same way as the SFM.

---

## How it connects to the feedback module

The feedback module receives `PathResult.steps` from the pathfinding engine.
For the final step, `to_feedback_json()` in `engine.py` resolves the landmark
name with this priority:

```python
landmark = (
    step.to_node.tags.get("admin_label", "").strip()
    or step.to_node.label
)
```

Because the patcher writes to `tags["admin_label"]`, the output becomes:

```json
{ "action": "stop", "landmark": "Starbucks" }
```

No changes are needed in the pathfinding engine or the feedback module.

---

## Design notes

**Why GlobalId and not the STEP instance number?**
The STEP `#9856` instance number in an IFC file is local to that specific file
export. If the architect re-exports the file, instance numbers can shift.
GlobalId is mandated by the IFC standard to be globally unique and persistent
across re-exports for the same logical entity.

**Why a separate `_shop_names.json` file?**
It keeps the admin data separate from the generated data. The three output JSON
files can always be regenerated from the IFC file. The shop names file is the
only file the admin owns — it does not get overwritten by a pipeline re-run.

**Why does the UI filter out corridors and restrooms?**
An admin naming corridors as `"Main Corridor"` adds no useful information to the
feedback module — users are never told to stop at a corridor. The filter keeps
the session short and focused.

**Why does the patcher write both `admin_label` and `admin_name`?**
`admin_label` is the key the existing pathfinding engine already reads (from the
inter-floor linker design). `admin_name` is the field name used in the SFM and
occupancy grid for semantic clarity. Both carry the same value.

---

## Implementation checklist

The following work is required before this module is functional:

- [ ] Add `admin_name: str = ""` to `Zone` dataclass in `semantic_floor_map.py`
      and include it in `Zone.to_dict()`
- [ ] Add `"admin_name": ""` to zone centroid node tags in `graph_builder.py`
      `_place_zone_centroid_nodes()`
- [ ] Implement `admin_naming/shop_name_ui.py`
- [ ] Implement `admin_naming/shop_name_patcher.py`
- [ ] Add `--admin-name` flag to `main.py`; call UI then patcher after `save()`
- [ ] Add `python -m admin_naming.shop_name_patcher` standalone entry point
- [ ] Update implementation status table in the top-level `README.md`

---

*Minus197 — Group 38 — University of Moratuwa*
