# Multi-Floor Support from a Single IFC

**Module:** `minus197_mapping`

Ingest **one** IFC file that contains several `IfcBuildingStorey`s, split it into
floors *internally*, and emit all **five** output artifacts **separately per floor**:

```
<stem>_<floor>_graph.json        navigation graph (nodes + edges)
<stem>_<floor>_sfm.json          semantic floor map
<stem>_<floor>_path_nodes.json   cane-trailing waypoints
<stem>_<floor>_shop_names.json   admin shop-name mappings (empty {} stub until named)
<stem>_<floor>_occupancy.json    hybrid occupancy grid for perception
```

…then stitch the floors together with the existing `InterFloorLinker` into one
`<building>_building_graph.json` for cross-floor pathfinding.

This sits alongside the two pre-existing modes and does not change either of them:

- **Single-floor** — one IFC = one floor (`--ifc … --floor …`).
- **Multi-floor, one IFC per floor** — several `--ifc/--floor` pairs.
- **Multi-floor, one IFC with several storeys** — the new mode described here
  (`--multifloor-ifc …`).

---

## 1. Design

Keep the single multi-storey IFC as the source of truth, but **split by
`IfcBuildingStorey` inside the parser** and run the *existing* per-floor extraction
once per populated storey. Nothing downstream of the parser changes its per-floor
behaviour — the change is only:

1. a **storey filter** in the parser (`IFCParser(target_storey_id=…)`),
2. a **storey-loop orchestrator** in the pipeline
   (`multi_floor_single_ifc` / `run_multi_single` / `save_multi_single`),
3. **per-floor output stems** so files don't overwrite each other.

Single-file ingest gives coordinate fidelity for free: all storeys share the project
origin and real storey elevations, so vertical connectors (stair / elevator shafts)
line up in XY across floors and the linker matches them at ~0 m distance.

---

## 2. How elements are keyed to storeys

Two different IFC relationships tie elements to a storey, and the parser reads
**both**:

| Element kind                         | Relationship                        | Field              |
|--------------------------------------|-------------------------------------|--------------------|
| Walls, doors, stairs, transport      | `IfcRelContainedInSpatialStructure` | `RelatingStructure`|
| Spaces                               | `IfcRelAggregates`                  | `RelatingObject`   |

`build_element_storey_map(model)` merges both into one `element_id → storey_id` map.
`IFCParser._on_target_storey(element)` uses it to skip elements that aren't on the
target storey; when `target_storey_id` is `None` (legacy single-floor mode) it always
returns `True`, so nothing is filtered.

---

## 3. Which storeys are processed

`list_populated_storeys(model)` returns `[(storey_id, name, elevation)]` sorted
bottom→top for storeys that actually contain navigable content — at least one wall
**or** at least one space. Empty reference storeys (e.g. a bare roof level) are
dropped so `GraphBuilder` / `StartNodeResolver` are never handed a floor with no
navigable nodes.

**True floor height** is computed as the elevation delta to the storey above and
passed to the linker as `floor_height_m`, so inter-floor edge distances reflect the
model rather than the linker's 4.0 m default. The topmost storey has no floor above,
so it carries no height.

**Floor labels** come from the storey name via `_label_from_storey_name`:
`"Level 3" → "L3"`, `"Level 4" → "L4"`; a name with no digits falls back to the
sanitised name (`"Ground Floor" → "Ground_Floor"`).

---

## 4. Files changed

### `map_extraction/ifc_parser.py`
- **`list_populated_storeys(model)`** and **`build_element_storey_map(model)`** —
  new module-level helpers.
- **`IFCParser.__init__(ifc_path, target_storey_id=None)`** — optional storey filter;
  builds `self._storey_of` in `parse()`.
- **`IFCParser._on_target_storey(element)`** — guard used at the top of
  `_extract_spaces`, `_extract_walls`, and each loop in `_extract_features`
  (transport, furnishing, door, stair).

### `map_extraction/pipeline.py`
- **`MapExtractionPipeline.__init__`** gains `target_storey_id` and `output_stem`;
  `run()` threads the storey id into the parser; `save()` uses the per-floor stem
  and now always writes a `<stem>_shop_names.json` **stub** (`{}`) so every floor has
  all five artifacts even before an admin naming session.
- **`multi_floor_single_ifc(...)`** (factory), **`run_multi_single()`** (storey-loop
  runner), **`save_multi_single(...)`** (writes 5 files/floor + one building graph).
- **`_label_from_storey_name(name)`** — module-level label helper.

### `admin_naming/shop_name_ui.py`
- **`_stem_from_sfm`** now derives the floor stem from the **SFM filename**
  (`floors_3-4_combined_L3_sfm.json → floors_3-4_combined_L3`) instead of
  `meta.source_file`. In multi-floor mode several floors share one IFC, so the
  source-file stem dropped the `_L3`/`_L4` suffix — the patcher then wrote/looked for
  files that don't exist, and named zones silently failed to apply. The filename
  always carries the correct per-floor stem; single-floor behaviour is unchanged.

### `admin_naming/shop_name_gui.py`
- **`ShopNameGUI.__init__`** gains a `container` param: it builds its widgets into
  that frame (a notebook tab) instead of always owning the toplevel; window-level
  setup (title/geometry/protocol) and `destroy()` only run when it owns the window.
- **`ShopNameGUI.on_save_patch`** — in tabbed mode the per-tab "Save & Apply to
  outputs" button now applies **that floor's** names to its output JSONs immediately
  (`_apply_patch_now`) and keeps the window open, instead of calling `destroy()`
  (a no-op for a tab) and applying nothing.
- **`run_gui_multi(sfm_paths, output_dir)`** — one Tk window with a `ttk.Notebook`,
  a full naming tab per floor, plus global Save & Apply / Save & Close buttons.
  Single-floor input falls back to `run_gui`.

### `main.py`
- New **`--multifloor-ifc`** flag, handled before the single/multi `--ifc` branch.
- `--admin-gui` on the multi-floor path calls `run_gui_multi` (tabbed);
  `--admin-name` stays per-floor sequential.
- **`--ifc`** is no longer `required`; you must supply either `--ifc` (one or more)
  or `--multifloor-ifc`.
- `sys.stdout`/`sys.stderr` are reconfigured to UTF-8 at startup so the box-drawing
  characters in the console banners don't raise `UnicodeEncodeError` on Windows'
  default `cp1252` console.

---

## 5. Run commands

```bash
cd minus197_mapping

# Split one multi-floor IFC into per-floor outputs + a building graph
python main.py --multifloor-ifc data/ifc_files/floors_3-4_combined.ifc \
               --building "ITFAC Mall"

# …then drive shop naming afterwards (GUI — one window, a tab per floor)
python main.py --multifloor-ifc data/ifc_files/floors_3-4_combined.ifc \
               --building "ITFAC Mall" --admin-gui
```

Each floor's SFM is written as `<stem>_<floor>_sfm.json`, so the naming tool
produces `<stem>_<floor>_shop_names.json` per floor.

### Admin naming across floors

- **`--admin-gui`** opens **one window with a tab per floor**
  (`admin_naming.shop_name_gui.run_gui_multi`). Each tab is a full click-to-name
  floor plan for one storey — name zones on the *Floor L3* tab, switch to *Floor L4*,
  etc. Every tab autosaves its own `<stem>_<floor>_shop_names.json`, and the global
  **Save & Apply all floors** / **Save & Close** buttons under the tabs finish the
  whole session at once. With only one populated floor it falls back to the plain
  single-window `run_gui`.
- **`--admin-name`** (text CLI) is inherently sequential and still names one floor at
  a time, one after another.

The single-floor GUI (`run_gui`) is unchanged; `ShopNameGUI` simply learned to build
its widgets into a supplied `container` (a notebook tab) instead of always owning the
top-level window.

---

## 6. Output layout

For a single IFC with populated Level 3 and Level 4 (and an empty Level 5 that is
skipped), `data/outputs/` contains:

```
<stem>_L3_graph.json
<stem>_L3_sfm.json
<stem>_L3_path_nodes.json
<stem>_L3_shop_names.json     (stub until admin names it)
<stem>_L3_occupancy.json

<stem>_L4_graph.json
<stem>_L4_sfm.json
<stem>_L4_path_nodes.json
<stem>_L4_shop_names.json
<stem>_L4_occupancy.json

<Building>_building_graph.json   (L3 ↔ L4 via stair + elevator shafts)
```

Perception consumes one `<floor>_occupancy.json` at a time; pathfinding consumes the
single building graph.

---

## 7. Verification checklist

- [ ] `list_populated_storeys()` drops empty reference storeys and returns only
      populated ones, bottom→top.
- [ ] Each floor's `_graph.json` contains only that floor's nodes (no two nodes from
      different floors sharing the same XY at different Z).
- [ ] `<Building>_building_graph.json` `meta.total_inter_edges` is **> 0** (expect 4
      for a two-floor file: up+down for the stair shaft and the elevator shaft). If
      it's 0, the shaft XY match failed.
- [ ] Inter-floor edge `distance` equals the storey elevation delta, not 4.0.
- [ ] All **five** files exist for **each** populated floor.
- [ ] Single-floor mode (`--ifc … --floor …`) still works unchanged
      (`target_storey_id=None` path) and now also emits its `_shop_names.json` stub.

---

## 8. Gotchas

1. **Empty reference storeys** must be skipped, or `GraphBuilder` / `StartNodeResolver`
   raise on a floor with no navigable nodes — handled by `list_populated_storeys`.
2. **Floor height** comes from the elevation delta between storeys; don't leave the
   linker on its 4.0 m default.
3. **Shaft XY alignment** is free with single-file ingest — both floors share the
   project origin, so `CONNECTOR_MATCH_RADIUS = 1.5 m` matches at ~0 m. This is the
   coordinate-consistency advantage of single-IFC ingest over separate files.
4. **IFC2x3 vs IFC4** — the parser already handles the IFC2x3 polyline fallbacks; no
   schema-specific change is needed. `by_type("IfcWall")` already catches
   `IfcWallStandardCase`.
5. **Windows console encoding** — the banners use box-drawing characters; `main.py`
   forces UTF-8 on stdout/stderr so a `cp1252` console doesn't crash the run.
