# Run Commands — Multi-Floor Single-IFC Mall (generate + admin GUI + visualize)

Terminal commands for the full workflow when **one IFC file contains several
storeys** (a multi-floor mall in a single IFC). Covers:

1. Generating the per-floor outputs.
2. Naming shops with the admin GUI (one window, a tab per floor).
3. Running the visualizers and **saving** what they render to PNG.

All commands are run **from the `minus197_mapping/` directory**. JSON outputs land
in `data/outputs/`; saved PNGs land in `data/outputs/images/`. The examples use
`itfac_mall_floors_3-4_combined.ifc` — swap in your own IFC path / building name.

```bash
cd minus197_mapping
```

Design and file layout background: see [MULTIFLOOR_SINGLE_IFC.md](MULTIFLOOR_SINGLE_IFC.md).

---

## 1. Generate per-floor outputs from the single multi-storey IFC

Splits the one IFC into per-floor artifacts (`_graph`, `_sfm`, `_path_nodes`,
`_shop_names`, `_occupancy`) plus a stitched `<Building>_building_graph.json`.

```bash
python main.py --multifloor-ifc data/ifc_files/itfac_mall_floors_3-4_combined.ifc --building "ITFAC Mall"
```

Each floor's SFM is written as `<stem>_<floor>_sfm.json`
(e.g. `itfac_mall_floors_3-4_combined_L3_sfm.json`,
`itfac_mall_floors_3-4_combined_L4_sfm.json`), so downstream tools key off the
per-floor prefix `itfac_mall_floors_3-4_combined_L3`,
`itfac_mall_floors_3-4_combined_L4`, etc.

The `--building` value names the stitched graph file
(`<building>_building_graph.json`), so keep it consistent across runs.

---

## 2. Name shops with the admin GUI (one window, a tab per floor)

Same command **plus `--admin-gui`**. It re-runs generation, then opens a single
Tk window with a `ttk.Notebook` — a full click-to-name floor plan per storey.
Each tab autosaves its own `<stem>_<floor>_shop_names.json`; the global buttons
under the tabs finish the whole session at once.

```bash
python main.py --multifloor-ifc data/ifc_files/itfac_mall_floors_3-4_combined.ifc --building "ITFAC Mall" --admin-gui
```

> With only one populated floor it falls back to the plain single-window GUI.
> `--admin-name` (text CLI) instead names one floor after another, sequentially.

---

## 3. Visualize and save the results

The three visualizers are **read-only** — they import nothing from the pipeline
and only read `data/outputs/`. Each takes a **file prefix**
(e.g. `itfac_mall_floors_3-4_combined_L3`), not a full path. Run them once per
floor prefix, or pass multiple prefixes where supported.

**Where PNGs are saved.** Point `--save` at `data/outputs/images/` to keep every
rendered image in one place. The folder must already exist — matplotlib will not
create it — so create it once if it is missing:

```bash
mkdir -p data/outputs/images
```

### 3a. Map visualizer — walls / zones / features / graph

`--save <file>` writes that exact PNG path instead of opening a window.

```bash
# List available prefixes in data/outputs/
python visualizer/map_visualizer.py --list

# Save the vector overlay for each floor into data/outputs/images/
python visualizer/map_visualizer.py itfac_mall_floors_3-4_combined_L3 --save data/outputs/images/itfac_mall_floors_3-4_combined_L3.png
python visualizer/map_visualizer.py itfac_mall_floors_3-4_combined_L4 --save data/outputs/images/itfac_mall_floors_3-4_combined_L4.png

# Same, with the five-state occupancy raster underlay
python visualizer/map_visualizer.py itfac_mall_floors_3-4_combined_L3 --occupancy --save data/outputs/images/itfac_mall_floors_3-4_combined_L3_occ.png
```

### 3b. Path-node visualizer — path nodes in blue over the base map

Save behaviour depends on how `--save` is given:

| Form | Result |
|---|---|
| `--save` (no value) | Auto-names into **`visualizer/`** as `<prefix>_path_nodes.png` |
| `--save <file>`, one prefix | Writes exactly that file |
| `--save <file>`, several prefixes | Keeps only the **directory** and `.png` suffix, then writes one `<prefix>_path_nodes.png` per floor |

So to land the images in `data/outputs/images/`, always pass an explicit path —
a bare `--save` would put them in `visualizer/` instead.

```bash
# List prefixes that have a _path_nodes.json
python visualizer/path_node_visualizer.py --list

# Both floors at once -> data/outputs/images/<prefix>_path_nodes.png each.
# With several prefixes the filename below is ignored (only its folder and
# .png suffix are used), so it acts purely as a directory placeholder.
python visualizer/path_node_visualizer.py itfac_mall_floors_3-4_combined_L3 itfac_mall_floors_3-4_combined_L4 --save data/outputs/images/x.png

# One floor at a time — here the filename IS honoured exactly
python visualizer/path_node_visualizer.py itfac_mall_floors_3-4_combined_L3 --save data/outputs/images/itfac_mall_floors_3-4_combined_L3_path_nodes.png
python visualizer/path_node_visualizer.py itfac_mall_floors_3-4_combined_L4 --save data/outputs/images/itfac_mall_floors_3-4_combined_L4_path_nodes.png

# Render and save EVERY available prefix, each as its own image
python visualizer/path_node_visualizer.py --all --save data/outputs/images/x.png

# With the occupancy underlay
python visualizer/path_node_visualizer.py itfac_mall_floors_3-4_combined_L3 --occupancy --save data/outputs/images/itfac_mall_floors_3-4_combined_L3_path_nodes_occ.png
```

### 3c. Path-node GUI — interactive, click a node to inspect its search circle

Interactive inspector (no `--save`; take a screenshot to capture it).

```bash
# Give the prefix directly, then click path nodes to inspect them
python visualizer/path_node_gui.py itfac_mall_floors_3-4_combined_L3

# Prompt for a prefix / add the occupancy underlay
python visualizer/path_node_gui.py itfac_mall_floors_3-4_combined_L4 --occupancy
```

---

## Output prefixes at a glance

For `itfac_mall_floors_3-4_combined.ifc` with populated Level 3 and Level 4:

| Prefix | Files under `data/outputs/` |
|---|---|
| `itfac_mall_floors_3-4_combined_L3` | `_graph.json`, `_sfm.json`, `_path_nodes.json`, `_shop_names.json`, `_occupancy.json` |
| `itfac_mall_floors_3-4_combined_L4` | `_graph.json`, `_sfm.json`, `_path_nodes.json`, `_shop_names.json`, `_occupancy.json` |
| `--building` value | `<building>_building_graph.json` (L3 ↔ L4 via stair + elevator shafts) |

Saved PNGs land in `data/outputs/images/` when you pass an explicit
`--save data/outputs/images/<name>.png`. A bare `--save` on
`path_node_visualizer.py` writes to `visualizer/` instead.
