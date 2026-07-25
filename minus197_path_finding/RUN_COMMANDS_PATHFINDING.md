# Run Commands — Path Finding (generate a path + visualize it)

Terminal commands for generating a route with `run_pathfinding.py` and then
rendering it with the visualizers.

All commands are run **from the `minus197_path_finding/` directory** unless noted:

```bash
cd minus197_path_finding
```

Graph/map artefacts live in `../minus197_mapping/data/outputs/`. Route JSON
files are saved into `data/outputs/`. Replace the prefix
(`floor_4_itfac_mall_with_open_area`, `floors_3-4_combined_L3`, …), the node ids,
and the shop names with your own.

> **Node id vs. name.** You can specify start/destination either by **node id**
> (`--from` / `--to`) or by **shop/admin name** (`--from-name` / `--to-name`).
> The examples below use ids because that's what you asked for; swap in the
> `--*-name` flags anywhere if that's easier.

---

## 1. Generate a path — current node id → destination node id (single floor)

Basic query. `--from` is the **current node id**, `--to` is the **destination
node id**. Prints `found`, distance/cost/scores, and the turn-by-turn steps.

```bash
python run_pathfinding.py --graph ../minus197_mapping/data/outputs/floor_4_itfac_mall_with_open_area_graph.json --from ZONE-2SAGvqDRX6DRBKPckG2ybK --to ZONE-233bRpVtr7PADnPMIOrRa2
```

> The **current node id can be a graph node** (junction / door / stair / lift /
> zone_centroid) **or a path node id** (`PATH-xxxx`). In this single-floor mode a
> path-node current node is only recognised when `--path-nodes` is also supplied
> (it's how the engine looks the id up) — see §1b. In multi-floor mode the path
> nodes load automatically — see §2a.

### 1a. Save the route JSON (required for visualization)

Add `--save-feedback <file>`. The visualizers read this file. Without it, the
route is only printed to the terminal.

```bash
python run_pathfinding.py --graph ../minus197_mapping/data/outputs/floor_4_itfac_mall_with_open_area_graph.json --from ZONE-2SAGvqDRX6DRBKPckG2ybK --to ZONE-233bRpVtr7PADnPMIOrRa2 --save-feedback data/outputs/route.json
```

Add `--feedback-json` to also print that JSON to the terminal, or `--evaluate`
to run the endpoint check (route actually starts/ends at the requested nodes).

### 1b. Path-node-adjusted route (hug the cane-trailing wall layer)

Add `--path-nodes` + `--sfm` so the route snaps onto the cane-trailing path-node
layer along corridor legs. **Both flags are required together.** This is also
what lets a `PATH-xxxx` id be used as the current node.

```bash
python run_pathfinding.py --graph ../minus197_mapping/data/outputs/floor_4_itfac_mall_with_open_area_graph.json --path-nodes ../minus197_mapping/data/outputs/floor_4_itfac_mall_with_open_area_path_nodes.json --sfm ../minus197_mapping/data/outputs/floor_4_itfac_mall_with_open_area_sfm.json --from ZONE-2SAGvqDRX6DRBKPckG2ybK --to ZONE-233bRpVtr7PADnPMIOrRa2 --save-feedback data/outputs/adjusted_route.json
```

---

## 2. Generate a path across MULTIPLE FLOORS (inter-floor routing)

When the current node and the destination are on **different floors**, use
`--building-graph` **instead of** `--graph`. The router runs the single-floor
pipeline once per floor and stitches them with a `take_elevator` action (elevator,
not stairs — the users are visually impaired).

- `--building-graph` → the stitched `<Building>_building_graph.json`.
- `--outputs-dir`    → where the per-floor `_path_nodes.json` / `_sfm.json` live
  (defaults to the building graph's own directory, so usually omit it).

> **Do not pass a building graph to `--graph`.** A building graph has a different
> shape (`meta` / `floors` / `inter_floor_edges`) with no top-level `nodes`, so it
> would load as an empty graph. The script now detects this and tells you to
> switch to `--building-graph`.

The path-node layer is loaded **automatically per floor** in this mode — there is
no `--path-nodes` / `--sfm` flag to pass. Each floor's `_path_nodes.json` /
`_sfm.json` are resolved from `--outputs-dir` by the `<ifc_stem>_<floor>_*`
naming convention, so both legs are path-node-adjusted and a `PATH-xxxx` id works
as the current node.

```bash
python run_pathfinding.py --building-graph ../minus197_mapping/data/outputs/ITFAC_Mall_building_graph.json --from ZONE-<current-node-on-floor-L3> --to ZONE-<destination-node-on-floor-L4> --save-feedback data/outputs/route_L3_to_L4.json
```

By name instead of id (building-wide lookup):

```bash
python run_pathfinding.py --building-graph ../minus197_mapping/data/outputs/ITFAC_Mall_building_graph.json --from-name SINGER --to-name KEELLS --save-feedback data/outputs/route_LG_to_KEELLS.json
```

> If both nodes turn out to be on the **same floor**, the building-graph mode
> still works — it just routes on that one floor. `same_floor: true` is printed.

### 2a. Path-node current node across floors — qualify the floor

**Every floor numbers its own path nodes from `PATH-0000`**, so a bare
`PATH-xxxx` id is ambiguous in a multi-floor building. Prefix it with the floor
label — `L3:PATH-0000` — to say which one you mean:

```bash
python run_pathfinding.py --building-graph ../minus197_mapping/data/outputs/itfac_mall_floors_3-4_combined_full_building_graph.json --from L3:PATH-0000 --to ZONE-233bRpVtr7PADnPMIOrRa2 --save-feedback data/outputs/itfac_mall_floors_3-4_combined_PATH-0000-POPEYES_Route.json
```

The `FLOOR:NODE` prefix works for any node id, not just path nodes. Rules:

- **Graph node ids** (`ZONE-*`, `FEAT-*`) are unique building-wide — never need a
  prefix.
- **Path-node ids** need one whenever more than one floor defines them. If only
  one floor does, the bare id still works.
- An ambiguous bare id is **rejected with an error listing the options** rather
  than silently picking a floor:
  ```
  'PATH-0000' is ambiguous - it exists on L3, L4. Qualify it: L3:PATH-0000, L4:PATH-0000
  ```
- A path node can only be the **current node**, never the destination. `--to`
  must be a graph node (e.g. a `ZONE-*` id).

---

## 3. Visualize the generated path

Both visualizers are **read-only**: they only read JSON from disk and take a map
**prefix** (e.g. `floor_4_itfac_mall_with_open_area`), not a full graph path.
They auto-resolve `<prefix>_sfm.json` / `_graph.json` / `_occupancy.json` from
the mapping output directory, and overlay the route from your `--path <file>`
(the `--save-feedback` JSON from §1/§2).

`--save <file.png>` writes a PNG; omit it to open an interactive window.

### 3a. Plain route over the map — `path_visualizer.py`

Use this for a **single-floor** route that was generated **without** the
path-node adjustment (§1 / §1a).

```bash
# List available map prefixes
python visualizer/path_visualizer.py --list

# Open an interactive window
python visualizer/path_visualizer.py floor_4_itfac_mall_with_open_area --path data/outputs/route.json

# Save straight to a PNG
python visualizer/path_visualizer.py floor_4_itfac_mall_with_open_area --path data/outputs/route.json --save data/outputs/route_view.png

# With the five-state occupancy raster underlay
python visualizer/path_visualizer.py floor_4_itfac_mall_with_open_area --path data/outputs/route.json --occupancy --save data/outputs/route_view.png
```

Layer toggles: `--no-zones`, `--no-walls`, `--no-features`, `--no-nodes`.

### 3b. Path-node-adjusted route — `path_node_visualizer.py`

Use this for a route generated **with** `--path-nodes` (§1b) — it additionally
draws the full cane-trailing path-node layer and colours which legs hug the wall.

```bash
python visualizer/path_node_visualizer.py floor_4_itfac_mall_with_open_area --path data/outputs/adjusted_route.json --save data/outputs/adjusted_route_view.png

# Hide the background (un-used) path nodes, keep only the route's own
python visualizer/path_node_visualizer.py floor_4_itfac_mall_with_open_area --path data/outputs/adjusted_route.json --no-path-node-layer --save
```

### 3c. Visualize a MULTI-FLOOR route

A route from §2 spans two floors and can't share one image, so
`path_node_visualizer.py` **splits it at the elevator and renders each floor
separately** against its own `<stem>_<floor>_*` artefacts. Give the base prefix;
the floor label is appended to the `--save` filename automatically.

```bash
python visualizer/path_node_visualizer.py itfac_mall_floors_3-4_combined --path data/outputs/itfac_mall_floors_3-4_combined_PATH-0000-POPEYES_Route.json --save data/outputs/itfac_mall_floors_3-4_combined_PATH-0000-POPEYES_Route.png
# -> ..._Route_L3.png   (start -> elevator)
# -> ..._Route_L4.png   (elevator -> destination)
```

Give the **base** prefix (`itfac_mall_floors_3-4_combined`), not a per-floor one
— the visualizer appends `_L3` / `_L4` itself to find each floor's artefacts.
`--list` shows the available prefixes.

Without `--save`, each floor opens in its own window in turn (close one to see
the next).

---

## Quick reference

| Task | Command shape |
|---|---|
| Path by node id (single floor) | `run_pathfinding.py --graph <graph> --from <id> --to <id>` |
| Path by shop name | `run_pathfinding.py --graph <graph> --from-name <A> --to-name <B>` |
| Save route JSON (for visualizing) | add `--save-feedback data/outputs/route.json` |
| Hug path-node wall layer (single floor) | add `--path-nodes <pn> --sfm <sfm>` |
| Path across floors | `run_pathfinding.py --building-graph <bg> --from <id> --to <id> --save-feedback …` |
| Start from a path node, across floors | `--building-graph <bg> --from L3:PATH-0000 --to ZONE-<dest>` (path nodes auto-load; floor prefix required) |
| Visualize plain route | `visualizer/path_visualizer.py <prefix> --path <route.json> --save <png>` |
| Visualize adjusted / multi-floor route | `visualizer/path_node_visualizer.py <prefix> --path <route.json> --save <png>` |

Related: [ACCEPT_PATH_NODE_AS_CURRENT_NODE.md](ACCEPT_PATH_NODE_AS_CURRENT_NODE.md)
(using a path-node id as the current node) and
[../minus197_mapping/RUN_COMMANDS_MULTIFLOOR_SINGLE_IFC.md](../minus197_mapping/RUN_COMMANDS_MULTIFLOOR_SINGLE_IFC.md)
(generating the per-floor + building-graph artefacts these commands consume).
