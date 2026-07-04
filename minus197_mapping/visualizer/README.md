# Map Extraction Visualizer

A standalone, read-only inspector for the JSON artefacts produced by the Map
Extraction pipeline. It imports **nothing** from `map_extraction`,
`pathfinding`, or `shared` and only reads files under `data/outputs/`, so it
cannot affect the production pipeline.

## Install

```bash
pip install matplotlib      # numpy is already a project dependency
```

## Usage

The only required input is a **file prefix**. The script derives the three
artefact paths from it (`<prefix>_sfm.json`, `<prefix>_graph.json`,
`<prefix>_occupancy.json`) and draws whichever exist.

```bash
# From the minus197_mapping/ directory:

# List available prefixes in data/outputs/
python visualizer/map_visualizer.py --list

# Vector overlay (walls + zones + features + graph nodes/edges)
python visualizer/map_visualizer.py floor_4_sankha_spaces

# Add the five-state occupancy raster underlay
python visualizer/map_visualizer.py floor_4_sankha_spaces --occupancy

# Save to PNG instead of opening a window
python visualizer/map_visualizer.py floor_4_sankha_spaces --save floor4.png

# Turn individual layers off
python visualizer/map_visualizer.py floor_4_sankha_spaces --no-edges --no-nodes
```

## Layers

| Layer | Source file | Notes |
|---|---|---|
| Occupancy raster | `_occupancy.json` | Off by default; enable with `--occupancy` |
| Zones (filled) | `_sfm.json` | Colour-coded by category, labelled at centroid |
| Walls | `_sfm.json` | Shore-linable walls highlighted in blue |
| Features | `_sfm.json` | Markers by type (door/stair/elevator/...) |
| Graph edges | `_graph.json` | Blue = shore-linable; line width ∝ safety_score |
| Graph nodes | `_graph.json` | Colour by node_type (junction/door/centroid/...) |

All coordinates are in the IFC project frame in metres, so every layer aligns
in the same axes.
