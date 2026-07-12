"""
path_visualizer.py
------------------
Standalone visual inspector for Path Finding outputs.

Companion to minus197_mapping/visualizer/map_visualizer.py. It draws the same
static floor context (zones, walls, features, and *all* graph nodes) and then
overlays the single generated route produced by run_pathfinding.py.

Per the visualisation brief this tool deliberately does NOT draw the graph
edges. Only the zones / areas / corridors, every node, and the generated path
are shown, so the route reads clearly against an uncluttered map.

It imports nothing from map_extraction / pathfinding / shared and only reads
JSON artefacts already on disk, so running or editing it cannot affect either
production pipeline.

Given a map *prefix* (e.g. "floor_4_itfac_mall_open_area_removed") it locates,
in the mapping output directory:

    <prefix>_sfm.json         walls, zones, features
    <prefix>_graph.json       navigation nodes (edges intentionally ignored)
    <prefix>_occupancy.json   five-state raster underlay (optional layer)

and overlays the route from a pathfinding feedback/output JSON — the file
written by `run_pathfinding.py --save-feedback ...`. That file is a list of
step objects, each with a node_id, a world-space position, and an
action/direction/landmark used for turn annotations.

Usage
-----
    # Auto-resolve map artefacts from a prefix; overlay one path output file
    python visualizer/path_visualizer.py \
        floor_4_itfac_mall_open_area_removed \
        --path ../minus197_path_finding/data/outputs/output_07112159.json

    # Point at a different mapping output directory
    python visualizer/path_visualizer.py <prefix> --path <out.json> \
        --map-dir ../minus197_mapping/data/outputs

    # Add the occupancy raster underlay
    python visualizer/path_visualizer.py <prefix> --path <out.json> --occupancy

    # Save straight to PNG instead of opening a window
    python visualizer/path_visualizer.py <prefix> --path <out.json> \
        --save route.png

    # List available map prefixes in the mapping output dir and exit
    python visualizer/path_visualizer.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
except ImportError as exc:  # pragma: no cover - dependency guard
    sys.exit(
        f"Missing dependency: {exc.name}. "
        f"Install with:  pip install matplotlib\n"
        f"(numpy is already a project dependency.)"
    )


# ── Appearance (kept in step with map_visualizer.py) ─────────────────────────

ZONE_FILL = {
    "corridor":   "#fff3b0",
    "shop":       "#a8e6cf",
    "food_court": "#ffd3b6",
    "restroom":   "#c7ceea",
    "entrance":   "#b5ead7",
    "exit":       "#ffb7b2",
    "storage":    "#d5c6e0",
    "office":     "#c9e4de",
    "unknown":    "#e8e8e8",
}

FEATURE_STYLE = {
    # feature_type -> (marker, colour, label)
    "door":       ("^", "#2e7d32", "door"),
    "stair":      ("*", "#c62828", "stair"),
    "elevator":   ("*", "#1565c0", "elevator"),
    "escalator":  ("*", "#6a1b9a", "escalator"),
    "info_desk":  ("D", "#7b1fa2", "info_desk"),
    "bench":      ("s", "#8d6e63", "bench"),
    "furnishing": (".", "#9e9e9e", "furnishing"),
}

NODE_COLOR = {
    "door":          "#2e7d32",
    "junction":      "#ef6c00",
    "zone_centroid": "#6a1b9a",
    "stair":         "#c62828",
    "elevator":      "#1565c0",
    "escalator":     "#6a1b9a",
}
NODE_DEFAULT_COLOR = "#455a64"

# Occupancy cell value -> RGB (0-1). Matches cell_legend in the JSON.
OCC_COLORS = {
    0: (1.00, 1.00, 1.00),   # walkable  - white
    1: (0.25, 0.25, 0.28),   # wall      - dark grey
    2: (0.30, 0.80, 0.35),   # door      - green
    3: (0.05, 0.05, 0.08),   # outside   - near black
    4: (1.00, 0.85, 0.20),   # uncertain - amber
}

# Path overlay palette.
PATH_LINE_COLOR = "#d81b60"   # magenta route line
PATH_HALO_COLOR = "#ffffff"   # white halo drawn under the route for contrast
PATH_NODE_COLOR = "#ad1457"   # waypoints along the route
START_COLOR = "#1b5e20"       # green start
END_COLOR = "#b71c1c"         # red goal
TURN_TEXT_COLOR = "#880e4f"


# ── Data loading ─────────────────────────────────────────────────────────────

def _load(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_map_paths(prefix: str, map_dir: Path):
    """Map a prefix to its three expected mapping artefact paths."""
    return {
        "sfm":       map_dir / f"{prefix}_sfm.json",
        "graph":     map_dir / f"{prefix}_graph.json",
        "occupancy": map_dir / f"{prefix}_occupancy.json",
    }


def list_prefixes(map_dir: Path):
    """Return sorted unique prefixes discoverable in the mapping output dir."""
    prefixes = set()
    for p in map_dir.glob("*_graph.json"):
        prefixes.add(p.name[: -len("_graph.json")])
    for p in map_dir.glob("*_sfm.json"):
        prefixes.add(p.name[: -len("_sfm.json")])
    return sorted(prefixes)


def extract_path_points(path_data, graph):
    """Turn a pathfinding output list into a list of step dicts with a
    resolved (x, y) position.

    Each output step carries its own ``position`` (world coords written by
    run_pathfinding.py); we prefer that but fall back to the node position in
    the graph if a step somehow lacks one. Steps whose position cannot be
    resolved at all are skipped with a warning.
    """
    graph_pos = {}
    if graph:
        graph_pos = {n["node_id"]: n.get("position") for n in graph.get("nodes", [])}

    steps = []
    for i, step in enumerate(path_data):
        node_id = step.get("node_id")
        pos = step.get("position") or graph_pos.get(node_id)
        if not pos:
            print(f"  ! step {i} ({node_id}) has no resolvable position - skipped")
            continue
        steps.append({
            "node_id": node_id,
            "pos": (float(pos[0]), float(pos[1])),
            "action": step.get("action"),
            "direction": step.get("direction"),
            "landmark": step.get("landmark"),
            "distance": step.get("distance"),
        })
    return steps


# ── Static layer renderers (mirror map_visualizer.py) ────────────────────────

def draw_occupancy(ax, occ):
    """Raster underlay. Drawn first so vector layers sit on top."""
    grid = np.asarray(occ["grid"], dtype=np.int16)
    h, w = grid.shape

    rgb = np.zeros((h, w, 3), dtype=float)
    for val, color in OCC_COLORS.items():
        rgb[grid == val] = color

    ox = occ.get("grid_origin_x", occ.get("origin", {}).get("x", 0.0))
    oy = occ.get("grid_origin_y", occ.get("origin", {}).get("y", 0.0))
    res = occ["resolution_m"]
    extent = [ox, ox + w * res, oy, oy + h * res]

    ax.imshow(rgb, extent=extent, origin="lower", interpolation="nearest",
              zorder=0, alpha=0.45)


def draw_zones(ax, sfm):
    for z in sfm.get("zones", []):
        poly = z.get("boundary_polygon")
        if not poly:
            continue
        cat = z.get("category", "unknown")
        color = ZONE_FILL.get(cat, ZONE_FILL["unknown"])
        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        ax.fill(xs, ys, facecolor=color, edgecolor="#888888",
                linewidth=0.6, alpha=0.45, zorder=1)

        cen = z.get("centroid")
        if cen:
            name = z.get("admin_name") or z.get("name") or cat
            ax.text(cen[0], cen[1], str(name), fontsize=6, ha="center",
                    va="center", color="#333333", zorder=6)


def draw_walls(ax, sfm):
    for w in sfm.get("walls", []):
        s, e = w.get("start"), w.get("end")
        if not s or not e:
            continue
        shore = w.get("shore_linable", False)
        ax.plot([s[0], e[0]], [s[1], e[1]],
                color="#1565c0" if shore else "#555555",
                linewidth=1.6 if shore else 1.1, zorder=3,
                solid_capstyle="round")


def draw_features(ax, sfm):
    for f in sfm.get("features", []):
        pos = f.get("position")
        if not pos:
            continue
        ftype = f.get("feature_type", "furnishing")
        marker, color, _ = FEATURE_STYLE.get(ftype, FEATURE_STYLE["furnishing"])
        ax.scatter(pos[0], pos[1], marker=marker, c=color, s=55,
                   edgecolors="black", linewidths=0.4, zorder=7)


def draw_nodes(ax, graph):
    """All graph nodes - edges are intentionally omitted per the brief."""
    for n in graph.get("nodes", []):
        pos = n.get("position")
        if not pos:
            continue
        ntype = n.get("node_type", "")
        color = NODE_COLOR.get(ntype, NODE_DEFAULT_COLOR)
        size = 14 if ntype == "junction" else 24
        ax.scatter(pos[0], pos[1], c=color, s=size,
                   edgecolors="black", linewidths=0.3, zorder=5, alpha=0.85)


# ── Path overlay ─────────────────────────────────────────────────────────────

def draw_path(ax, steps):
    """Draw the generated route: a haloed polyline, per-hop direction arrows,
    waypoint dots, start/goal markers, and turn/landmark annotations."""
    if len(steps) < 1:
        print("  ! path has no drawable points")
        return

    xs = [s["pos"][0] for s in steps]
    ys = [s["pos"][1] for s in steps]

    # White halo underneath, then the coloured route on top, so the line stays
    # readable over both light zones and the dark occupancy underlay.
    ax.plot(xs, ys, color=PATH_HALO_COLOR, linewidth=6.5, alpha=0.9,
            zorder=8, solid_capstyle="round", solid_joinstyle="round")
    ax.plot(xs, ys, color=PATH_LINE_COLOR, linewidth=3.0, alpha=0.95,
            zorder=9, solid_capstyle="round", solid_joinstyle="round")

    # Direction arrows at the midpoint of each hop.
    for a, b in zip(steps[:-1], steps[1:]):
        (x0, y0), (x1, y1) = a["pos"], b["pos"]
        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        dx, dy = x1 - x0, y1 - y0
        if dx == 0 and dy == 0:
            continue
        ax.annotate(
            "", xy=(mx + dx * 0.06, my + dy * 0.06),
            xytext=(mx - dx * 0.06, my - dy * 0.06),
            arrowprops=dict(arrowstyle="-|>", color=PATH_LINE_COLOR, lw=2.2),
            zorder=10,
        )

    # Intermediate waypoints.
    for s in steps[1:-1]:
        ax.scatter(*s["pos"], c=PATH_NODE_COLOR, s=45, edgecolors="white",
                   linewidths=0.8, zorder=11)

    # Turn / landmark annotations for steps that change direction or stop.
    for i, s in enumerate(steps):
        label = None
        if s.get("action") == "turn" and s.get("direction"):
            label = f"turn {s['direction']}"
        elif s.get("action") == "stop":
            label = f"STOP: {s['landmark']}" if s.get("landmark") else "STOP"
        # Skip the plain start/goal here; those get their own big markers.
        if label and 0 < i < len(steps):
            x, y = s["pos"]
            ax.annotate(label, xy=(x, y), xytext=(6, 6),
                        textcoords="offset points", fontsize=6.5,
                        color=TURN_TEXT_COLOR, weight="bold", zorder=12,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec=TURN_TEXT_COLOR, lw=0.5, alpha=0.85))

    # Start and goal markers.
    sx, sy = steps[0]["pos"]
    gx, gy = steps[-1]["pos"]
    ax.scatter(sx, sy, marker="o", c=START_COLOR, s=170, edgecolors="white",
               linewidths=1.4, zorder=13)
    ax.annotate("START", xy=(sx, sy), xytext=(8, -12),
                textcoords="offset points", fontsize=8, color="white",
                weight="bold", zorder=14,
                bbox=dict(boxstyle="round,pad=0.2", fc=START_COLOR, ec="white",
                          lw=0.6))
    goal_label = steps[-1].get("landmark") or "GOAL"
    ax.scatter(gx, gy, marker="*", c=END_COLOR, s=340, edgecolors="white",
               linewidths=1.4, zorder=13)
    ax.annotate(str(goal_label), xy=(gx, gy), xytext=(8, -12),
                textcoords="offset points", fontsize=8, color="white",
                weight="bold", zorder=14,
                bbox=dict(boxstyle="round,pad=0.2", fc=END_COLOR, ec="white",
                          lw=0.6))


# ── Legend ───────────────────────────────────────────────────────────────────

def build_legend(ax, layers_drawn):
    handles = []
    if "zones" in layers_drawn:
        for cat in ("corridor", "shop", "food_court", "unknown"):
            handles.append(Patch(facecolor=ZONE_FILL[cat], edgecolor="#888",
                                  alpha=0.6, label=f"zone: {cat}"))
    if "walls" in layers_drawn:
        handles.append(Line2D([0], [0], color="#555555", lw=1.5, label="wall"))
        handles.append(Line2D([0], [0], color="#1565c0", lw=1.8,
                              label="wall (shore-linable)"))
    if "nodes" in layers_drawn:
        for ntype, color in (("junction", NODE_COLOR["junction"]),
                             ("zone_centroid", NODE_COLOR["zone_centroid"]),
                             ("door", NODE_COLOR["door"])):
            handles.append(Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=color, markersize=7,
                                  markeredgecolor="black", label=f"node: {ntype}"))
    if "path" in layers_drawn:
        handles.append(Line2D([0], [0], color=PATH_LINE_COLOR, lw=3.0,
                              label="generated path"))
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=START_COLOR, markersize=9,
                              markeredgecolor="white", label="start"))
        handles.append(Line2D([0], [0], marker="*", color="w",
                              markerfacecolor=END_COLOR, markersize=13,
                              markeredgecolor="white", label="goal"))
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  fontsize=7, framealpha=0.9, borderaxespad=0.0)


# ── Orchestration ────────────────────────────────────────────────────────────

def load_artefacts(prefix, map_dir, path_file, want_occupancy):
    """Load sfm/graph/occupancy + the path output, reporting their state."""
    paths = resolve_map_paths(prefix, map_dir)
    sfm = _load(paths["sfm"])
    graph = _load(paths["graph"])
    occ = _load(paths["occupancy"]) if want_occupancy else None
    path_data = _load(Path(path_file))

    if sfm is None and graph is None:
        sys.exit(f"No _sfm.json or _graph.json found for prefix '{prefix}' "
                 f"in {map_dir}. Try --list to see available prefixes.")
    if path_data is None:
        sys.exit(f"Path output file not found or unreadable: {path_file}")
    if not isinstance(path_data, list):
        sys.exit(f"Unexpected path output format in {path_file} "
                 f"(expected a JSON list of steps).")

    for key, data in (("sfm", sfm), ("graph", graph)):
        state = "loaded" if data is not None else "MISSING"
        print(f"  {paths[key].name:<45} {state}")
    if want_occupancy:
        state = "loaded" if occ is not None else "MISSING"
        print(f"  {paths['occupancy'].name:<45} {state}")
    print(f"  {Path(path_file).name:<45} loaded ({len(path_data)} steps)")

    return sfm, graph, occ, path_data


def render_figure(prefix, sfm, graph, occ, steps, want, title_suffix=""):
    """Build one figure: static context (no edges) + the route overlay."""
    fig, ax = plt.subplots(figsize=(15, 9))
    layers_drawn = set()

    if occ is not None and want.get("occupancy"):
        draw_occupancy(ax, occ)
        layers_drawn.add("occupancy")

    if sfm is not None:
        if want.get("zones"):
            draw_zones(ax, sfm); layers_drawn.add("zones")
        if want.get("walls"):
            draw_walls(ax, sfm); layers_drawn.add("walls")
        if want.get("features"):
            draw_features(ax, sfm); layers_drawn.add("features")

    # Nodes only - edges are deliberately never drawn here.
    if graph is not None and want.get("nodes"):
        draw_nodes(ax, graph); layers_drawn.add("nodes")

    draw_path(ax, steps); layers_drawn.add("path")

    n_zones = len(sfm.get("zones", [])) if sfm else 0
    n_walls = len(sfm.get("walls", [])) if sfm else 0
    n_feats = len(sfm.get("features", [])) if sfm else 0
    n_nodes = len(graph.get("nodes", [])) if graph else 0

    start_id = steps[0]["node_id"] if steps else "?"
    goal_id = steps[-1]["node_id"] if steps else "?"
    title = (f"{prefix}   |   route: {start_id} -> {goal_id}   |   "
             f"steps={len(steps)}  zones={n_zones}  walls={n_walls}  "
             f"features={n_feats}  nodes={n_nodes}")
    if title_suffix:
        title += f"   |   {title_suffix}"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X (metres, IFC project frame)")
    ax.set_ylabel("Y (metres, IFC project frame)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

    build_legend(ax, layers_drawn)
    fig.tight_layout()
    return fig


def visualize(prefix, map_dir, path_file, want, save_path=None):
    sfm, graph, occ, path_data = load_artefacts(
        prefix, map_dir, path_file, want["occupancy"])
    steps = extract_path_points(path_data, graph)

    fig = render_figure(prefix, sfm, graph, occ, steps, want)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved figure -> {save_path}")
    else:
        print("\nOpening interactive window (close it to exit)...")
        plt.show()
    plt.close(fig)


def main():
    default_map_dir = (Path(__file__).resolve().parent.parent.parent
                       / "minus197_mapping" / "data" / "outputs")

    parser = argparse.ArgumentParser(
        description="Visualize a Path Finding route over its floor map.")
    parser.add_argument("prefix", nargs="?",
                        help="Map file prefix, e.g. "
                             "floor_4_itfac_mall_open_area_removed")
    parser.add_argument("--path", default=None,
                        help="Path output JSON written by run_pathfinding.py "
                             "--save-feedback (a list of route steps)")
    parser.add_argument("--map-dir", default=str(default_map_dir),
                        help=f"Mapping output directory "
                             f"(default: {default_map_dir})")
    parser.add_argument("--occupancy", action="store_true",
                        help="Draw the occupancy raster underlay")
    parser.add_argument("--no-zones",    action="store_true", help="Hide zones")
    parser.add_argument("--no-walls",    action="store_true", help="Hide walls")
    parser.add_argument("--no-features", action="store_true", help="Hide features")
    parser.add_argument("--no-nodes",    action="store_true", help="Hide graph nodes")
    parser.add_argument("--save", default=None,
                        help="Save to this PNG path instead of showing a window")
    parser.add_argument("--list", action="store_true",
                        help="List available map prefixes and exit")
    args = parser.parse_args()

    map_dir = Path(args.map_dir)
    if not map_dir.exists():
        sys.exit(f"Mapping output directory not found: {map_dir}")

    if args.list:
        prefixes = list_prefixes(map_dir)
        if not prefixes:
            print(f"No graph/sfm outputs found in {map_dir}")
        else:
            print(f"Available map prefixes in {map_dir}:")
            for p in prefixes:
                print(f"  {p}")
        return

    if not args.prefix:
        parser.error("a map prefix is required (or use --list). e.g. "
                     "python visualizer/path_visualizer.py "
                     "floor_4_itfac_mall_open_area_removed "
                     "--path ../minus197_path_finding/data/outputs/output_07112159.json")
    if not args.path:
        parser.error("--path is required: the pathfinding output JSON to overlay.")

    want = {
        "occupancy": args.occupancy,
        "zones":     not args.no_zones,
        "walls":     not args.no_walls,
        "features":  not args.no_features,
        "nodes":     not args.no_nodes,
    }

    print(f"Visualizing route from {args.path}")
    print(f"  over map prefix '{args.prefix}' in {map_dir}")
    visualize(args.prefix, map_dir, args.path, want, save_path=args.save)


if __name__ == "__main__":
    main()
