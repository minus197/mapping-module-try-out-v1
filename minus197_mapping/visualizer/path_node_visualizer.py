"""
path_node_visualizer.py
-----------------------
Standalone visual inspector for the PATH-NODE layer produced by
map_extraction/path_nodes.py (<prefix>_path_nodes.json).

Like map_visualizer.py, this tool is completely independent of the main
pipeline — it imports nothing from map_extraction / pathfinding / shared and
only reads the JSON artefacts already written to data/outputs/. Running or
editing it cannot affect the production pipeline.

It draws the same base map as map_visualizer.py — zones, walls, features and
the navigation graph nodes — and then overlays the generated path nodes in
BLUE on top so they are easy to tell apart from the ordinary graph nodes.

Given a file *prefix* (e.g. "floor_4_itfac_mall_with_open_area"), it locates
and overlays:

    <prefix>_sfm.json          walls, zones, features        (base layer)
    <prefix>_graph.json        navigation nodes (+ edges)     (base layer)
    <prefix>_path_nodes.json   path nodes                     (BLUE overlay)
    <prefix>_occupancy.json    five-state raster underlay     (optional)

Every layer is drawn only if its file exists; missing files are reported and
skipped rather than treated as errors.

Usage
-----
    # Auto-resolve files from a prefix in the default output dir and open a window
    python visualizer/path_node_visualizer.py floor_4_itfac_mall_with_open_area

    # Point at a different output directory
    python visualizer/path_node_visualizer.py <prefix> --dir data/outputs

    # Add the occupancy raster underlay
    python visualizer/path_node_visualizer.py <prefix> --occupancy

    # Hide the ordinary graph nodes/edges so only path nodes stand out
    python visualizer/path_node_visualizer.py <prefix> --no-graph-nodes --no-edges

    # Save straight to PNG (into visualizer/ by default) instead of a window
    python visualizer/path_node_visualizer.py <prefix> --save

    # List available prefixes (those that have a _path_nodes.json) and exit
    python visualizer/path_node_visualizer.py --list
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


# ── Appearance ────────────────────────────────────────────────────────────────
# Base-map colours mirror map_visualizer.py so the two tools look consistent.

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

# Ordinary navigation-graph nodes are drawn muted grey so the BLUE path nodes
# clearly dominate the picture.
GRAPH_NODE_COLOR = "#9e9e9e"

# The path nodes — the whole point of this tool — in a strong blue.
PATH_NODE_COLOR = "#1565c0"

# Occupancy cell value -> RGB (0-1). Matches cell_legend in the JSON.
OCC_COLORS = {
    0: (1.00, 1.00, 1.00),   # walkable  - white
    1: (0.25, 0.25, 0.28),   # wall      - dark grey
    2: (0.30, 0.80, 0.35),   # door      - green
    3: (0.05, 0.05, 0.08),   # outside   - near black
    4: (1.00, 0.85, 0.20),   # uncertain - amber
}


# ── Data loading ────────────────────────────────────────────────────────────

def _load(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_paths(prefix: str, out_dir: Path):
    """Map a prefix to its expected artefact paths."""
    return {
        "sfm":        out_dir / f"{prefix}_sfm.json",
        "graph":      out_dir / f"{prefix}_graph.json",
        "path_nodes": out_dir / f"{prefix}_path_nodes.json",
        "occupancy":  out_dir / f"{prefix}_occupancy.json",
    }


def list_prefixes(out_dir: Path):
    """Return sorted prefixes that have a _path_nodes.json in the output dir."""
    prefixes = set()
    for p in out_dir.glob("*_path_nodes.json"):
        prefixes.add(p.name[: -len("_path_nodes.json")])
    return sorted(prefixes)


# ── Layer renderers ───────────────────────────────────────────────────────────

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
              zorder=0, alpha=0.55)


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
                solid_capstyle="round", alpha=0.9 if shore else 1.0)


def draw_features(ax, sfm):
    for f in sfm.get("features", []):
        pos = f.get("position")
        if not pos:
            continue
        ftype = f.get("feature_type", "furnishing")
        marker, color, _ = FEATURE_STYLE.get(ftype, FEATURE_STYLE["furnishing"])
        ax.scatter(pos[0], pos[1], marker=marker, c=color, s=55,
                   edgecolors="black", linewidths=0.4, zorder=7)


def draw_edges(ax, graph):
    """Ordinary navigation-graph edges, drawn thin and grey as context."""
    pos = {n["node_id"]: n["position"] for n in graph.get("nodes", [])}
    for e in graph.get("edges", []):
        a = pos.get(e["source_id"])
        b = pos.get(e["target_id"])
        if a is None or b is None:
            continue
        ax.plot([a[0], b[0]], [a[1], b[1]],
                color="#bdbdbd", linewidth=0.6, alpha=0.7,
                zorder=4, solid_capstyle="round")


def draw_graph_nodes(ax, graph):
    """Ordinary navigation-graph nodes, muted grey so path nodes dominate."""
    for n in graph.get("nodes", []):
        pos = n.get("position")
        if not pos:
            continue
        ntype = n.get("node_type", "")
        size = 14 if ntype == "junction" else 22
        ax.scatter(pos[0], pos[1], c=GRAPH_NODE_COLOR, s=size,
                   edgecolors="#555555", linewidths=0.3, zorder=5, alpha=0.8)


def draw_path_nodes(ax, path_nodes):
    """The generated path nodes — BLUE, on top of everything else."""
    pts = [pn.get("position") for pn in path_nodes.get("path_nodes", [])
           if pn.get("position")]
    if not pts:
        return 0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.scatter(xs, ys, c=PATH_NODE_COLOR, s=40,
               edgecolors="white", linewidths=0.6, zorder=8, marker="o")
    return len(pts)


# ── Legend ────────────────────────────────────────────────────────────────────

def build_legend(ax, layers_drawn):
    handles = []
    if "zones" in layers_drawn:
        for cat in ("corridor", "shop", "unknown"):
            handles.append(Patch(facecolor=ZONE_FILL[cat], edgecolor="#888",
                                  alpha=0.6, label=f"zone: {cat}"))
    if "walls" in layers_drawn:
        handles.append(Line2D([0], [0], color="#555555", lw=1.5, label="wall"))
        handles.append(Line2D([0], [0], color="#1565c0", lw=1.8,
                              label="wall (shore-linable)"))
    if "edges" in layers_drawn:
        handles.append(Line2D([0], [0], color="#bdbdbd", lw=1.0,
                              label="graph edge"))
    if "graph_nodes" in layers_drawn:
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=GRAPH_NODE_COLOR, markersize=7,
                              markeredgecolor="#555555", label="graph node"))
    if "path_nodes" in layers_drawn:
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=PATH_NODE_COLOR, markersize=9,
                              markeredgecolor="white",
                              label="path node (0.3 m off wall)"))
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  fontsize=7, framealpha=0.9, borderaxespad=0.0)


# ── Orchestration ──────────────────────────────────────────────────────────────

def load_artefacts(prefix, out_dir, want_occupancy):
    """Load sfm/graph/path_nodes/occupancy JSON for a prefix and report state."""
    paths = resolve_paths(prefix, out_dir)
    sfm        = _load(paths["sfm"])
    graph      = _load(paths["graph"])
    path_nodes = _load(paths["path_nodes"])
    occ        = _load(paths["occupancy"]) if want_occupancy else None

    if path_nodes is None:
        sys.exit(
            f"No _path_nodes.json found for prefix '{prefix}' in {out_dir}. "
            f"Run the pipeline first (it writes <prefix>_path_nodes.json), "
            f"or use --list to see available prefixes."
        )

    for key in ("sfm", "graph", "path_nodes"):
        data = {"sfm": sfm, "graph": graph, "path_nodes": path_nodes}[key]
        state = "loaded" if data is not None else "MISSING"
        print(f"  {paths[key].name:<50} {state}")
    if want_occupancy:
        state = "loaded" if occ is not None else "MISSING"
        print(f"  {paths['occupancy'].name:<50} {state}")

    return sfm, graph, path_nodes, occ


def render_figure(prefix, sfm, graph, path_nodes, occ, want):
    """Build one Matplotlib figure/axes for the requested layer combination."""
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

    if graph is not None:
        if want.get("edges"):
            draw_edges(ax, graph); layers_drawn.add("edges")
        if want.get("graph_nodes"):
            draw_graph_nodes(ax, graph); layers_drawn.add("graph_nodes")

    n_path = 0
    if path_nodes is not None and want.get("path_nodes"):
        n_path = draw_path_nodes(ax, path_nodes)
        layers_drawn.add("path_nodes")

    n_zones = len(sfm.get("zones", [])) if sfm else 0
    n_walls = len(sfm.get("walls", [])) if sfm else 0
    n_nodes = len(graph.get("nodes", [])) if graph else 0

    title = (f"{prefix}   |   zones={n_zones}  walls={n_walls}  "
             f"graph_nodes={n_nodes}  path_nodes={n_path}")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X (metres, IFC project frame)")
    ax.set_ylabel("Y (metres, IFC project frame)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

    build_legend(ax, layers_drawn)
    fig.tight_layout()
    return fig


def visualize(prefix, out_dir, want, save_path=None):
    sfm, graph, path_nodes, occ = load_artefacts(
        prefix, out_dir, want["occupancy"]
    )
    fig = render_figure(prefix, sfm, graph, path_nodes, occ, want)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved figure -> {save_path}")
    else:
        print("\nOpening interactive window (close it to exit)...")
        plt.show()
    plt.close(fig)


def main():
    default_dir = Path(__file__).resolve().parent.parent / "data" / "outputs"
    visualizer_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Visualize generated PATH NODES (blue) over the map.")
    parser.add_argument("prefix", nargs="?",
                        help="File prefix, e.g. floor_4_itfac_mall_with_open_area")
    parser.add_argument("--dir", default=str(default_dir),
                        help=f"Output directory (default: {default_dir})")
    parser.add_argument("--occupancy", action="store_true",
                        help="Draw the occupancy raster underlay")
    parser.add_argument("--no-zones",       action="store_true", help="Hide zones")
    parser.add_argument("--no-walls",       action="store_true", help="Hide walls")
    parser.add_argument("--no-features",    action="store_true", help="Hide features")
    parser.add_argument("--no-edges",       action="store_true", help="Hide graph edges")
    parser.add_argument("--no-graph-nodes", action="store_true",
                        help="Hide the ordinary navigation-graph nodes")
    parser.add_argument("--save", nargs="?", const="__auto__", default=None,
                        help="Save to PNG instead of showing a window. With no "
                             "value, saves to visualizer/<prefix>_path_nodes.png")
    parser.add_argument("--list", action="store_true",
                        help="List prefixes that have a _path_nodes.json and exit")
    args = parser.parse_args()

    out_dir = Path(args.dir)
    if not out_dir.exists():
        sys.exit(f"Output directory not found: {out_dir}")

    if args.list:
        prefixes = list_prefixes(out_dir)
        if not prefixes:
            print(f"No *_path_nodes.json outputs found in {out_dir}")
        else:
            print(f"Available path-node prefixes in {out_dir}:")
            for p in prefixes:
                print(f"  {p}")
        return

    if not args.prefix:
        parser.error("a prefix is required (or use --list). e.g. "
                     "python visualizer/path_node_visualizer.py "
                     "floor_4_itfac_mall_with_open_area")

    want = {
        "occupancy":   args.occupancy,
        "zones":       not args.no_zones,
        "walls":       not args.no_walls,
        "features":    not args.no_features,
        "edges":       not args.no_edges,
        "graph_nodes": not args.no_graph_nodes,
        "path_nodes":  True,
    }

    save_path = None
    if args.save is not None:
        if args.save == "__auto__":
            save_path = visualizer_dir / f"{args.prefix}_path_nodes.png"
        else:
            save_path = Path(args.save)

    print(f"Visualizing path nodes for prefix '{args.prefix}' from {out_dir}")
    visualize(args.prefix, out_dir, want, save_path=save_path)


if __name__ == "__main__":
    main()
