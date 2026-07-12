"""
map_visualizer.py
-----------------
Standalone visual inspector for Map Extraction outputs.

This tool is completely independent of the main pipeline — it imports nothing
from map_extraction / pathfinding / shared and only reads the JSON artefacts
already written to data/outputs/. Running or editing it cannot affect the
production pipeline.

Given a file *prefix* (e.g. "floor_4_sankha_spaces"), it locates and overlays:

    <prefix>_sfm.json         walls, zones, features
    <prefix>_graph.json       navigation nodes and edges
    <prefix>_occupancy.json   five-state raster underlay (optional layer)

Every layer is drawn only if its file exists; missing files are reported and
skipped rather than treated as errors.

Usage
-----
    # Auto-resolve all three files from a prefix in the default output dir
    python visualizer/map_visualizer.py floor_4_sankha_spaces

    # Point at a different output directory
    python visualizer/map_visualizer.py floor_4_sankha_spaces --dir data/outputs

    # Add the occupancy raster underlay
    python visualizer/map_visualizer.py floor_4_sankha_spaces --occupancy

    # Toggle individual layers off
    python visualizer/map_visualizer.py floor_4_sankha_spaces --no-edges --no-zones

    # Save straight to PNG instead of opening a window
    python visualizer/map_visualizer.py floor_4_sankha_spaces --save floor4.png

    # Render the standard 5-image progression (zones/features, +nodes,
    # +shore-linable edges, +width&safety edges, +both edge kinds) straight
    # into the visualizer/ folder
    python visualizer/map_visualizer.py floor_4_sankha_spaces --stages

    # List available prefixes in the output dir and exit
    python visualizer/map_visualizer.py --list
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


# ── Data loading ────────────────────────────────────────────────────────────

def _load(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_paths(prefix: str, out_dir: Path):
    """Map a prefix to its three expected artefact paths."""
    return {
        "sfm":       out_dir / f"{prefix}_sfm.json",
        "graph":     out_dir / f"{prefix}_graph.json",
        "occupancy": out_dir / f"{prefix}_occupancy.json",
    }


def list_prefixes(out_dir: Path):
    """Return sorted unique prefixes discoverable in the output directory."""
    prefixes = set()
    for p in out_dir.glob("*_graph.json"):
        prefixes.add(p.name[: -len("_graph.json")])
    for p in out_dir.glob("*_sfm.json"):
        prefixes.add(p.name[: -len("_sfm.json")])
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

    # grid row 0 == minimum y (see query_hint), so origin='lower' aligns
    # the raster with the world-coordinate vector layers.
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


def draw_edges_shorelinable(ax, graph):
    """Edge layer emphasising the shore-linable flag: shore-linable edges in
    blue, all others thin and grey. Line width is constant (flag-only view)."""
    pos = {n["node_id"]: n["position"] for n in graph.get("nodes", [])}
    for e in graph.get("edges", []):
        a = pos.get(e["source_id"])
        b = pos.get(e["target_id"])
        if a is None or b is None:
            continue
        shore = e.get("shore_linable", False)
        ax.plot([a[0], b[0]], [a[1], b[1]],
                color="#1e88e5" if shore else "#9e9e9e",
                linewidth=1.8 if shore else 0.6,
                alpha=0.9 if shore else 0.75,
                zorder=4.5 if shore else 4, solid_capstyle="round")


def draw_edges_width_safety(ax, graph):
    """Edge layer emphasising width/safety: line width scales with the
    edge's safety_score, coloured uniformly (safety-only view)."""
    pos = {n["node_id"]: n["position"] for n in graph.get("nodes", [])}
    for e in graph.get("edges", []):
        a = pos.get(e["source_id"])
        b = pos.get(e["target_id"])
        if a is None or b is None:
            continue
        safety = float(e.get("safety_score", 0.5))
        ax.plot([a[0], b[0]], [a[1], b[1]],
                color="#43a047",
                linewidth=0.6 + 1.8 * safety,
                alpha=0.75, zorder=4, solid_capstyle="round")


def draw_edges_combined(ax, graph):
    """Edge layer combining both: colour by shore-linable flag, width by
    safety score. This is the original behaviour, kept for the 'all' image."""
    pos = {n["node_id"]: n["position"] for n in graph.get("nodes", [])}
    for e in graph.get("edges", []):
        a = pos.get(e["source_id"])
        b = pos.get(e["target_id"])
        if a is None or b is None:
            continue
        shore = e.get("shore_linable", False)
        safety = float(e.get("safety_score", 0.5))
        ax.plot([a[0], b[0]], [a[1], b[1]],
                color="#1e88e5" if shore else "#9e9e9e",
                linewidth=0.6 + 1.8 * safety,
                alpha=0.75, zorder=4, solid_capstyle="round")


def draw_nodes(ax, graph):
    for n in graph.get("nodes", []):
        pos = n.get("position")
        if not pos:
            continue
        ntype = n.get("node_type", "")
        color = NODE_COLOR.get(ntype, NODE_DEFAULT_COLOR)
        size = 18 if ntype == "junction" else 30
        ax.scatter(pos[0], pos[1], c=color, s=size,
                   edgecolors="black", linewidths=0.3, zorder=5)


# ── Legend ────────────────────────────────────────────────────────────────────

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
    if "edges_shorelinable" in layers_drawn:
        handles.append(Line2D([0], [0], color="#9e9e9e", lw=0.6,
                              label="edge"))
        handles.append(Line2D([0], [0], color="#1e88e5", lw=1.8,
                              label="edge (shore-linable)"))
    if "edges_width_safety" in layers_drawn:
        handles.append(Line2D([0], [0], color="#43a047", lw=1.8,
                              label="edge (width ∝ safety)"))
    if "edges_combined" in layers_drawn:
        handles.append(Line2D([0], [0], color="#9e9e9e", lw=1.5,
                              label="edge (width ∝ safety)"))
        handles.append(Line2D([0], [0], color="#1e88e5", lw=1.8,
                              label="edge (shore-linable)"))
    if "nodes" in layers_drawn:
        for ntype, color in (("junction", NODE_COLOR["junction"]),
                             ("zone_centroid", NODE_COLOR["zone_centroid"]),
                             ("door", NODE_COLOR["door"])):
            handles.append(Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=color, markersize=7,
                                  markeredgecolor="black", label=f"node: {ntype}"))
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  fontsize=7, framealpha=0.9, borderaxespad=0.0)


# ── Orchestration ──────────────────────────────────────────────────────────────

def load_artefacts(prefix, out_dir, want_occupancy):
    """Load sfm/graph/occupancy JSON for a prefix and report their state."""
    paths = resolve_paths(prefix, out_dir)
    sfm = _load(paths["sfm"])
    graph = _load(paths["graph"])
    occ = _load(paths["occupancy"]) if want_occupancy else None

    if sfm is None and graph is None:
        sys.exit(f"No _sfm.json or _graph.json found for prefix '{prefix}' "
                 f"in {out_dir}. Try --list to see available prefixes.")

    for key, data in (("sfm", sfm), ("graph", graph)):
        state = "loaded" if data is not None else "MISSING"
        print(f"  {paths[key].name:<45} {state}")
    if want_occupancy:
        state = "loaded" if occ is not None else "MISSING"
        print(f"  {paths['occupancy'].name:<45} {state}")

    return sfm, graph, occ


def render_figure(prefix, sfm, graph, occ, want, title_suffix=""):
    """Build one Matplotlib figure/axes for the requested layer combination.

    `want` keys: occupancy, zones, walls, features, nodes,
    edges_shorelinable, edges_width_safety, edges_combined.
    Returns the created Figure.
    """
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
        # width/safety drawn before shore-linable so the shore-linable
        # highlight (drawn on top) stays visible when both layers are on.
        if want.get("edges_width_safety"):
            draw_edges_width_safety(ax, graph); layers_drawn.add("edges_width_safety")
        if want.get("edges_shorelinable"):
            draw_edges_shorelinable(ax, graph); layers_drawn.add("edges_shorelinable")
        if want.get("edges_combined"):
            draw_edges_combined(ax, graph); layers_drawn.add("edges_combined")
        if want.get("nodes"):
            draw_nodes(ax, graph); layers_drawn.add("nodes")

    n_zones = len(sfm.get("zones", [])) if sfm else 0
    n_walls = len(sfm.get("walls", [])) if sfm else 0
    n_feats = len(sfm.get("features", [])) if sfm else 0
    n_nodes = len(graph.get("nodes", [])) if graph else 0
    n_edges = len(graph.get("edges", [])) if graph else 0

    title = (f"{prefix}   |   zones={n_zones}  walls={n_walls}  "
             f"features={n_feats}  nodes={n_nodes}  edges={n_edges}")
    if title_suffix:
        title += f"   |   {title_suffix}"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X (metres, IFC project frame)")
    ax.set_ylabel("Y (metres, IFC project frame)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

    build_legend(ax, layers_drawn)
    fig.tight_layout()
    return fig


def visualize(prefix, out_dir, want, save_path=None):
    sfm, graph, occ = load_artefacts(prefix, out_dir, want["occupancy"])

    render_want = {
        "occupancy": want["occupancy"],
        "zones": want["zones"],
        "walls": want["walls"],
        "features": want["features"],
        "nodes": want["nodes"],
        "edges_combined": want["edges"],
    }
    fig = render_figure(prefix, sfm, graph, occ, render_want)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved figure -> {save_path}")
    else:
        print("\nOpening interactive window (close it to exit)...")
        plt.show()
    plt.close(fig)


# The five standard stages, in order: each dict is the `want` layer selection
# for render_figure, and each tuple also carries the output filename suffix
# and a human title suffix.
STAGES = [
    (
        "1_zones_features",
        "Zones + features (no nodes/edges)",
        {"zones": True, "walls": True, "features": True},
    ),
    (
        "2_zones_features_nodes",
        "Zones + features + nodes (no edges)",
        {"zones": True, "walls": True, "features": True, "nodes": True},
    ),
    (
        "3_edges_shorelinable",
        "Zones + features + nodes + edges (shore-linable)",
        {"zones": True, "walls": True, "features": True, "nodes": True,
         "edges_shorelinable": True},
    ),
    (
        "4_edges_width_safety",
        "Zones + features + nodes + edges (width & safety)",
        {"zones": True, "walls": True, "features": True, "nodes": True,
         "edges_width_safety": True},
    ),
    (
        "5_all_edges",
        "Zones + features + nodes + edges (shore-linable + width & safety)",
        {"zones": True, "walls": True, "features": True, "nodes": True,
         "edges_shorelinable": True, "edges_width_safety": True},
    ),
]


def render_stages(prefix, out_dir, save_dir, occupancy=False):
    """Render the 5 standard progression images and save them into save_dir."""
    sfm, graph, occ = load_artefacts(prefix, out_dir, occupancy)
    save_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for suffix, title_suffix, want in STAGES:
        render_want = dict(want)
        render_want["occupancy"] = occupancy
        fig = render_figure(prefix, sfm, graph, occ, render_want,
                             title_suffix=title_suffix)
        out_path = save_dir / f"{prefix}_stage{suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {out_path}")
        saved.append(out_path)

    return saved


def main():
    default_dir = Path(__file__).resolve().parent.parent / "data" / "outputs"

    parser = argparse.ArgumentParser(
        description="Visualize Map Extraction JSON outputs by file prefix.")
    parser.add_argument("prefix", nargs="?",
                        help="File prefix, e.g. floor_4_sankha_spaces")
    parser.add_argument("--dir", default=str(default_dir),
                        help=f"Output directory (default: {default_dir})")
    parser.add_argument("--occupancy", action="store_true",
                        help="Draw the occupancy raster underlay")
    parser.add_argument("--no-zones",    action="store_true", help="Hide zones")
    parser.add_argument("--no-walls",    action="store_true", help="Hide walls")
    parser.add_argument("--no-features", action="store_true", help="Hide features")
    parser.add_argument("--no-edges",    action="store_true", help="Hide graph edges")
    parser.add_argument("--no-nodes",    action="store_true", help="Hide graph nodes")
    parser.add_argument("--save", default=None,
                        help="Save to this PNG path instead of showing a window")
    parser.add_argument("--stages", action="store_true",
                        help="Render the 5 standard progressive images "
                             "(zones/features -> +nodes -> +shore-linable "
                             "edges -> +width&safety edges -> +both) and "
                             "save them into the visualizer/ folder")
    parser.add_argument("--list", action="store_true",
                        help="List available prefixes in the output dir and exit")
    args = parser.parse_args()

    out_dir = Path(args.dir)
    if not out_dir.exists():
        sys.exit(f"Output directory not found: {out_dir}")

    if args.list:
        prefixes = list_prefixes(out_dir)
        if not prefixes:
            print(f"No graph/sfm outputs found in {out_dir}")
        else:
            print(f"Available prefixes in {out_dir}:")
            for p in prefixes:
                print(f"  {p}")
        return

    if not args.prefix:
        parser.error("a prefix is required (or use --list). "
                     "e.g. python visualizer/map_visualizer.py floor_4_sankha_spaces")

    if args.stages:
        visualizer_dir = Path(__file__).resolve().parent
        print(f"Rendering 5-stage progression for prefix '{args.prefix}' "
              f"from {out_dir}")
        render_stages(args.prefix, out_dir, visualizer_dir,
                      occupancy=args.occupancy)
        return

    want = {
        "occupancy": args.occupancy,
        "zones":     not args.no_zones,
        "walls":     not args.no_walls,
        "features":  not args.no_features,
        "edges":     not args.no_edges,
        "nodes":     not args.no_nodes,
    }

    print(f"Visualizing prefix '{args.prefix}' from {out_dir}")
    visualize(args.prefix, out_dir, want, save_path=args.save)


if __name__ == "__main__":
    main()
