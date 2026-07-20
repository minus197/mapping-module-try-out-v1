"""
path_node_visualizer.py
-----------------------
Standalone visual inspector for a Path Finding route that has been ADJUSTED to
hug the cane-trailing path-node layer (see pathfinding/path_node_adjuster.py).

It is a thin companion to path_visualizer.py: it reuses that module's static
layer renderers (zones / walls / features / graph nodes) and its route overlay,
and adds two things on top:

  1. the full PATH-NODE LAYER — every cane-trailing waypoint from
     <prefix>_path_nodes.json is drawn as a small hollow marker, so you can see
     the whole shore-lining layer the route was allowed to snap onto, not just
     the nodes the route actually used;

  2. route-aware node colouring — waypoints the adjusted route passes through
     are drawn in their own colour, and PATH-xxxx waypoints (the ones the
     adjuster spliced in) are marked distinctly from junction / door waypoints
     so it is obvious which legs are hugging the wall and which cut straight
     across (doorways / corridor crossings).

Like path_visualizer.py it imports nothing from map_extraction / pathfinding /
shared and only reads JSON artefacts on disk, so running or editing it cannot
affect either production pipeline.

Given a map *prefix* it locates, in the mapping output directory:

    <prefix>_sfm.json         walls, zones, features
    <prefix>_graph.json       navigation nodes (edges intentionally ignored)
    <prefix>_path_nodes.json  cane-trailing path-node layer
    <prefix>_occupancy.json   five-state raster underlay (optional layer)

and overlays the route from a pathfinding feedback/output JSON — ideally one
produced WITH the path-node adjustment, e.g.:

    python run_pathfinding.py --graph <prefix>_graph.json \
        --path-nodes <prefix>_path_nodes.json --sfm <prefix>_sfm.json \
        --from-name SINGER --to-name POPEYES \
        --save-feedback data/outputs/route.json

Usage
-----
    # Auto-resolve map artefacts from a prefix; overlay one adjusted route file
    python visualizer/path_node_visualizer.py \
        itfac_mall_floor_with_path_nodes \
        --path data/outputs/route.json

    # Point at a different mapping output directory
    python visualizer/path_node_visualizer.py <prefix> --path <out.json> \
        --map-dir ../minus197_mapping/data/outputs

    # Hide the un-used (background) path nodes, keeping only the route's own
    python visualizer/path_node_visualizer.py <prefix> --path <out.json> \
        --no-path-node-layer

    # Save straight to PNG instead of opening a window
    python visualizer/path_node_visualizer.py <prefix> --path <out.json> \
        --save route.png

    # List available map prefixes in the mapping output dir and exit
    python visualizer/path_node_visualizer.py --list

Inter-floor routes
------------------
A route produced by the --building-graph mode (pathfinding/multi_floor.py) spans
two floors and contains a `take_elevator` action at the floor change. Because the
two floors have different geometry, they cannot share one image: the route is
split at the elevator and each floor is rendered SEPARATELY, against its own
`{stem}_{floor_label}_*` artefacts.

    python visualizer/path_node_visualizer.py floors_3-4_combined \
        --path data/outputs/route_L3_to_L4.json --save route.png
    # -> route_L3.png  (start -> elevator)
    # -> route_L4.png  (elevator -> destination)

The floor label is appended to the --save filename automatically, and the base
prefix may be given either bare (`floors_3-4_combined`) or already floor-specific.
Without --save, each floor opens in its own window in turn.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Reuse the sibling visualizer's loaders / renderers verbatim so the two tools
# stay pixel-consistent and there is a single source of truth for the static
# layers. This file only adds the path-node layer + route-aware colouring.
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError as exc:  # pragma: no cover - dependency guard
    sys.exit(
        f"Missing dependency: {exc.name}. "
        f"Install with:  pip install matplotlib\n"
        f"(numpy is already a project dependency.)"
    )

import path_visualizer as pv


# ── Path-node layer appearance ───────────────────────────────────────────────

# Every cane-trailing path node in the layer (background, hollow).
PATH_NODE_LAYER_COLOR = "#00838f"   # teal
# A PATH-xxxx waypoint the adjusted route actually walks through.
ROUTE_PATH_NODE_COLOR = "#006064"   # darker teal, filled
# A non-path-node waypoint on the route (junction / door / zone centroid).
ROUTE_GRAPH_NODE_COLOR = pv.PATH_NODE_COLOR   # reuse the existing pink


def _is_path_node(step) -> bool:
    """True when a route step sits on a spliced-in cane-trailing path node."""
    nid = step.get("node_id") or ""
    return nid.startswith("PATH-")


# ── Inter-floor route splitting ──────────────────────────────────────────────

def split_route_by_floor(path_data):
    """Split a route into one leg per floor.

    An inter-floor route (see pathfinding/multi_floor.py) is a flat action list
    with a single ``take_elevator`` action marking the floor change: everything
    before it walks the start floor, everything after walks the destination
    floor. The elevator action itself carries both floor labels and both XY
    positions, so it is appended to the outgoing leg (as the point the user
    walks to) and prepended to the incoming leg (as the point they exit at).

    Returns a list of ``(floor_label, actions)`` pairs — a single unlabelled
    (None) pair for an ordinary same-floor route, so callers can treat both
    shapes uniformly.
    """
    idx = [i for i, a in enumerate(path_data)
           if a.get("action") == "take_elevator"]
    if not idx:
        return [(None, path_data)]

    legs = []
    prev_cut = 0
    for i in idx:
        elev = path_data[i]
        # Outgoing leg ends AT the elevator on the floor being left.
        before = path_data[prev_cut:i] + [{
            "action": "stop",
            "landmark": f"elevator ({elev['direction']} to {elev['to_floor']})",
            "node_id": elev["node_id"],
            "position": elev["position"],
        }]
        legs.append((elev["from_floor"], before))
        # Incoming leg starts AT the elevator on the floor being entered.
        prev_cut = i + 1
        pending_entry = {
            "action": "continue",
            "node_id": elev["arrive_node_id"],
            "position": elev["arrive_position"],
        }
    legs.append((path_data[idx[-1]]["to_floor"],
                 [pending_entry] + path_data[prev_cut:]))
    return legs


def _floor_prefix(prefix: str, floor_label: str, map_dir: Path) -> str:
    """Per-floor artefact prefix for a floor label.

    Inter-floor artefacts follow the ``{ifc_stem}_{floor_label}_*`` convention
    written by the mapping module. The prefix may be given bare
    (``floors_3-4_combined``) or already pointing at one floor
    (``floors_3-4_combined_L3``); in the latter case the existing floor suffix
    is stripped first, so every leg resolves to ITS OWN floor rather than
    inheriting the one named on the command line.
    """
    if prefix.endswith(f"_{floor_label}"):
        return prefix

    base = prefix
    # Strip a trailing floor suffix belonging to a DIFFERENT floor.
    tail = prefix.rsplit("_", 1)[-1]
    if tail != prefix and (map_dir / f"{prefix.rsplit('_', 1)[0]}_{floor_label}_sfm.json").exists():
        base = prefix.rsplit("_", 1)[0]

    candidate = f"{base}_{floor_label}"
    if (map_dir / f"{candidate}_sfm.json").exists() or \
       (map_dir / f"{candidate}_graph.json").exists():
        return candidate
    return prefix


def _floor_save_path(save_path, floor_label: str) -> Path:
    """Insert the floor label before the suffix: route.png -> route_L3.png."""
    p = Path(save_path)
    return p.with_name(f"{p.stem}_{floor_label}{p.suffix or '.png'}")


# ── Path-node layer renderer ─────────────────────────────────────────────────

def draw_path_node_layer(ax, path_nodes_data):
    """Draw every path node in <prefix>_path_nodes.json as a small hollow
    marker — the full shore-lining layer, drawn under the route so the route
    reads clearly on top."""
    if not path_nodes_data:
        return 0
    nodes = path_nodes_data.get("path_nodes", [])
    n_drawn = 0
    for n in nodes:
        pos = n.get("position")
        if not pos:
            continue
        ax.scatter(pos[0], pos[1], marker="o", s=26,
                   facecolors="none", edgecolors=PATH_NODE_LAYER_COLOR,
                   linewidths=1.0, zorder=4, alpha=0.7)
        n_drawn += 1
    return n_drawn


def draw_route_waypoints(ax, steps):
    """Redraw the route's intermediate waypoints on top of pv.draw_path(),
    colour-coded so spliced-in path nodes (teal) are distinct from graph
    waypoints — junction/door/etc. (pink). Start/goal keep their own markers
    from pv.draw_path(), so we only touch the interior steps."""
    for s in steps[1:-1]:
        if _is_path_node(s):
            ax.scatter(*s["pos"], c=ROUTE_PATH_NODE_COLOR, s=52,
                       marker="D", edgecolors="white", linewidths=0.8,
                       zorder=12)
        else:
            ax.scatter(*s["pos"], c=ROUTE_GRAPH_NODE_COLOR, s=48,
                       edgecolors="white", linewidths=0.8, zorder=12)


# ── Legend (extends pv.build_legend with the path-node entries) ──────────────

def build_legend(ax, layers_drawn):
    pv.build_legend(ax, layers_drawn)
    # pv.build_legend already placed a legend; append our extra handles by
    # rebuilding it with the union of both sets.
    handles = []
    if "zones" in layers_drawn:
        from matplotlib.patches import Patch
        for cat in ("corridor", "shop", "food_court", "unknown"):
            handles.append(Patch(facecolor=pv.ZONE_FILL[cat], edgecolor="#888",
                                  alpha=0.6, label=f"zone: {cat}"))
    if "walls" in layers_drawn:
        handles.append(Line2D([0], [0], color="#555555", lw=1.5, label="wall"))
        handles.append(Line2D([0], [0], color="#1565c0", lw=1.8,
                              label="wall (shore-linable)"))
    if "nodes" in layers_drawn:
        for ntype, color in (("junction", pv.NODE_COLOR["junction"]),
                             ("zone_centroid", pv.NODE_COLOR["zone_centroid"]),
                             ("door", pv.NODE_COLOR["door"])):
            handles.append(Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=color, markersize=7,
                                  markeredgecolor="black", label=f"node: {ntype}"))
    if "path_node_layer" in layers_drawn:
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="none",
                              markeredgecolor=PATH_NODE_LAYER_COLOR,
                              markersize=7, label="path node (layer)"))
    if "path" in layers_drawn:
        handles.append(Line2D([0], [0], color=pv.PATH_LINE_COLOR, lw=3.0,
                              label="generated path"))
        handles.append(Line2D([0], [0], marker="D", color="w",
                              markerfacecolor=ROUTE_PATH_NODE_COLOR,
                              markersize=8, markeredgecolor="white",
                              label="route via path node"))
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=ROUTE_GRAPH_NODE_COLOR,
                              markersize=8, markeredgecolor="white",
                              label="route via graph node"))
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=pv.START_COLOR, markersize=9,
                              markeredgecolor="white", label="start"))
        handles.append(Line2D([0], [0], marker="*", color="w",
                              markerfacecolor=pv.END_COLOR, markersize=13,
                              markeredgecolor="white", label="goal"))
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  fontsize=7, framealpha=0.9, borderaxespad=0.0)


# ── Orchestration ────────────────────────────────────────────────────────────

def render_figure(prefix, sfm, graph, occ, path_nodes_data, steps, want):
    fig, ax = plt.subplots(figsize=(15, 9))
    layers_drawn = set()

    if occ is not None and want.get("occupancy"):
        pv.draw_occupancy(ax, occ)
        layers_drawn.add("occupancy")

    if sfm is not None:
        if want.get("zones"):
            pv.draw_zones(ax, sfm); layers_drawn.add("zones")
        if want.get("walls"):
            pv.draw_walls(ax, sfm); layers_drawn.add("walls")
        if want.get("features"):
            pv.draw_features(ax, sfm); layers_drawn.add("features")

    if graph is not None and want.get("nodes"):
        pv.draw_nodes(ax, graph); layers_drawn.add("nodes")

    n_path_nodes = 0
    if path_nodes_data is not None and want.get("path_node_layer"):
        n_path_nodes = draw_path_node_layer(ax, path_nodes_data)
        layers_drawn.add("path_node_layer")

    pv.draw_path(ax, steps)
    draw_route_waypoints(ax, steps)   # recolour interior waypoints by type
    layers_drawn.add("path")

    n_zones = len(sfm.get("zones", [])) if sfm else 0
    n_walls = len(sfm.get("walls", [])) if sfm else 0
    n_nodes = len(graph.get("nodes", [])) if graph else 0
    n_route_pn = sum(1 for s in steps if _is_path_node(s))

    start_id = steps[0]["node_id"] if steps else "?"
    goal_id = steps[-1]["node_id"] if steps else "?"
    title = (f"{prefix}   |   route: {start_id} -> {goal_id}   |   "
             f"steps={len(steps)} (path-node hops={n_route_pn})   "
             f"zones={n_zones} walls={n_walls} nodes={n_nodes} "
             f"path_nodes={n_path_nodes}")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X (metres, IFC project frame)")
    ax.set_ylabel("Y (metres, IFC project frame)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

    build_legend(ax, layers_drawn)
    fig.tight_layout()
    return fig


def _visualize_leg(prefix, map_dir, path_data, want, floor_label, save_path):
    """Render one already-split leg against that floor's own map artefacts.

    Mirrors pv.load_artefacts() for the static layers, but takes the route as an
    in-memory action list (one floor's leg) rather than re-reading it from disk.
    """
    paths = pv.resolve_map_paths(prefix, map_dir)
    sfm = pv._load(paths["sfm"])
    graph = pv._load(paths["graph"])
    occ = pv._load(paths["occupancy"]) if want["occupancy"] else None

    if sfm is None and graph is None:
        sys.exit(f"No _sfm.json or _graph.json found for prefix '{prefix}' "
                 f"in {map_dir}. Try --list to see available prefixes.")

    for key, data in (("sfm", sfm), ("graph", graph)):
        state = "loaded" if data is not None else "MISSING"
        print(f"  {paths[key].name:<45} {state}")
    if want["occupancy"]:
        state = "loaded" if occ is not None else "MISSING"
        print(f"  {paths['occupancy'].name:<45} {state}")

    # Path-node layer is our own extra artefact (not loaded by pv.load_artefacts).
    path_nodes_path = map_dir / f"{prefix}_path_nodes.json"
    path_nodes_data = pv._load(path_nodes_path)
    state = "loaded" if path_nodes_data is not None else "MISSING"
    n = len(path_nodes_data.get("path_nodes", [])) if path_nodes_data else 0
    print(f"  {path_nodes_path.name:<45} {state}"
          + (f" ({n} path nodes)" if path_nodes_data else ""))

    steps = pv.extract_path_points(path_data, graph)

    title_prefix = f"{prefix}  [{floor_label}]" if floor_label else prefix
    fig = render_figure(title_prefix, sfm, graph, occ, path_nodes_data, steps, want)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved figure -> {save_path}")
    else:
        print("\nOpening interactive window (close it to exit)...")
        plt.show()
    plt.close(fig)


def visualize(prefix, map_dir, path_file, want, save_path=None):
    path_data = pv._load(Path(path_file))
    if path_data is None:
        sys.exit(f"Path output file not found or unreadable: {path_file}")
    if not isinstance(path_data, list):
        sys.exit(f"Unexpected path output format in {path_file} "
                 f"(expected a JSON list of steps).")

    legs = split_route_by_floor(path_data)

    if len(legs) > 1 and not save_path:
        print(f"\nNote: this is an inter-floor route ({len(legs)} floors); "
              f"each floor opens in its own window, close one to see the next.")

    for floor_label, leg_actions in legs:
        if floor_label:
            floor_prefix = _floor_prefix(prefix, floor_label, map_dir)
            leg_save = _floor_save_path(save_path, floor_label) if save_path else None
            print(f"\n--- {floor_label}: {len(leg_actions)} actions "
                  f"over prefix '{floor_prefix}' ---")
        else:
            floor_prefix, leg_save = prefix, save_path

        _visualize_leg(floor_prefix, map_dir, leg_actions, want,
                       floor_label, leg_save)


def main():
    default_map_dir = (Path(__file__).resolve().parent.parent.parent
                       / "minus197_mapping" / "data" / "outputs")

    parser = argparse.ArgumentParser(
        description="Visualize a path-node-adjusted route over its floor map, "
                    "including the full cane-trailing path-node layer.")
    parser.add_argument("prefix", nargs="?",
                        help="Map file prefix, e.g. itfac_mall_floor_with_path_nodes")
    parser.add_argument("--path", default=None,
                        help="Route output JSON written by run_pathfinding.py "
                             "--save-feedback (ideally an adjusted route)")
    parser.add_argument("--map-dir", default=str(default_map_dir),
                        help=f"Mapping output directory (default: {default_map_dir})")
    parser.add_argument("--occupancy", action="store_true",
                        help="Draw the occupancy raster underlay")
    parser.add_argument("--no-zones",    action="store_true", help="Hide zones")
    parser.add_argument("--no-walls",    action="store_true", help="Hide walls")
    parser.add_argument("--no-features", action="store_true", help="Hide features")
    parser.add_argument("--no-nodes",    action="store_true", help="Hide graph nodes")
    parser.add_argument("--no-path-node-layer", action="store_true",
                        help="Hide the background cane-trailing path-node layer")
    parser.add_argument("--save", default=None,
                        help="Save to this PNG path instead of showing a window")
    parser.add_argument("--list", action="store_true",
                        help="List available map prefixes and exit")
    args = parser.parse_args()

    map_dir = Path(args.map_dir)
    if not map_dir.exists():
        sys.exit(f"Mapping output directory not found: {map_dir}")

    if args.list:
        prefixes = pv.list_prefixes(map_dir)
        if not prefixes:
            print(f"No graph/sfm outputs found in {map_dir}")
        else:
            print(f"Available map prefixes in {map_dir}:")
            for p in prefixes:
                print(f"  {p}")
        return

    if not args.prefix:
        parser.error("a map prefix is required (or use --list). e.g. "
                     "python visualizer/path_node_visualizer.py "
                     "itfac_mall_floor_with_path_nodes --path data/outputs/route.json")
    if not args.path:
        parser.error("--path is required: the pathfinding output JSON to overlay.")

    want = {
        "occupancy":       args.occupancy,
        "zones":           not args.no_zones,
        "walls":           not args.no_walls,
        "features":        not args.no_features,
        "nodes":           not args.no_nodes,
        "path_node_layer": not args.no_path_node_layer,
    }

    print(f"Visualizing path-node-adjusted route from {args.path}")
    print(f"  over map prefix '{args.prefix}' in {map_dir}")
    visualize(args.prefix, map_dir, args.path, want, save_path=args.save)


if __name__ == "__main__":
    main()
