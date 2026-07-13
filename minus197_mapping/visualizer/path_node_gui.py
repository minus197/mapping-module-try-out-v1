"""
path_node_gui.py
----------------
Interactive inspector for the PATH-NODE layer produced by
map_extraction/path_nodes.py (<prefix>_path_nodes.json).

Like map_visualizer.py / path_node_visualizer.py, this tool is completely
independent of the main pipeline — it imports nothing from map_extraction /
pathfinding / shared and only reads the JSON artefacts already written to
data/outputs/. Running or editing it cannot affect the production pipeline.

What it does
------------
Draws the same base map as path_node_visualizer.py (zones, walls, features and
the navigation-graph nodes) and overlays the generated path nodes in BLUE.
Then it makes the path nodes *clickable*:

    Click a path node  ->  the node is highlighted, its search circle
                           (radius 2 * nearest_wall_dist_m) is drawn, the
                           zones / doors / landmarks whose boundary falls
                           inside that circle are highlighted, and a side
                           panel lists them with:
                               node_id, kind, perpendicular distance to the
                               boundary, and distance to the centroid.

The search-circle hits themselves come straight from <prefix>_path_nodes.json
(each path node carries a "search_circle" block written by PathNodeBuilder).
This tool only *joins* each hit's node_id to a human-readable label using the
graph / sfm artefacts and renders the result — it does no geometry of its own.

Given a file *prefix* (e.g. "itfac_floor_4") it locates and overlays:

    <prefix>_sfm.json          walls, zones, features        (base layer)
    <prefix>_graph.json        navigation nodes (+ edges)     (base layer)
    <prefix>_path_nodes.json   path nodes + search circles    (clickable)
    <prefix>_occupancy.json    five-state raster underlay     (optional)

Every layer is drawn only if its file exists; missing files are reported and
skipped rather than treated as errors. If no prefix is given on the command
line, the tool prints the available prefixes and prompts for one.

Usage
-----
    # Prompt for a prefix, then open the interactive window
    python visualizer/path_node_gui.py

    # Give the prefix directly
    python visualizer/path_node_gui.py itfac_floor_4

    # Point at a different output directory / add the occupancy underlay
    python visualizer/path_node_gui.py itfac_floor_4 --dir data/outputs --occupancy

    # List available prefixes (those that have a _path_nodes.json) and exit
    python visualizer/path_node_gui.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Circle, Patch
    from matplotlib.lines import Line2D
except ImportError as exc:  # pragma: no cover - dependency guard
    sys.exit(
        f"Missing dependency: {exc.name}. "
        f"Install with:  pip install matplotlib\n"
        f"(numpy is already a project dependency.)"
    )


# ── Appearance ────────────────────────────────────────────────────────────────
# Base-map colours mirror path_node_visualizer.py / map_visualizer.py so the
# tools look consistent.

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

GRAPH_NODE_COLOR = "#9e9e9e"       # ordinary nav-graph nodes, muted
PATH_NODE_COLOR  = "#1565c0"       # path nodes, strong blue
PATH_NODE_SEL    = "#ff6d00"       # the selected path node, orange
CIRCLE_COLOR     = "#ff6d00"       # search-circle outline
HIT_COLOR        = "#d81b60"       # highlighted hit targets, magenta

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


# ── Target lookup (join hit node_ids -> label / kind / position) ──────────────

def build_target_index(sfm, graph):
    """
    Map each target node_id (ZONE-<id> / FEAT-<id>) referenced by a search-circle
    hit to a human-readable descriptor. Prefer the graph node (it already carries
    a friendly label), fall back to the sfm zone/feature, so the panel still
    reads well even if the graph file is missing.

    Returns { node_id: {"label", "kind", "position"} }.
    """
    index = {}

    # Graph nodes give us clean labels + a definitive node_type.
    for n in (graph or {}).get("nodes", []):
        nid = n.get("node_id")
        if not nid:
            continue
        index[nid] = {
            "label":    n.get("label") or nid,
            "kind":     n.get("node_type", ""),
            "position": n.get("position"),
        }

    # Fill gaps from the sfm (zones -> ZONE-<zone_id>, features -> FEAT-<id>).
    for z in (sfm or {}).get("zones", []):
        nid = f"ZONE-{z.get('zone_id')}"
        if nid not in index:
            index[nid] = {
                "label":    z.get("admin_name") or z.get("name")
                            or z.get("category", "zone"),
                "kind":     z.get("category", "zone"),
                "position": z.get("centroid"),
            }
    for f in (sfm or {}).get("features", []):
        nid = f"FEAT-{f.get('feature_id')}"
        if nid not in index:
            index[nid] = {
                "label":    f.get("name") or f.get("feature_type", "feature"),
                "kind":     f.get("feature_type", "feature"),
                "position": f.get("position"),
            }
    return index


# ── Static layer renderers ────────────────────────────────────────────────────

def draw_occupancy(ax, occ):
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
    for n in graph.get("nodes", []):
        pos = n.get("position")
        if not pos:
            continue
        ntype = n.get("node_type", "")
        size = 14 if ntype == "junction" else 22
        ax.scatter(pos[0], pos[1], c=GRAPH_NODE_COLOR, s=size,
                   edgecolors="#555555", linewidths=0.3, zorder=5, alpha=0.8)


# ── Interactive controller ────────────────────────────────────────────────────

class PathNodeInspector:
    """
    Owns the figure and the click interaction. The path nodes are drawn as a
    single pickable scatter; clicking one selects it, draws its search circle,
    highlights the hit targets and updates the side panel.
    """

    def __init__(self, prefix, sfm, graph, path_nodes, occ, want):
        self.prefix = prefix
        self.sfm = sfm
        self.graph = graph
        self.nodes = path_nodes.get("path_nodes", [])
        self.meta = path_nodes.get("meta", {})
        self.targets = build_target_index(sfm, graph)

        # Transient artists for the current selection (cleared on each click).
        self._overlay = []
        self._selected = None

        # Two side-by-side axes: the map (left) and a text panel (right).
        self.fig, (self.ax, self.panel) = plt.subplots(
            1, 2, figsize=(18, 9),
            gridspec_kw={"width_ratios": [4, 1.35]},
        )
        self._draw_base(occ, want)
        self._draw_path_nodes()
        self._init_panel()

        self.fig.canvas.mpl_connect("pick_event", self._on_pick)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # -- base map -----------------------------------------------------------

    def _draw_base(self, occ, want):
        ax = self.ax
        if occ is not None and want.get("occupancy"):
            draw_occupancy(ax, occ)
        if self.sfm is not None:
            if want.get("zones"):
                draw_zones(ax, self.sfm)
            if want.get("walls"):
                draw_walls(ax, self.sfm)
            if want.get("features"):
                draw_features(ax, self.sfm)
        if self.graph is not None:
            if want.get("edges"):
                draw_edges(ax, self.graph)
            if want.get("graph_nodes"):
                draw_graph_nodes(ax, self.graph)

        n_zones = len(self.sfm.get("zones", [])) if self.sfm else 0
        n_walls = len(self.sfm.get("walls", [])) if self.sfm else 0
        n_gnodes = len(self.graph.get("nodes", [])) if self.graph else 0
        ax.set_title(
            f"{self.prefix}   |   zones={n_zones}  walls={n_walls}  "
            f"graph_nodes={n_gnodes}  path_nodes={len(self.nodes)}\n"
            f"Click a blue path node to see what falls in its search circle "
            f"(press 'c' to clear)",
            fontsize=10,
        )
        ax.set_xlabel("X (metres, IFC project frame)")
        ax.set_ylabel("Y (metres, IFC project frame)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
        self._build_legend()

    def _draw_path_nodes(self):
        """Path nodes as one pickable scatter; picker radius in points."""
        pts = [pn.get("position") for pn in self.nodes]
        self._xy = np.array([[p[0], p[1]] for p in pts], dtype=float) \
            if pts else np.empty((0, 2))
        if len(self._xy):
            self._scatter = self.ax.scatter(
                self._xy[:, 0], self._xy[:, 1],
                c=PATH_NODE_COLOR, s=42, edgecolors="white", linewidths=0.6,
                zorder=8, marker="o", picker=6,
            )
        else:
            self._scatter = None

    def _build_legend(self):
        handles = [
            Patch(facecolor=ZONE_FILL["unknown"], edgecolor="#888", alpha=0.6,
                  label="zone (room/shop)"),
            Line2D([0], [0], color="#1565c0", lw=1.8, label="wall (shore-linable)"),
            Line2D([0], [0], color="#555555", lw=1.5, label="wall"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=GRAPH_NODE_COLOR,
                   markersize=7, markeredgecolor="#555", label="graph node"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=PATH_NODE_COLOR,
                   markersize=9, markeredgecolor="white", label="path node"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=PATH_NODE_SEL,
                   markersize=10, markeredgecolor="black", label="selected node"),
            Line2D([0], [0], color=CIRCLE_COLOR, lw=1.6, label="search circle (2x)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=HIT_COLOR,
                   markersize=9, markeredgecolor="black", label="hit in circle"),
        ]
        self.ax.legend(handles=handles, loc="upper left",
                       bbox_to_anchor=(0.0, -0.06), ncol=4,
                       fontsize=7, framealpha=0.9, borderaxespad=0.0)

    # -- side panel ---------------------------------------------------------

    def _init_panel(self):
        self.panel.axis("off")
        self._panel_text = self.panel.text(
            0.0, 1.0, "", transform=self.panel.transAxes,
            fontsize=8.5, va="top", ha="left", family="monospace",
        )
        self._set_panel_intro()

    def _set_panel_intro(self):
        gap = self.meta.get("gap_to_wall_m")
        gap_txt = f"{gap:g} m" if isinstance(gap, (int, float)) else "n/a"
        self._panel_text.set_text(
            "PATH NODE INSPECTOR\n"
            "===================\n\n"
            f"path nodes : {len(self.nodes)}\n"
            f"gap to wall: {gap_txt}\n\n"
            "Click a blue path node on the\n"
            "left to list the rooms / doors /\n"
            "landmarks whose boundary falls\n"
            "inside its search circle.\n\n"
            "The search circle radius is\n"
            "2 * nearest_wall_dist_m (x),\n"
            "the node's true distance to the\n"
            "closest wall.\n\n"
            "Keys:  c = clear selection\n"
        )
        self.fig.canvas.draw_idle()

    def _update_panel(self, node):
        pos = node.get("position", [0, 0])
        x = node.get("nearest_wall_dist_m", 0.0)
        circ = node.get("search_circle", {}) or {}
        radius = circ.get("radius_m", 2.0 * x)
        hits = circ.get("hits", [])

        lines = [
            f"SELECTED  {node.get('node_id', '?')}",
            "=" * 26,
            f"position    : ({pos[0]:.2f}, {pos[1]:.2f})",
            f"wall_id     : {str(node.get('wall_id', ''))[:14]}",
            f"gap_m       : {node.get('gap_m', 0.0):.3f}",
            f"x (nearest  : {x:.3f}",
            f"   wall dist)",
            f"circle R=2x : {radius:.3f}",
            "",
            f"IN SEARCH CIRCLE ({len(hits)})",
            "-" * 26,
        ]
        if not hits:
            lines.append("(nothing within 2x)")
        else:
            # hits already sorted by perp_dist_m in the JSON, but re-sort to be safe
            for h in sorted(hits, key=lambda h: h.get("perp_dist_m", 0.0)):
                nid = h.get("node_id", "?")
                tgt = self.targets.get(nid, {})
                label = str(tgt.get("label") or nid)
                kind = h.get("kind") or tgt.get("kind", "")
                lines.append(f"* {label[:22]}")
                lines.append(f"    id   {nid[:20]}")
                lines.append(f"    kind {kind}")
                lines.append(f"    perp {h.get('perp_dist_m', 0.0):6.2f} m"
                             f"  cen {h.get('centroid_dist_m', 0.0):6.2f} m")
        self._panel_text.set_text("\n".join(lines))

    # -- selection overlay --------------------------------------------------

    def _clear_overlay(self):
        for art in self._overlay:
            art.remove()
        self._overlay = []
        self._selected = None

    def _select(self, idx):
        self._clear_overlay()
        node = self.nodes[idx]
        self._selected = idx
        px, py = node["position"]
        x = node.get("nearest_wall_dist_m", 0.0)
        circ = node.get("search_circle", {}) or {}
        radius = circ.get("radius_m", 2.0 * x)

        # search circle
        c = Circle((px, py), radius, fill=False, edgecolor=CIRCLE_COLOR,
                   linewidth=1.8, linestyle="-", zorder=9, alpha=0.9)
        self.ax.add_patch(c)
        self._overlay.append(c)

        # selected node marker
        sel = self.ax.scatter([px], [py], c=PATH_NODE_SEL, s=130,
                              edgecolors="black", linewidths=1.0,
                              zorder=11, marker="o")
        self._overlay.append(sel)

        # highlight each hit target + a leader line to it
        for h in circ.get("hits", []):
            tgt = self.targets.get(h.get("node_id"))
            if not tgt or not tgt.get("position"):
                continue
            tx, ty = tgt["position"]
            hit = self.ax.scatter([tx], [ty], c=HIT_COLOR, s=120,
                                  edgecolors="black", linewidths=0.8,
                                  zorder=10, marker="o")
            leader, = self.ax.plot([px, tx], [py, ty], color=HIT_COLOR,
                                   linewidth=1.0, linestyle="--",
                                   alpha=0.7, zorder=9)
            self._overlay.extend([hit, leader])

        self._update_panel(node)
        self.fig.canvas.draw_idle()

    # -- events -------------------------------------------------------------

    def _on_pick(self, event):
        if self._scatter is None or event.artist is not self._scatter:
            return
        if not len(event.ind):
            return
        # If several nodes are under the cursor, take the nearest to the click.
        if event.mouseevent.xdata is None:
            idx = int(event.ind[0])
        else:
            mx, my = event.mouseevent.xdata, event.mouseevent.ydata
            cand = np.array(event.ind, dtype=int)
            d = np.hypot(self._xy[cand, 0] - mx, self._xy[cand, 1] - my)
            idx = int(cand[int(np.argmin(d))])
        self._select(idx)

    def _on_key(self, event):
        if event.key == "c":
            self._clear_overlay()
            self._set_panel_intro()
            self.fig.canvas.draw_idle()

    def show(self):
        self.fig.tight_layout()
        print("\nOpening interactive window (close it to exit)...")
        print("  - Click a blue path node to inspect its search circle.")
        print("  - Press 'c' to clear the current selection.")
        plt.show()


# ── Orchestration ──────────────────────────────────────────────────────────────

def load_artefacts(prefix, out_dir, want_occupancy):
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

    # Guard against an old _path_nodes.json produced before search circles existed.
    sample = (path_nodes.get("path_nodes") or [{}])[0]
    if "search_circle" not in sample:
        print("\n  WARNING: this _path_nodes.json has no 'search_circle' data.\n"
              "           Re-run the map-extraction pipeline to regenerate it,\n"
              "           otherwise clicking a node will show an empty list.")
    return sfm, graph, path_nodes, occ


def prompt_for_prefix(out_dir):
    """Interactive prefix picker used when none is supplied on the CLI."""
    prefixes = list_prefixes(out_dir)
    if not prefixes:
        sys.exit(f"No *_path_nodes.json outputs found in {out_dir}. "
                 f"Run the map-extraction pipeline first.")
    print(f"\nAvailable path-node prefixes in {out_dir}:")
    for i, p in enumerate(prefixes):
        print(f"  [{i}] {p}")
    print("\nEnter the file prefix (the shared part of the four output file")
    print("names, e.g. 'itfac_floor_4'), or its index number:")
    try:
        raw = input("prefix> ").strip()
    except (EOFError, KeyboardInterrupt):
        sys.exit("\nCancelled.")
    if not raw:
        sys.exit("No prefix entered.")
    if raw.isdigit() and int(raw) < len(prefixes):
        return prefixes[int(raw)]
    return raw


def main():
    default_dir = Path(__file__).resolve().parent.parent / "data" / "outputs"

    parser = argparse.ArgumentParser(
        description="Interactive GUI: click a path node to see the rooms / "
                    "doors / landmarks inside its search circle.")
    parser.add_argument("prefix", nargs="?",
                        help="File prefix shared by the 4 output files, "
                             "e.g. itfac_floor_4. Omit to be prompted.")
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

    prefix = args.prefix or prompt_for_prefix(out_dir)

    want = {
        "occupancy":   args.occupancy,
        "zones":       not args.no_zones,
        "walls":       not args.no_walls,
        "features":    not args.no_features,
        "edges":       not args.no_edges,
        "graph_nodes": not args.no_graph_nodes,
    }

    print(f"\nLoading path-node artefacts for prefix '{prefix}' from {out_dir}")
    sfm, graph, path_nodes, occ = load_artefacts(prefix, out_dir, want["occupancy"])
    inspector = PathNodeInspector(prefix, sfm, graph, path_nodes, occ, want)
    inspector.show()


if __name__ == "__main__":
    main()
