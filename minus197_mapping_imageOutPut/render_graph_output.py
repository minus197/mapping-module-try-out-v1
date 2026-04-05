from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon


NODE_STYLE = {
    "zone_centroid": {
        "color": "#0f766e",
        "marker": "o",
        "size": 180,
        "edgecolor": "#083344",
        "zorder": 6,
    },
    "door": {
        "color": "#b45309",
        "marker": "s",
        "size": 120,
        "edgecolor": "#7c2d12",
        "zorder": 7,
    },
    "stair": {
        "color": "#7c3aed",
        "marker": "^",
        "size": 160,
        "edgecolor": "#4c1d95",
        "zorder": 7,
    },
    "elevator": {
        "color": "#2563eb",
        "marker": "D",
        "size": 150,
        "edgecolor": "#1e3a8a",
        "zorder": 7,
    },
    "escalator": {
        "color": "#db2777",
        "marker": "D",
        "size": 150,
        "edgecolor": "#831843",
        "zorder": 7,
    },
    "junction": {
        "color": "#111827",
        "marker": "X",
        "size": 90,
        "edgecolor": "#111827",
        "zorder": 8,
    },
    "landmark": {
        "color": "#ca8a04",
        "marker": "P",
        "size": 120,
        "edgecolor": "#713f12",
        "zorder": 7,
    },
}

ZONE_FILL = {
    "shop": "#dbeafe",
    "storage": "#dcfce7",
    "office": "#fce7f3",
    "restroom": "#fef3c7",
    "corridor": "#e5e7eb",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_default_paths(script_dir: Path) -> tuple[Path, Path, Path]:
    project_root = script_dir.parent / "minus197_mapping"
    outputs_dir = project_root / "data" / "outputs"
    stem = "20201022mapping IFC4 Convenience store"
    graph_path = outputs_dir / f"{stem}_graph.json"
    sfm_path = outputs_dir / f"{stem}_sfm.json"
    output_path = script_dir / f"{stem}_graph_visualization.png"
    return graph_path, sfm_path, output_path


def node_position(node: dict[str, Any]) -> tuple[float, float]:
    x, y = node["position"]
    return float(x), float(y)


def shorten_label(label: str, max_length: int = 28) -> str:
    cleaned = " ".join(str(label).split())
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[: max_length - 1].rstrip() + "…"


def draw_zones(ax, sfm: dict[str, Any] | None) -> None:
    if not sfm:
        return

    for zone in sfm.get("zones", []):
        polygon = zone.get("boundary_polygon") or []
        if len(polygon) < 3:
            continue

        category = zone.get("category", "")
        fill_color = ZONE_FILL.get(category, "#e2e8f0")
        patch = Polygon(
            polygon,
            closed=True,
            facecolor=fill_color,
            edgecolor="#94a3b8",
            linewidth=1.2,
            alpha=0.22,
            zorder=1,
        )
        ax.add_patch(patch)

        centroid = zone.get("centroid")
        if centroid:
            label = zone.get("long_name") or zone.get("name") or zone.get("zone_id")
            ax.text(
                centroid[0],
                centroid[1],
                shorten_label(label, 18),
                fontsize=7,
                color="#334155",
                ha="center",
                va="center",
                zorder=2,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.7,
                },
            )


def draw_edges(ax, nodes_by_id: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    for edge in edges:
        source = nodes_by_id.get(edge["source_id"])
        target = nodes_by_id.get(edge["target_id"])
        if not source or not target:
            continue

        x1, y1 = node_position(source)
        x2, y2 = node_position(target)
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="#475569",
            linewidth=0.8,
            alpha=0.35,
            zorder=3,
        )


def draw_nodes(ax, nodes: list[dict[str, Any]]) -> None:
    seen_types: set[str] = set()

    for node in nodes:
        node_type = node.get("node_type", "landmark")
        style = NODE_STYLE.get(node_type, NODE_STYLE["landmark"])
        x, y = node_position(node)
        seen_types.add(node_type)

        ax.scatter(
            [x],
            [y],
            s=style["size"],
            marker=style["marker"],
            c=style["color"],
            edgecolors=style["edgecolor"],
            linewidths=0.8,
            zorder=style["zorder"],
        )

        label = node.get("label") or node.get("node_id")
        if label:
            ax.text(
                x + 0.08,
                y + 0.08,
                shorten_label(label, 26),
                fontsize=6.8,
                color="#0f172a",
                ha="left",
                va="bottom",
                zorder=9,
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.62,
                },
            )

    handles = []
    for node_type in sorted(seen_types):
        style = NODE_STYLE.get(node_type, NODE_STYLE["landmark"])
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color="w",
                markerfacecolor=style["color"],
                markeredgecolor=style["edgecolor"],
                markersize=9,
                linewidth=0,
                label=node_type,
            )
        )

    if handles:
        ax.legend(
            handles=handles,
            loc="upper right",
            frameon=True,
            framealpha=0.92,
            facecolor="white",
            edgecolor="#cbd5e1",
            fontsize=8,
            title="Node types",
            title_fontsize=9,
        )


def add_bounds_from_nodes(ax, nodes: list[dict[str, Any]]) -> None:
    xs = [float(node["position"][0]) for node in nodes]
    ys = [float(node["position"][1]) for node in nodes]
    if not xs or not ys:
        return

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad_x = max((max_x - min_x) * 0.08, 1.0)
    pad_y = max((max_y - min_y) * 0.08, 1.0)
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)


def render_graph(graph_path: Path, sfm_path: Path | None, output_path: Path) -> None:
    graph = load_json(graph_path)
    sfm = load_json(sfm_path) if sfm_path and sfm_path.exists() else None

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    nodes_by_id = {node["node_id"]: node for node in nodes}

    fig, ax = plt.subplots(figsize=(16, 10), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fafc")

    draw_zones(ax, sfm)
    draw_edges(ax, nodes_by_id, edges)
    draw_nodes(ax, nodes)
    add_bounds_from_nodes(ax, nodes)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Minus197 Mapping Graph Visualization", fontsize=18, pad=18)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, color="#dbe4ee", linewidth=0.7, alpha=0.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_graph, default_sfm, default_output = resolve_default_paths(script_dir)

    parser = argparse.ArgumentParser(
        description="Render the graph output from MapExtractionPipeline as an image."
    )
    parser.add_argument("--graph", type=Path, default=default_graph, help="Path to *_graph.json")
    parser.add_argument("--sfm", type=Path, default=default_sfm, help="Path to *_sfm.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output image path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_graph(args.graph, args.sfm, args.output)
    print(f"Saved image: {args.output}")


if __name__ == "__main__":
    main()
