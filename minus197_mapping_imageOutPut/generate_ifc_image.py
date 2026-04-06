from __future__ import annotations

import argparse
from pathlib import Path

import ifcopenshell
import ifcopenshell.geom
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection


def collect_projected_triangles(ifc_path: Path) -> np.ndarray:
    """Collect all mesh triangles and project them onto the XY plane."""
    model = ifcopenshell.open(str(ifc_path))

    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    polygons_xy: list[np.ndarray] = []

    for product in model.by_type("IfcProduct"):
        if not getattr(product, "Representation", None):
            continue

        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
            verts = np.asarray(shape.geometry.verts, dtype=float).reshape(-1, 3)
            faces = np.asarray(shape.geometry.faces, dtype=np.int32).reshape(-1, 3)
        except Exception:
            continue

        if verts.size == 0 or faces.size == 0:
            continue

        tris_xy = verts[faces][:, :, :2]
        polygons_xy.extend(tris_xy)

    if not polygons_xy:
        raise RuntimeError("No IFC geometry could be triangulated for rendering.")

    return np.asarray(polygons_xy, dtype=float)


def render_top_view(polygons_xy: np.ndarray, output_path: Path) -> None:
    mins = polygons_xy.reshape(-1, 2).min(axis=0)
    maxs = polygons_xy.reshape(-1, 2).max(axis=0)

    fig, ax = plt.subplots(figsize=(14, 10), dpi=200)

    collection = PolyCollection(
        polygons_xy,
        facecolors="#d4dbe6",
        edgecolors="#374151",
        linewidths=0.15,
        alpha=0.85,
    )
    ax.add_collection(collection)

    pad = max((maxs - mins).max() * 0.02, 0.5)
    ax.set_xlim(mins[0] - pad, maxs[0] + pad)
    ax.set_ylim(mins[1] - pad, maxs[1] + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("IFC Top View Projection", fontsize=16)
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_ifc = (
        script_dir.parent
        / "minus197_mapping"
        / "data"
        / "ifc_files"
        / "20201022mapping IFC4 Convenience store.ifc"
    )
    default_output = script_dir / "20201022mapping IFC4 Convenience store_top_view.png"

    parser = argparse.ArgumentParser(
        description="Render a 2D top-view image from an IFC model."
    )
    parser.add_argument("--ifc", type=Path, default=default_ifc, help="Input IFC file path")
    parser.add_argument(
        "--output", type=Path, default=default_output, help="Output image file path"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    polygons_xy = collect_projected_triangles(args.ifc)
    render_top_view(polygons_xy, args.output)
    print(f"Saved image: {args.output}")


if __name__ == "__main__":
    main()
