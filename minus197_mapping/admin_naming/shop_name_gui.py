"""
admin_naming/shop_name_gui.py
-----------------------------
Graphical (click-to-name) admin UI for assigning real tenant names to IFC zones.

The text CLI (shop_name_ui.py) lists zones by their IFC architect codes — "401",
"402", "C4" — which an admin cannot map to a physical location by eye. This module
draws the floor plan instead: the admin clicks a zone (or picks it from a synced
list), then types a **title** and **description** for it.

It is added *separately* and imported into the pipeline — nothing in the existing
CLI, the patcher's public contract, or the visualizer is rewired. It reuses:

    visualizer.map_visualizer   ZONE_FILL / draw_walls / draw_features  (styling)
    admin_naming.shop_name_ui   _stem_from_sfm                          (stem logic)
    admin_naming.shop_name_patcher.run_patcher                          (apply names)
    matplotlib.path.Path.contains_point                                 (hit-testing)

It writes the same {stem}_shop_names.json the CLI produces, plus a new
`admin_description` field per mapping.

Usage (standalone):
    python -m admin_naming.shop_name_gui \\
        --sfm data/outputs/mall_L1_sfm.json \\
        --output data/outputs/

Called programmatically (mirrors run_ui so main.py can swap it in):
    from admin_naming.shop_name_gui import run_gui
    run_gui(sfm_path="data/outputs/mall_L1_sfm.json",
            output_dir="data/outputs/",
            run_patcher=True)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Make `visualizer` and `admin_naming` importable regardless of the cwd / how the
# module is launched (python -m ..., or imported from main.py).
_MODULE_ROOT = Path(__file__).resolve().parent.parent
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

try:
    from matplotlib.figure import Figure
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.path import Path as MplPath
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg,
        NavigationToolbar2Tk,
    )
except ImportError as exc:  # pragma: no cover - dependency guard
    sys.exit(
        f"Missing dependency: {exc.name}. Install with:  pip install matplotlib"
    )

# Reuse the visualizer's styling and wall/feature renderers. Fall back to local
# copies if the visualizer package cannot be found, so the GUI never hard-fails.
try:
    from visualizer.map_visualizer import ZONE_FILL, draw_walls, draw_features
except Exception:  # pragma: no cover - resilience fallback
    ZONE_FILL = {
        "corridor": "#fff3b0", "shop": "#a8e6cf", "food_court": "#ffd3b6",
        "restroom": "#c7ceea", "entrance": "#b5ead7", "exit": "#ffb7b2",
        "storage": "#d5c6e0", "office": "#c9e4de", "unknown": "#e8e8e8",
    }

    def draw_walls(ax, sfm):  # type: ignore[no-redef]
        for w in sfm.get("walls", []):
            s, e = w.get("start"), w.get("end")
            if s and e:
                ax.plot([s[0], e[0]], [s[1], e[1]], color="#555555",
                        linewidth=1.1, zorder=3, solid_capstyle="round")

    def draw_features(ax, sfm):  # type: ignore[no-redef]
        for f in sfm.get("features", []):
            pos = f.get("position")
            if pos:
                ax.scatter(pos[0], pos[1], marker=".", c="#9e9e9e", s=40, zorder=7)

from admin_naming.shop_name_ui import _stem_from_sfm


# ---------------------------------------------------------------------------
# Pure logic (no Tk) — unit-testable without a display
# ---------------------------------------------------------------------------

def load_zones(sfm_data: dict) -> List[dict]:
    """Build lightweight view-models for every zone in the SFM.

    Zones without a usable boundary polygon are still returned (selectable via
    the list, drawn as a centroid marker) but carry ``path=None`` so they are
    not click-hit-tested on the map.
    """
    zones: List[dict] = []
    for z in sfm_data.get("zones", []):
        poly = z.get("boundary_polygon") or None
        path = MplPath(poly) if poly and len(poly) >= 3 else None
        zones.append({
            "zone_id":   z.get("zone_id", ""),
            "ifc_name":  z.get("name") or z.get("long_name") or z.get("zone_id", ""),
            "long_name": z.get("long_name", ""),
            "category":  z.get("category", "unknown"),
            "area_m2":   float(z.get("area_m2", 0) or 0),
            "centroid":  z.get("centroid"),
            "polygon":   poly,
            "path":      path,
        })
    return zones


def hit_test(x: float, y: float, zones: List[dict]) -> Optional[dict]:
    """Return the zone under (x, y), or None.

    When polygons overlap (e.g. a shop sitting inside a large corridor's bbox),
    prefer the smallest-area match so the inner zone wins.
    """
    hits = [z for z in zones
            if z.get("path") is not None and z["path"].contains_point((x, y))]
    if not hits:
        return None
    return min(hits, key=lambda z: z["area_m2"] if z["area_m2"] > 0 else float("inf"))


def load_existing_names(names_path: str | Path) -> Dict[str, dict]:
    """Read a {stem}_shop_names.json into {zone_id: {admin_name, admin_description}}.

    Tolerant of files written by the text CLI (which omit ``admin_description``).
    """
    names_path = Path(names_path)
    if not names_path.exists():
        return {}
    try:
        data = json.loads(names_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, dict] = {}
    for m in data.get("mappings", []):
        zid = m.get("zone_id")
        if not zid:
            continue
        out[zid] = {
            "admin_name":        (m.get("admin_name") or "").strip(),
            "admin_description": (m.get("admin_description") or "").strip(),
        }
    return out


def save_names(
    names_path: str | Path,
    stem: str,
    floor_label: str,
    mappings: Dict[str, dict],
) -> List[dict]:
    """Write the extended {stem}_shop_names.json. Returns the written records.

    Only zones with a non-empty title or description are written. Every record
    carries all four keys so the text CLI's loader stays happy.
    """
    records: List[dict] = []
    for zid, rec in mappings.items():
        name = (rec.get("admin_name") or "").strip()
        desc = (rec.get("admin_description") or "").strip()
        if not name and not desc:
            continue
        records.append({
            "zone_id":           zid,
            "ifc_name":          rec.get("ifc_name", ""),
            "admin_name":        name,
            "admin_description": desc,
        })
    records.sort(key=lambda r: r["ifc_name"].lower())

    data = {
        "source_stem":  stem,
        "floor_label":  floor_label,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "mappings":     records,
    }
    Path(names_path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return records


# ---------------------------------------------------------------------------
# Tk GUI
# ---------------------------------------------------------------------------

# Zone outline styles: (edgecolor, linewidth, alpha)
_STYLE_BASE     = ("#888888", 0.8, 0.45)
_STYLE_NAMED    = ("#2e7d32", 2.2, 0.55)
_STYLE_SELECTED = ("#ff6d00", 2.8, 0.68)


class ShopNameGUI:
    """Interactive window: floor plan on the left, naming form on the right."""

    def __init__(
        self,
        root,
        sfm_data: dict,
        zones: List[dict],
        stem: str,
        floor_label: str,
        names_path: Path,
        run_patcher: bool = False,
        container=None,
    ) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        # Widgets are built into `container`. In single-floor mode this is the
        # root window itself; in tabbed multi-floor mode it is a notebook tab
        # frame, so several ShopNameGUIs share one Tk root without fighting over
        # window-level state (title, geometry, protocol).
        self.container = container if container is not None else root
        self._owns_window = container is None
        self.sfm_data = sfm_data
        self.zones = zones
        self.zone_by_id = {z["zone_id"]: z for z in zones}
        self.stem = stem
        self.floor_label = floor_label
        self.names_path = Path(names_path)
        self._force_patch = run_patcher

        # In-memory naming state, seeded from any existing file.
        self.mappings: Dict[str, dict] = {}
        for zid, rec in load_existing_names(self.names_path).items():
            if zid in self.zone_by_id:
                self.mappings[zid] = {
                    "ifc_name":          self.zone_by_id[zid]["ifc_name"],
                    "admin_name":        rec["admin_name"],
                    "admin_description": rec["admin_description"],
                }

        self.selected_id: Optional[str] = None
        self.patch_by_id: Dict[str, MplPolygon] = {}
        self.label_by_id: Dict[str, object] = {}
        self.tree_item_by_id: Dict[str, str] = {}
        self.id_by_tree_item: Dict[str, str] = {}
        self._patch_requested = False

        if self._owns_window:
            root.title(f"Zone Naming — {stem} (floor {floor_label})")
            root.geometry("1320x840")
            root.minsize(1000, 640)
            root.protocol("WM_DELETE_WINDOW", self.on_window_close)

        self._build_layout()
        self._build_tree_rows()
        self._update_progress()
        self._set_status("Click a zone on the map, or pick one from the list.")

        root.bind("<Control-s>", lambda e: self.on_save_zone())

    # ── Layout ──────────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        tk, ttk = self.tk, self.ttk

        paned = ttk.PanedWindow(self.container, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned, width=380)
        right.pack_propagate(False)
        paned.add(left, weight=4)
        paned.add(right, weight=0)

        # --- Map (embedded matplotlib) ---
        self.fig = Figure(figsize=(11, 8))
        self.ax = self.fig.add_subplot(111)
        self._draw_base()

        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, left, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)

        # --- Right control panel ---
        header = ttk.Label(right, text=f"{self.stem}",
                           font=("Segoe UI", 11, "bold"))
        header.pack(anchor="w", padx=12, pady=(10, 0))
        ttk.Label(right, text=f"Floor {self.floor_label or '—'}").pack(
            anchor="w", padx=12)

        self.progress_var = tk.StringVar(master=self.root, value="")
        ttk.Label(right, textvariable=self.progress_var,
                  foreground="#2e7d32").pack(anchor="w", padx=12, pady=(2, 6))

        # Filter
        filt = ttk.Frame(right)
        filt.pack(fill="x", padx=12)
        ttk.Label(filt, text="Filter:").pack(side="left")
        self.filter_var = tk.StringVar(master=self.root)
        ent = ttk.Entry(filt, textvariable=self.filter_var)
        ent.pack(side="left", fill="x", expand=True, padx=(6, 0))
        ent.bind("<KeyRelease>", lambda e: self._build_tree_rows(self.filter_var.get()))

        # Zone list
        list_frame = ttk.Frame(right)
        list_frame.pack(fill="both", expand=True, padx=12, pady=(6, 6))
        cols = ("name", "cat", "area", "status")
        self.tree = ttk.Treeview(list_frame, columns=cols, show="headings", height=14)
        self.tree.heading("name", text="IFC name")
        self.tree.heading("cat", text="Category")
        self.tree.heading("area", text="Area m²")
        self.tree.heading("status", text="✓")
        self.tree.column("name", width=115, anchor="w")
        self.tree.column("cat", width=95, anchor="w")
        self.tree.column("area", width=60, anchor="e")
        self.tree.column("status", width=32, anchor="center")
        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=12, pady=4)

        # Selected zone details
        det = ttk.Frame(right)
        det.pack(fill="x", padx=12)
        self.lbl_ifc = self._detail_row(det, "IFC name:", 0)
        self.lbl_cat = self._detail_row(det, "Category:", 1)
        self.lbl_area = self._detail_row(det, "Area:", 2)
        self.lbl_zid = self._detail_row(det, "Zone id:", 3, small=True)

        # Title + description form
        form = ttk.Frame(right)
        form.pack(fill="x", padx=12, pady=(8, 4))
        ttk.Label(form, text="Title", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.ent_title_var = tk.StringVar(master=self.root)
        self.ent_title = ttk.Entry(form, textvariable=self.ent_title_var)
        self.ent_title.pack(fill="x", pady=(0, 6))
        self.ent_title.bind("<Return>", lambda e: self.on_save_zone())

        ttk.Label(form, text="Description", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        desc_frame = ttk.Frame(form)
        desc_frame.pack(fill="x")
        self.txt_desc = tk.Text(desc_frame, height=5, wrap="word",
                                font=("Segoe UI", 9))
        dsb = ttk.Scrollbar(desc_frame, orient="vertical", command=self.txt_desc.yview)
        self.txt_desc.configure(yscrollcommand=dsb.set)
        self.txt_desc.pack(side="left", fill="both", expand=True)
        dsb.pack(side="right", fill="y")

        btns = ttk.Frame(right)
        btns.pack(fill="x", padx=12, pady=(8, 4))
        ttk.Button(btns, text="Save zone  (Ctrl+S)",
                   command=self.on_save_zone).pack(side="left")
        ttk.Button(btns, text="Clear",
                   command=self.on_clear_zone).pack(side="left", padx=(6, 0))

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=12, pady=6)

        foot = ttk.Frame(right)
        foot.pack(fill="x", padx=12, pady=(0, 6))
        ttk.Button(foot, text="Save & Apply to outputs",
                   command=self.on_save_patch).pack(fill="x")
        row2 = ttk.Frame(right)
        row2.pack(fill="x", padx=12)
        ttk.Button(row2, text="Save & Close",
                   command=self.on_save_close).pack(side="left", fill="x", expand=True)
        ttk.Button(row2, text="Cancel",
                   command=self.on_cancel).pack(side="left", fill="x",
                                                expand=True, padx=(6, 0))

        self.status_var = tk.StringVar(master=self.root, value="")
        ttk.Label(right, textvariable=self.status_var, wraplength=350,
                  foreground="#555555").pack(anchor="w", padx=12, pady=(6, 10))

    def _detail_row(self, parent, label, row, small=False):
        ttk = self.ttk
        ttk.Label(parent, text=label, font=("Segoe UI", 9, "bold")).grid(
            row=row, column=0, sticky="w", pady=1)
        font = ("Consolas", 7) if small else ("Segoe UI", 9)
        val = ttk.Label(parent, text="—", font=font)
        val.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=1)
        return val

    # ── Map rendering ────────────────────────────────────────────────────────

    def _draw_base(self) -> None:
        ax = self.ax
        try:
            draw_walls(ax, self.sfm_data)
            draw_features(ax, self.sfm_data)
        except Exception:
            pass

        for z in self.zones:
            poly = z["polygon"]
            color = ZONE_FILL.get(z["category"], ZONE_FILL.get("unknown", "#e8e8e8"))
            if poly and len(poly) >= 3:
                patch = MplPolygon(poly, closed=True, facecolor=color,
                                   edgecolor=_STYLE_BASE[0], linewidth=_STYLE_BASE[1],
                                   alpha=_STYLE_BASE[2], zorder=2)
                ax.add_patch(patch)
                self.patch_by_id[z["zone_id"]] = patch
            cen = z.get("centroid")
            if cen:
                t = ax.text(cen[0], cen[1], z["ifc_name"], fontsize=6, ha="center",
                            va="center", color="#222222", zorder=6)
                self.label_by_id[z["zone_id"]] = t

        # Highlight ring for the current selection (works for any zone).
        (self._marker,) = ax.plot([], [], marker="o", markersize=18,
                                  markerfacecolor="none", markeredgecolor="#ff6d00",
                                  markeredgewidth=2.5, zorder=8, visible=False)

        for zid in self.patch_by_id:
            self._style_zone(zid)
        for zid in self.label_by_id:
            self._update_label(zid)

        bb = self.sfm_data.get("bounding_box", {})
        if bb and all(k in bb for k in ("min_x", "max_x", "min_y", "max_y")):
            px = (bb["max_x"] - bb["min_x"]) * 0.02 or 1.0
            py = (bb["max_y"] - bb["min_y"]) * 0.02 or 1.0
            ax.set_xlim(bb["min_x"] - px, bb["max_x"] + px)
            ax.set_ylim(bb["min_y"] - py, bb["max_y"] + py)
        else:
            ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set_title(f"{self.stem} (floor {self.floor_label or '—'}) — click a zone",
                     fontsize=10)
        ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)

    def _is_named(self, zid: str) -> bool:
        rec = self.mappings.get(zid, {})
        return bool(rec.get("admin_name") or rec.get("admin_description"))

    def _style_zone(self, zid: str) -> None:
        patch = self.patch_by_id.get(zid)
        if patch is None:
            return
        if zid == self.selected_id:
            edge, lw, alpha = _STYLE_SELECTED
        elif self._is_named(zid):
            edge, lw, alpha = _STYLE_NAMED
        else:
            edge, lw, alpha = _STYLE_BASE
        patch.set_edgecolor(edge)
        patch.set_linewidth(lw)
        patch.set_alpha(alpha)

    def _update_label(self, zid: str) -> None:
        txt = self.label_by_id.get(zid)
        if txt is None:
            return
        name = self.mappings.get(zid, {}).get("admin_name") or ""
        txt.set_text(name if name else self.zone_by_id[zid]["ifc_name"])
        txt.set_fontweight("bold" if name else "normal")
        txt.set_color("#0d47a1" if name else "#222222")

    # ── Selection ────────────────────────────────────────────────────────────

    def select_zone(self, zid: str, from_tree: bool = False) -> None:
        if zid not in self.zone_by_id:
            return
        prev = self.selected_id
        self.selected_id = zid
        if prev and prev != zid:
            self._style_zone(prev)
        self._style_zone(zid)

        cen = self.zone_by_id[zid].get("centroid")
        if cen:
            self._marker.set_data([cen[0]], [cen[1]])
            self._marker.set_visible(True)
        else:
            self._marker.set_visible(False)
        self.canvas.draw_idle()

        self._populate_form(zid)

        if not from_tree:
            iid = self.tree_item_by_id.get(zid)
            if iid:
                self.tree.selection_set(iid)
                self.tree.see(iid)

    def _populate_form(self, zid: str) -> None:
        z = self.zone_by_id[zid]
        rec = self.mappings.get(zid, {})
        self.lbl_ifc.config(text=z["ifc_name"] or "—")
        self.lbl_cat.config(text=z["category"])
        self.lbl_area.config(text=f"{z['area_m2']:.1f} m²")
        self.lbl_zid.config(text=z["zone_id"])
        self.ent_title_var.set(rec.get("admin_name", ""))
        self.txt_desc.delete("1.0", "end")
        self.txt_desc.insert("1.0", rec.get("admin_description", ""))
        self._set_status(f"Selected {z['ifc_name']} ({z['category']}).")

    # ── Zone list ──────────────────────────────────────────────────────────────

    def _build_tree_rows(self, filter_text: str = "") -> None:
        self.tree.delete(*self.tree.get_children())
        self.tree_item_by_id.clear()
        self.id_by_tree_item.clear()
        ft = (filter_text or "").lower().strip()
        for z in sorted(self.zones, key=lambda z: str(z["ifc_name"]).lower()):
            rec = self.mappings.get(z["zone_id"], {})
            name = rec.get("admin_name") or ""
            hay = f"{z['ifc_name']} {z['category']} {name}".lower()
            if ft and ft not in hay:
                continue
            status = "✓" if self._is_named(z["zone_id"]) else "—"
            iid = self.tree.insert(
                "", "end",
                values=(z["ifc_name"], z["category"], f"{z['area_m2']:.0f}", status),
            )
            self.tree_item_by_id[z["zone_id"]] = iid
            self.id_by_tree_item[iid] = z["zone_id"]
        # Keep the current selection highlighted in the rebuilt list.
        if self.selected_id in self.tree_item_by_id:
            self.tree.selection_set(self.tree_item_by_id[self.selected_id])

    def _refresh_tree_row(self, zid: str) -> None:
        iid = self.tree_item_by_id.get(zid)
        if not iid:
            return
        z = self.zone_by_id[zid]
        status = "✓" if self._is_named(zid) else "—"
        self.tree.item(iid, values=(z["ifc_name"], z["category"],
                                    f"{z['area_m2']:.0f}", status))

    def on_tree_select(self, _event) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        zid = self.id_by_tree_item.get(sel[0])
        if zid and zid != self.selected_id:
            self.select_zone(zid, from_tree=True)

    # ── Canvas interaction ──────────────────────────────────────────────────────

    def on_canvas_click(self, event) -> None:
        if event.inaxes != self.ax:
            return
        if self.toolbar.mode:          # pan/zoom tool active → ignore clicks
            return
        if event.xdata is None or event.ydata is None:
            return
        z = hit_test(event.xdata, event.ydata, self.zones)
        if z:
            self.select_zone(z["zone_id"])
        else:
            self._set_status("No zone at that point — try clicking inside a shape.")

    # ── Actions ──────────────────────────────────────────────────────────────

    def on_save_zone(self) -> None:
        if not self.selected_id:
            self._set_status("Select a zone first.")
            return
        zid = self.selected_id
        z = self.zone_by_id[zid]
        name = self.ent_title_var.get().strip()
        desc = self.txt_desc.get("1.0", "end").strip()
        if not name and not desc:
            self.mappings.pop(zid, None)
        else:
            self.mappings[zid] = {
                "ifc_name": z["ifc_name"],
                "admin_name": name,
                "admin_description": desc,
            }
        self._style_zone(zid)
        self._update_label(zid)
        self._refresh_tree_row(zid)
        self._update_progress()
        self.canvas.draw_idle()
        self._autosave()
        self._set_status(f"Saved {z['ifc_name']} → {name or '(cleared)'}")

    def on_clear_zone(self) -> None:
        if not self.selected_id:
            return
        zid = self.selected_id
        self.mappings.pop(zid, None)
        self.ent_title_var.set("")
        self.txt_desc.delete("1.0", "end")
        self._style_zone(zid)
        self._update_label(zid)
        self._refresh_tree_row(zid)
        self._update_progress()
        self.canvas.draw_idle()
        self._autosave()
        self._set_status(f"Cleared {self.zone_by_id[zid]['ifc_name']}")

    def _close_window(self) -> None:
        """Destroy the toplevel — but only when this GUI owns it. In tabbed
        mode the notebook owns the window, so the tab just autosaves instead."""
        if self._owns_window:
            self.root.destroy()

    def on_save_patch(self) -> None:
        self._commit_dirty_form()
        self._autosave()
        self._patch_requested = True
        if self._owns_window:
            # Single-floor window: patching runs in run_gui after mainloop ends.
            self._close_window()
        else:
            # Tabbed mode: closing would kill every floor's tab, so instead
            # apply THIS floor's names to its output JSONs right now and keep
            # the window open for the other tabs.
            self._apply_patch_now()

    def _apply_patch_now(self) -> None:
        """Run the patcher for this floor immediately (tabbed mode)."""
        try:
            from admin_naming.shop_name_patcher import run_patcher as do_patch
            do_patch(names_path=self.names_path,
                     output_dir=self.names_path.parent)
            named = self._named_count()
            self._set_status(
                f"Applied {named} name(s) to floor {self.floor_label or self.stem} "
                f"outputs → {self.names_path.name}"
            )
        except Exception as exc:  # pragma: no cover - IO/patcher guard
            self._set_status(f"Apply failed: {exc}")

    def on_save_close(self) -> None:
        self._commit_dirty_form()
        self._autosave()
        self._close_window()

    def on_cancel(self) -> None:
        from tkinter import messagebox
        if self._has_unsaved_form():
            if not messagebox.askyesno(
                "Discard current edits?",
                "The title/description you're editing hasn't been saved.\n"
                "Zones you already saved are kept. Close anyway?",
            ):
                return
        self._close_window()

    def on_window_close(self) -> None:
        from tkinter import messagebox
        if self._has_unsaved_form():
            ans = messagebox.askyesnocancel(
                "Save current zone?",
                "Save the title/description for the current zone before closing?",
            )
            if ans is None:
                return
            if ans:
                self.on_save_zone()
        self._autosave()
        self.root.destroy()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _has_unsaved_form(self) -> bool:
        if not self.selected_id:
            return False
        rec = self.mappings.get(self.selected_id, {})
        name = self.ent_title_var.get().strip()
        desc = self.txt_desc.get("1.0", "end").strip()
        return (name != (rec.get("admin_name") or "")
                or desc != (rec.get("admin_description") or ""))

    def _commit_dirty_form(self) -> None:
        if self._has_unsaved_form():
            self.on_save_zone()

    def _autosave(self) -> None:
        try:
            save_names(self.names_path, self.stem, self.floor_label, self.mappings)
        except Exception as exc:  # pragma: no cover - IO guard
            self._set_status(f"Autosave failed: {exc}")

    def _named_count(self) -> int:
        return sum(1 for zid in self.mappings if self._is_named(zid))

    def _update_progress(self) -> None:
        self.progress_var.set(f"{self._named_count()} / {len(self.zones)} zones named")

    def _set_status(self, msg: str) -> None:
        self.status_var.set(msg)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _open_naming_window():
    """Create the root window the naming GUI will live in.

    ``Tk_MainLoop()`` (what Tkinter's ``mainloop()`` calls) only returns once
    the process-wide count of *all* Tk main windows reaches zero — that count
    is global, not scoped to the particular ``Tk()`` instance whose
    ``mainloop()`` you called. So if a host application already has a ``Tk()``
    root running (the normal case when this GUI is embedded), opening a
    second ``Tk()`` here and calling its ``mainloop()`` would never return
    when the admin closes this window — it would keep blocking until the
    *host's* root is also destroyed, silently pumping the host window's
    events in the meantime.

    When a root already exists we instead open a ``Toplevel`` of it and block
    with ``wait_window()``, which returns as soon as this specific window is
    destroyed — the same technique ``tkinter.filedialog``/``messagebox`` use
    internally. Standalone (CLI) use, where no root exists yet, still gets
    its own ``Tk()`` + ``mainloop()``.

    Returns (root, owns_root).
    """
    import tkinter as tk
    existing_root = tk._default_root
    if existing_root is not None:
        return tk.Toplevel(existing_root), False
    return tk.Tk(), True


def _run_naming_window(root, owns_root: bool) -> None:
    if owns_root:
        root.mainloop()
    else:
        root.wait_window()


def run_gui(
    sfm_path: str | Path,
    output_dir: str | Path,
    run_patcher: bool = False,
) -> Path:
    """Launch the graphical naming session.

    Parameters mirror ``shop_name_ui.run_ui`` so the pipeline can swap them.
    Returns the path to {stem}_shop_names.json (written on save).
    """
    sfm_path = Path(sfm_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sfm_data = json.loads(sfm_path.read_text(encoding="utf-8"))
    stem = _stem_from_sfm(sfm_path, sfm_data)
    floor_label = sfm_data.get("meta", {}).get("floor_label", "")
    names_path = output_dir / f"{stem}_shop_names.json"

    zones = load_zones(sfm_data)
    if not zones:
        print("[AdminNamingGUI] No zones found in the SFM — nothing to name.")
        return names_path

    try:
        root, owns_root = _open_naming_window()
    except Exception as exc:
        print(f"[AdminNamingGUI] Could not open a GUI window: {exc}")
        print("  This environment has no display. Use the text UI instead:")
        print(f'    python -m admin_naming.shop_name_ui '
              f'--sfm "{sfm_path}" --output "{output_dir}"')
        return names_path

    app = ShopNameGUI(root, sfm_data, zones, stem, floor_label,
                      names_path, run_patcher)
    _run_naming_window(root, owns_root)

    if app._patch_requested or run_patcher:
        from admin_naming.shop_name_patcher import run_patcher as do_patch
        do_patch(names_path=names_path, output_dir=output_dir)

    print(f"[AdminNamingGUI] Names saved -> {names_path.resolve()}")
    return names_path


def run_gui_multi(
    sfm_paths: List[str | Path],
    output_dir: str | Path,
    run_patcher: bool = False,
) -> List[Path]:
    """Launch ONE window with a tab per floor.

    Each ``sfm_path`` (a ``{stem}_sfm.json``) becomes a notebook tab hosting a
    full ShopNameGUI, so the admin names zones for every floor in a single
    session — floor L3 on one tab, floor L4 on the next, etc. Each tab
    autosaves its own ``{stem}_shop_names.json``.

    Returns the list of names-file paths (one per floor). Mirrors ``run_gui``
    for the single-floor case, so ``main.py`` can pick whichever fits.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load every floor's SFM up front; skip any with no nameable zones so we
    # never build an empty tab.
    floors: List[dict] = []
    for sp in sfm_paths:
        sp = Path(sp)
        if not sp.exists():
            print(f"[AdminNamingGUI] SFM not found, skipping: {sp}")
            continue
        sfm_data = json.loads(sp.read_text(encoding="utf-8"))
        zones = load_zones(sfm_data)
        if not zones:
            print(f"[AdminNamingGUI] No zones in {sp.name} — skipping tab.")
            continue
        stem = _stem_from_sfm(sp, sfm_data)
        floor_label = sfm_data.get("meta", {}).get("floor_label", "")
        floors.append({
            "sfm_path":    sp,
            "sfm_data":    sfm_data,
            "zones":       zones,
            "stem":        stem,
            "floor_label": floor_label,
            "names_path":  output_dir / f"{stem}_shop_names.json",
        })

    if not floors:
        print("[AdminNamingGUI] No floors with nameable zones — nothing to do.")
        return []

    # A single floor doesn't need tabs — reuse the plain window.
    if len(floors) == 1:
        f = floors[0]
        return [run_gui(f["sfm_path"], output_dir, run_patcher)]

    try:
        from tkinter import ttk
        root, owns_root = _open_naming_window()
    except Exception as exc:
        print(f"[AdminNamingGUI] Could not open a GUI window: {exc}")
        print("  This environment has no display. Use the text UI per floor instead.")
        return [f["names_path"] for f in floors]

    root.title("Zone Naming — all floors")
    root.geometry("1360x880")
    root.minsize(1000, 640)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    apps: List[ShopNameGUI] = []
    for f in floors:
        tab = ttk.Frame(notebook)
        label = f["floor_label"] or f["stem"]
        notebook.add(tab, text=f"Floor {label}")
        app = ShopNameGUI(
            root         = root,
            sfm_data     = f["sfm_data"],
            zones        = f["zones"],
            stem         = f["stem"],
            floor_label  = f["floor_label"],
            names_path   = f["names_path"],
            run_patcher  = run_patcher,
            container    = tab,
        )
        apps.append(app)

    # One window-level close: commit + autosave every tab, then tear down.
    def _on_close() -> None:
        for a in apps:
            a._commit_dirty_form()
            a._autosave()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)

    # A global toolbar under the tabs so the admin can finish the whole session
    # in one click regardless of which tab is active.
    bar = ttk.Frame(root)
    bar.pack(fill="x", side="bottom")
    ttk.Button(bar, text="Save & Apply all floors",
               command=lambda: (_mark_patch(apps), _on_close())
               ).pack(side="right", padx=8, pady=6)
    ttk.Button(bar, text="Save & Close",
               command=_on_close).pack(side="right", pady=6)

    _run_naming_window(root, owns_root)

    names_paths = [f["names_path"] for f in floors]
    if run_patcher or any(a._patch_requested for a in apps):
        from admin_naming.shop_name_patcher import run_patcher as do_patch
        for np_ in names_paths:
            if np_.exists():
                do_patch(names_path=np_, output_dir=output_dir)

    for np_ in names_paths:
        print(f"[AdminNamingGUI] Names saved -> {np_.resolve()}")
    return names_paths


def _mark_patch(apps: List["ShopNameGUI"]) -> None:
    for a in apps:
        a._patch_requested = True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Graphical admin zone-naming UI (click a zone, add title + description)"
    )
    parser.add_argument("--sfm", required=True, help="Path to {stem}_sfm.json")
    parser.add_argument("--output", required=True,
                        help="Output directory for {stem}_shop_names.json")
    parser.add_argument("--patch", action="store_true",
                        help="Apply names to the output JSONs after the window closes")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    run_gui(sfm_path=args.sfm, output_dir=args.output, run_patcher=args.patch)


if __name__ == "__main__":
    main()
