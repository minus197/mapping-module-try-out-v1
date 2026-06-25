"""
admin_naming/shop_name_ui.py
----------------------------
Interactive CLI for assigning real tenant names to IFC spaces.

Reads {stem}_sfm.json, presents nameable zones (shops, food courts, unknowns),
prompts the admin for each, then writes {stem}_shop_names.json.

Usage (standalone):
    python -m admin_naming.shop_name_ui \\
        --sfm data/outputs/mall_L1_sfm.json \\
        --output data/outputs/

Called programmatically:
    from admin_naming.shop_name_ui import run_ui
    run_ui(sfm_path="data/outputs/mall_L1_sfm.json",
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

# Categories the admin is allowed to name — others are filtered out.
NAMEABLE_CATEGORIES = {"shop", "food_court", "unknown"}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_ui(
    sfm_path: str | Path,
    output_dir: str | Path,
    run_patcher: bool = False,
) -> Path:
    """
    Run the interactive naming session.

    Parameters
    ----------
    sfm_path    : path to {stem}_sfm.json
    output_dir  : directory where {stem}_shop_names.json is written
    run_patcher : if True, invoke shop_name_patcher immediately after saving

    Returns
    -------
    Path to the written {stem}_shop_names.json file.
    """
    sfm_path   = Path(sfm_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sfm_data   = json.loads(sfm_path.read_text(encoding="utf-8"))
    stem       = _stem_from_sfm(sfm_path, sfm_data)
    floor_label = sfm_data.get("meta", {}).get("floor_label", "")

    names_path = output_dir / f"{stem}_shop_names.json"
    existing   = _load_existing(names_path)

    zones = _nameable_zones(sfm_data)

    if not zones:
        print("[AdminNaming] No nameable zones found — nothing to do.")
        return names_path

    _print_header(stem, floor_label, zones, existing)
    mappings = _prompt_session(zones, existing)
    _save_names(names_path, stem, floor_label, mappings)

    _print_summary(len(mappings), len(zones) - len(mappings), names_path)

    if run_patcher or _ask_run_patcher():
        from admin_naming.shop_name_patcher import run_patcher as do_patch
        do_patch(names_path=names_path, output_dir=output_dir)

    return names_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stem_from_sfm(sfm_path: Path, sfm_data: dict) -> str:
    source = sfm_data.get("meta", {}).get("source_file", "")
    if source:
        return Path(source).stem
    # Fallback: strip the trailing _sfm from the filename
    name = sfm_path.stem
    if name.endswith("_sfm"):
        return name[:-4]
    return name


def _load_existing(names_path: Path) -> Dict[str, str]:
    """Return {zone_id: admin_name} from an existing shop_names file, or {}."""
    if not names_path.exists():
        return {}
    try:
        data = json.loads(names_path.read_text(encoding="utf-8"))
        return {m["zone_id"]: m["admin_name"] for m in data.get("mappings", [])}
    except Exception:
        return {}


def _nameable_zones(sfm_data: dict) -> List[dict]:
    return [
        z for z in sfm_data.get("zones", [])
        if z.get("category", "unknown") in NAMEABLE_CATEGORIES
    ]


def _print_header(
    stem: str,
    floor_label: str,
    zones: List[dict],
    existing: Dict[str, str],
) -> None:
    named = sum(1 for z in zones if z["zone_id"] in existing)
    need  = len(zones) - named
    width = 62
    print()
    print("─" * width)
    print(f" Shop Naming — {stem}  (floor {floor_label})")
    print(f" {len(zones)} zones total · {need} need names · "
          "press Enter to skip a zone")
    print("─" * width)

    col_w = [3, 22, 12, 8, 10]
    header = (
        f" {'#':>{col_w[0]}} │ "
        f"{'IFC name (architect)':<{col_w[1]}} │ "
        f"{'Category':<{col_w[2]}} │ "
        f"{'Area':>{col_w[3]}} │ "
        f"{'Status':<{col_w[4]}}"
    )
    print(header)
    print("─" * col_w[0] + "──┼──" +
          "─" * col_w[1] + "──┼──" +
          "─" * col_w[2] + "──┼──" +
          "─" * col_w[3] + "──┼──" +
          "─" * col_w[4])

    for idx, z in enumerate(zones, start=1):
        zone_id  = z["zone_id"]
        ifc_name = (z.get("name") or "")[:col_w[1]]
        category = z.get("category", "")[:col_w[2]]
        area     = f"{z.get('area_m2', 0):.0f} m²"
        status   = existing.get(zone_id, "(empty)")
        print(
            f" {idx:>{col_w[0]}} │ "
            f"{ifc_name:<{col_w[1]}} │ "
            f"{category:<{col_w[2]}} │ "
            f"{area:>{col_w[3]}} │ "
            f"{status:<{col_w[4]}}"
        )
    print("─" * width)
    print()


def _prompt_session(
    zones: List[dict],
    existing: Dict[str, str],
) -> List[dict]:
    """Prompt the admin for each zone; return a list of completed mappings."""
    mappings: List[dict] = []
    total = len(zones)

    for idx, zone in enumerate(zones, start=1):
        zone_id  = zone["zone_id"]
        ifc_name = zone.get("name") or zone_id
        category = zone.get("category", "")
        area     = zone.get("area_m2", 0)
        current  = existing.get(zone_id, "")

        if current:
            prompt = f"  Enter shop name [current: {current}]: "
        else:
            prompt = "  Enter shop name [skip=Enter]: "

        print(f"[{idx}/{total}] {ifc_name} ({category}, {area:.0f} m²)")
        try:
            answer = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        chosen = answer if answer else current
        if chosen:
            mappings.append({
                "zone_id":    zone_id,
                "ifc_name":   ifc_name,
                "admin_name": chosen,
            })
        print()

    return mappings


def _save_names(
    names_path: Path,
    stem: str,
    floor_label: str,
    mappings: List[dict],
) -> None:
    data = {
        "source_stem":   stem,
        "floor_label":   floor_label,
        "generated_at":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "mappings":      mappings,
    }
    names_path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                          encoding="utf-8")


def _print_summary(named: int, skipped: int, names_path: Path) -> None:
    width = 62
    print("─" * width)
    print(f" {named} names entered · {skipped} skipped")
    print(f" Saved → {names_path.resolve()}")


def _ask_run_patcher() -> bool:
    try:
        answer = input(" Run patcher? [Y/n]: ").strip().lower()
        print("─" * 62)
        return answer in ("", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Admin shop-naming UI — assigns real names to IFC spaces"
    )
    parser.add_argument("--sfm",    required=True,
                        help="Path to {stem}_sfm.json")
    parser.add_argument("--output", required=True,
                        help="Output directory for {stem}_shop_names.json")
    parser.add_argument("--patch",  action="store_true",
                        help="Automatically run patcher after saving (no prompt)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    run_ui(
        sfm_path   = args.sfm,
        output_dir = args.output,
        run_patcher = args.patch,
    )


if __name__ == "__main__":
    main()
