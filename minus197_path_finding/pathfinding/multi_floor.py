"""
pathfinding/multi_floor.py
--------------------------
Inter-floor routing for the Minus197 pathfinding module.

Same-floor routes are produced by the existing single-floor pipeline. When the
start and destination are on different floors, we run that SAME pipeline twice —
start → start-floor elevator, then dest-floor elevator → destination — and stitch
the two action lists with a `take_elevator` action between them.

Elevator (not stairs) is used for the floor change: the target users are visually
impaired.

Public API
----------
    router = MultiFloorRouter.from_building_graph(bg_path, outputs_dir)
    route  = router.route(start_id, dest_id)        # dict with "actions" + meta
    router.find_node_id_by_name("SINGER")           # building-wide name lookup
"""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

from shared.types import FloorGraph, NavigationNode
from pathfinding.engine import PathfindingEngine, to_feedback_json
from pathfinding.cost import CostWeights, compute_edge_costs, normalise_landmark_scores
from pathfinding.path_node_adjuster import (
    adjust_with_path_nodes, load_corridor_walls, load_path_nodes, load_zones,
)


def _no_wall_checker(x1: float, y1: float, x2: float, y2: float) -> bool:
    # start/dest are given as node_ids, so the KD-tree resolver is never invoked;
    # a permissive checker is correct here (mirrors run_pathfinding.py).
    return False


def _stem_of(source_file: str) -> str:
    """IFC stem from a (possibly Windows) source_file path."""
    if "\\" in source_file:
        return PureWindowsPath(source_file).stem
    return PurePosixPath(source_file).stem


class _FloorContext:
    """One floor's engine + per-floor path-node adjustment inputs."""

    def __init__(self, floor_label: str, graph: FloorGraph, outputs_dir: Path):
        self.floor_label = floor_label
        self.graph = graph

        # Locate this floor's path_nodes / sfm by the output stem convention
        # {ifc_stem}_{floor_label}_*.
        stem = f"{_stem_of(graph.source_file)}_{floor_label}"
        self._pn_path = outputs_dir / f"{stem}_path_nodes.json"
        self._sfm_path = outputs_dir / f"{stem}_sfm.json"

        self._pn = self._walls = self._zones = None  # lazy

        # Path nodes must reach the engine at construction so a path-node id can
        # be passed as the current (start) node to find_path() — mirrors the
        # single-floor path in run_pathfinding.py.
        path_nodes, _, _ = self._load_adjust_inputs()

        # PathfindingEngine already runs normalise_landmark_scores +
        # compute_edge_costs at construction; nothing to do here beforehand.
        self.engine = PathfindingEngine(graph, _no_wall_checker,
                                        path_nodes=path_nodes)

    def _load_adjust_inputs(self):
        if self._pn is None:
            self._pn = load_path_nodes(self._pn_path) if self._pn_path.exists() else []
            self._walls = load_corridor_walls(self._sfm_path) if self._sfm_path.exists() else []
            self._zones = load_zones(self._sfm_path) if self._sfm_path.exists() else []
        return self._pn, self._walls, self._zones

    def path_node_ids(self) -> set:
        """Ids of this floor's path nodes (not part of the graph node set)."""
        pn, _, _ = self._load_adjust_inputs()
        return {p.node_id for p in pn}

    def leg_actions(self, start_id: str, dest_id: str, terminal: bool) -> List[Dict[str, Any]]:
        """Run find_path + path-node adjustment for one leg, return action dicts."""
        result = self.engine.find_path(start_id, dest_id)
        if not result.found:
            return []
        pn, walls, zones = self._load_adjust_inputs()
        if pn and walls:
            # Pin the start path node when the leg begins on one, so the
            # adjuster keeps it as the route's first node.
            start_path_node_id = (
                start_id if start_id in {p.node_id for p in pn} else None
            )
            result = adjust_with_path_nodes(
                result, pn, walls, zones=zones,
                start_path_node_id=start_path_node_id,
            )
        return to_feedback_json(result, terminal=terminal, as_list=True)

    def elevators(self) -> List[NavigationNode]:
        return [n for n in self.graph.nodes if n.node_type == "elevator"]

    def node(self, node_id: str) -> Optional[NavigationNode]:
        return self.graph.node(node_id)


class MultiFloorRouter:
    def __init__(self,
                 floors: Dict[str, _FloorContext],
                 floor_order: List[str],
                 inter_edges: List[Dict[str, Any]]):
        self.floors = floors
        self.floor_order = floor_order      # bottom → top, e.g. ["L3","L4"]
        self.inter_edges = inter_edges
        # node_id -> floor_label
        self.node_floor: Dict[str, str] = {}
        for fl, ctx in floors.items():
            for n in ctx.graph.nodes:
                self.node_floor[n.node_id] = fl

    # ── construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_building_graph(cls, bg_path: str | Path, outputs_dir: str | Path) -> "MultiFloorRouter":
        bg = json.loads(Path(bg_path).read_text(encoding="utf-8"))
        outputs_dir = Path(outputs_dir)

        floors: Dict[str, _FloorContext] = {}
        for fl in bg["floors"]:
            label = fl["floor_label"]
            source_file = fl.get("source_file", "")

            # The building graph embeds a snapshot of each floor's nodes that can
            # lag behind the per-floor _graph.json (e.g. missing zone admin names),
            # so prefer the standalone file when it is present.
            per_floor = outputs_dir / f"{_stem_of(source_file)}_{label}_graph.json"
            if per_floor.exists():
                data = json.loads(per_floor.read_text(encoding="utf-8"))
                data["floor_label"] = label
                data.setdefault("source_file", source_file)
            else:
                data = {
                    "floor_label": label,
                    "source_file": source_file,
                    "nodes": fl["nodes"],
                    "edges": fl["edges"],
                }
            graph = FloorGraph.from_dict(data)
            floors[label] = _FloorContext(label, graph, outputs_dir)

        floor_order = bg.get("meta", {}).get("floor_labels", list(floors.keys()))
        return cls(floors, floor_order, bg.get("inter_floor_edges", []))

    # ── helpers ───────────────────────────────────────────────────────────────

    def find_node_id_by_name(self, name: str) -> Optional[str]:
        """Building-wide name lookup (admin_label / admin_name / label)."""
        name = name.strip().lower()
        for ctx in self.floors.values():
            for n in ctx.graph.nodes:
                admin = n.tags.get("admin_label", "") or n.tags.get("admin_name", "")
                if admin.strip().lower() == name or n.label.strip().lower() == name:
                    return n.node_id
        return None

    def _floor_of(self, node_id: str) -> Optional[str]:
        return self.node_floor.get(node_id)

    def resolve_node_ref(self, ref: str) -> Tuple[Optional[str], Optional[str], str]:
        """
        Resolve a user-supplied node reference to (node_id, floor_label, error).

        Accepts a plain graph node_id, a path-node id, or a floor-qualified
        "L3:PATH-0000". Path-node ids repeat across floors (each floor numbers
        its own from PATH-0000), so an unqualified path-node id is only accepted
        when exactly one floor defines it.
        """
        if ":" in ref:
            floor, _, bare = ref.partition(":")
            if floor not in self.floors:
                return None, None, (f"unknown floor '{floor}' in '{ref}' "
                                    f"(known: {', '.join(self.floor_order)})")
            ctx = self.floors[floor]
            if ctx.node(bare) is not None or bare in ctx.path_node_ids():
                return bare, floor, ""
            return None, None, f"node '{bare}' not found on floor {floor}"

        floor = self.node_floor.get(ref)
        if floor is not None:
            return ref, floor, ""

        # Not a graph node — look for it among the per-floor path nodes.
        hits = [fl for fl, ctx in self.floors.items() if ref in ctx.path_node_ids()]
        if len(hits) == 1:
            return ref, hits[0], ""
        if len(hits) > 1:
            opts = ", ".join(f"{fl}:{ref}" for fl in sorted(hits))
            return None, None, (f"'{ref}' is ambiguous - it exists on "
                                f"{', '.join(sorted(hits))}. Qualify it: {opts}")
        return None, None, f"'{ref}' is not a node id or path-node id in this building"

    def _direction(self, from_floor: str, to_floor: str) -> str:
        return "up" if self.floor_order.index(to_floor) > self.floor_order.index(from_floor) else "down"

    def _paired_elevator(self, start_elev: NavigationNode, dest_floor: str) -> Optional[NavigationNode]:
        """The dest-floor elevator in the same shaft as start_elev.

        Primary: follow an inter_floor edge (connector=elevator) whose source is
        start_elev and whose to_floor is dest_floor. Fallback: match by identical
        XY on the dest floor (same shaft shares XY — verified in this building).
        """
        for e in self.inter_edges:
            if (e.get("tags", {}).get("connector") == "elevator"
                    and e["source_id"] == start_elev.node_id
                    and e.get("tags", {}).get("to_floor") == dest_floor):
                cand = self.floors[dest_floor].node(e["target_id"])
                if cand is not None:
                    return cand
        # Fallback: nearest-XY elevator on the destination floor
        sx, sy = start_elev.position
        best, best_d = None, 1e9
        for n in self.floors[dest_floor].elevators():
            d = (n.position[0] - sx) ** 2 + (n.position[1] - sy) ** 2
            if d < best_d:
                best, best_d = n, d
        return best

    def _choose_start_elevator(self, start_id: str, start_floor: str
                               ) -> Optional[Tuple[NavigationNode, List[Dict[str, Any]]]]:
        """Pick the start-floor elevator giving the lowest-cost leg-A route.

        With one elevator per floor this is unambiguous; the loop generalises to
        multiple elevators/shafts.
        """
        ctx = self.floors[start_floor]
        best = None  # (elevator_node, legA_actions, action_count)
        for elev in ctx.elevators():
            if elev.node_id == start_id:
                # Already standing at the elevator — leg A is empty.
                return (elev, [])
            actions = ctx.leg_actions(start_id, elev.node_id, terminal=False)
            if not actions:
                continue
            if best is None or len(actions) < best[2]:
                best = (elev, actions, len(actions))
        return (best[0], best[1]) if best else None

    # ── public entry point ────────────────────────────────────────────────────

    def route(self, start_id: str, dest_id: str) -> Dict[str, Any]:
        """
        Returns:
            {
              "found": bool,
              "same_floor": bool,
              "start_floor": "L3", "dest_floor": "L4",
              "actions": [ ...flat action list... ],
              "message": str        # present when found is False
            }
        """
        start_id, start_floor, start_err = self.resolve_node_ref(start_id)
        dest_id, dest_floor, dest_err = self.resolve_node_ref(dest_id)

        if start_err or dest_err:
            return {"found": False, "actions": [],
                    "message": "; ".join(m for m in (start_err, dest_err) if m)}

        # find_path resolves a path node only as the START; a destination must be
        # a real graph node.
        if self.floors[dest_floor].node(dest_id) is None:
            return {"found": False, "actions": [],
                    "message": f"destination '{dest_id}' is a path node; "
                               f"destinations must be graph nodes (e.g. a ZONE-* id)"}

        # ── Same floor — existing single-floor behaviour ──────────────────────
        if start_floor == dest_floor:
            actions = self.floors[start_floor].leg_actions(start_id, dest_id, terminal=True)
            return {"found": bool(actions), "same_floor": True,
                    "start_floor": start_floor, "dest_floor": dest_floor,
                    "actions": actions,
                    "message": "" if actions else "no same-floor path found"}

        # ── Different floors — leg A + take_elevator + leg B ──────────────────
        chosen = self._choose_start_elevator(start_id, start_floor)
        if chosen is None:
            return {"found": False, "actions": [],
                    "message": f"no elevator reachable on start floor {start_floor}"}
        start_elev, legA = chosen

        dest_elev = self._paired_elevator(start_elev, dest_floor)
        if dest_elev is None:
            return {"found": False, "actions": [],
                    "message": f"no paired elevator on destination floor {dest_floor}"}

        if dest_elev.node_id == dest_id:
            # Destination is the elevator itself — leg B is just the arrival stop.
            legB = [{
                "action": "stop",
                "landmark": (dest_elev.tags.get("admin_label", "").strip()
                             or dest_elev.label),
                "node_id": dest_elev.node_id,
                "position": [round(dest_elev.position[0], 4),
                             round(dest_elev.position[1], 4)],
            }]
        else:
            legB = self.floors[dest_floor].leg_actions(dest_elev.node_id, dest_id, terminal=True)
        if not legB:
            return {"found": False, "actions": [],
                    "message": f"no path from elevator to destination on {dest_floor}"}

        take = {
            "action":          "take_elevator",
            "direction":       self._direction(start_floor, dest_floor),
            "from_floor":      start_floor,
            "to_floor":        dest_floor,
            "connector":       "elevator",
            "node_id":         start_elev.node_id,
            "position":        [round(start_elev.position[0], 4), round(start_elev.position[1], 4)],
            "arrive_node_id":  dest_elev.node_id,
            "arrive_position": [round(dest_elev.position[0], 4), round(dest_elev.position[1], 4)],
        }

        return {"found": True, "same_floor": False,
                "start_floor": start_floor, "dest_floor": dest_floor,
                "actions": legA + [take] + legB, "message": ""}
