"""
run_pathfinding.py
-------------------
CLI to load a saved _graph.json, compute edge costs, and optionally run a
find_path query.

Usage
-----
Inspect a graph (list zone_centroid nodes, connectivity, costs summary):
    python run_pathfinding.py --graph ../minus197_mapping/data/outputs/FLOOR_graph.json

Run a route query:
    python run_pathfinding.py --graph ../minus197_mapping/data/outputs/FLOOR_graph.json \\
        --from ZONE-2SAGvqDRX6DRBKPckG2ybK --to ZONE-233bRpVtr7PADnPMIOrRa2

Run a route query by shop/admin name instead of node_id:
    python run_pathfinding.py --graph ../minus197_mapping/data/outputs/FLOOR_graph.json \\
        --from-name SINGER --to-name POPEYES

Evaluate the found path (endpoint check — starts/ends at requested nodes):
    python run_pathfinding.py --graph ../minus197_mapping/data/outputs/FLOOR_graph.json \\
        --from-name SINGER --to-name POPEYES --evaluate
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import networkx as nx

from shared.types import FloorGraph
from pathfinding.engine import PathfindingEngine, to_feedback_json
from pathfinding.search import build_nx_graph
from pathfinding.cost import CostWeights, compute_edge_costs, normalise_landmark_scores
from pathfinding.path_node_adjuster import (
    adjust_with_path_nodes, load_corridor_walls, load_path_nodes,
)
from evaluation import EndpointEvaluator, EvaluationRunner


def _no_wall_checker(x1: float, y1: float, x2: float, y2: float) -> bool:
    return False


def _find_by_name(graph: FloorGraph, name: str):
    name = name.strip().lower()
    for n in graph.nodes:
        admin_name = n.tags.get("admin_name", "")
        if admin_name.strip().lower() == name or n.label.strip().lower() == name:
            return n
    return None


def _print_summary(graph: FloorGraph) -> None:
    print(f"floor_label : {graph.floor_label}")
    print(f"source_file : {graph.source_file}")
    print(f"nodes       : {len(graph.nodes)}")
    print(f"edges       : {len(graph.edges)}")
    print("node_type counts:", dict(Counter(n.node_type for n in graph.nodes)))

    G = build_nx_graph(graph)
    comps = list(nx.connected_components(G))
    print(f"connected components: {len(comps)}")
    for i, c in enumerate(comps):
        zones = [
            graph.node(nid).tags.get("admin_name") or graph.node(nid).label
            for nid in c
            if graph.node(nid).node_type == "zone_centroid"
        ]
        print(f"  component {i} (size {len(c)}): zones = {zones}")

    print()
    print("zone_centroid nodes (node_id -> shop/admin name):")
    for n in graph.nodes:
        if n.node_type == "zone_centroid":
            print(f"  {n.node_id}  ->  {n.tags.get('admin_name') or n.label}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a _graph.json, compute edge costs, and optionally find a path."
    )
    parser.add_argument("--graph", required=True, help="Path to the _graph.json file")
    parser.add_argument("--from", dest="start_id", help="Start node_id")
    parser.add_argument("--to", dest="dest_id", help="Destination node_id")
    parser.add_argument("--from-name", dest="start_name", help="Start shop/admin name")
    parser.add_argument("--to-name", dest="dest_name", help="Destination shop/admin name")
    parser.add_argument(
        "--feedback-json", action="store_true",
        help="Also print the simplified feedback-module action JSON",
    )
    parser.add_argument(
        "--save-feedback", dest="save_feedback_path", default=None,
        help="Write the feedback-module action JSON to this file path",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run evaluation checks on the found path (e.g. endpoint check)",
    )
    parser.add_argument(
        "--path-nodes", dest="path_nodes_path", default=None,
        help="Path to the *_path_nodes.json file. When given (with --sfm), "
             "the found route is adjusted to hug nearby path nodes on legs "
             "that run along a corridor.",
    )
    parser.add_argument(
        "--sfm", dest="sfm_path", default=None,
        help="Path to the *_sfm.json file (wall geometry for the path-node "
             "adjustment). Required together with --path-nodes.",
    )
    args = parser.parse_args()

    with open(args.graph, encoding="utf-8") as f:
        data = json.load(f)
    graph = FloorGraph.from_dict(data)

    normalise_landmark_scores(graph, None)
    compute_edge_costs(graph, CostWeights())

    start_id = args.start_id
    dest_id = args.dest_id

    if args.start_name:
        node = _find_by_name(graph, args.start_name)
        if node is None:
            print(f"ERROR: no node found with name '{args.start_name}'")
            sys.exit(1)
        start_id = node.node_id

    if args.dest_name:
        node = _find_by_name(graph, args.dest_name)
        if node is None:
            print(f"ERROR: no node found with name '{args.dest_name}'")
            sys.exit(1)
        dest_id = node.node_id

    if not start_id or not dest_id:
        _print_summary(graph)
        return

    engine = PathfindingEngine(graph, _no_wall_checker)
    result = engine.find_path(start_id, dest_id)

    print(f"found          : {result.found}")
    if not result.found:
        print("No path found — start and destination may be in disconnected "
              "parts of the graph. Run without --from/--to to see components.")
        return

    if args.path_nodes_path and args.sfm_path:
        path_nodes = load_path_nodes(args.path_nodes_path)
        walls = load_corridor_walls(args.sfm_path)
        result = adjust_with_path_nodes(result, path_nodes, walls)
        print(f"path-node adjustment: {len(path_nodes)} path nodes, "
              f"{len(walls)} walls loaded")

    print(f"total_distance : {result.total_distance} m")
    print(f"total_cost     : {result.total_cost}")
    print(f"safety_score   : {result.safety_score}")
    print(f"shore_score    : {result.shore_score}")
    print(f"landmark_score : {result.landmark_score}")
    print(f"alternatives   : {len(result.alternatives)}")
    print()
    print("steps:")
    for step in result.steps:
        print(f"  - {step.instruction}")

    feedback_json = None
    if args.feedback_json or args.save_feedback_path:
        feedback_json = to_feedback_json(result)

    if args.feedback_json:
        print()
        print("feedback json:")
        print(feedback_json)

    if args.save_feedback_path:
        with open(args.save_feedback_path, "w", encoding="utf-8") as f:
            f.write(feedback_json)
        print(f"\nSaved feedback JSON -> {args.save_feedback_path}")

    if args.evaluate:
        runner = EvaluationRunner([EndpointEvaluator()])
        report = runner.run(result, start_id, dest_id)

        print()
        print("evaluation:")
        for r in report.results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.name}: {r.message}")
        print(f"  overall: {'PASS' if report.all_passed() else 'FAIL'}")


if __name__ == "__main__":
    main()
