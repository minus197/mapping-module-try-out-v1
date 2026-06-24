"""
pathfinding/search.py  —  WP4  (Stage 3 + Stage 4)
----------------------------------------------------
Builds a networkx graph from a FloorGraph whose edges already carry
combined_cost (populated by cost.compute_edge_costs), then exposes three
search functions:

    build_nx_graph(graph)                    → nx.Graph
    optimal_path(G, start_id, dest_id)       → list[str] | None   (Dijkstra)
    optimal_path_astar(G, start_id, dest_id) → list[str] | None   (A*)
    k_best_paths(G, start_id, dest_id, k=4) → list[list[str]]    (Yen's)

All functions return None / [] on bad/unreachable inputs — they never raise.

Yen's k-shortest loopless paths uses networkx.algorithms.simple_paths.
shortest_simple_paths which implements Yen's algorithm internally.

Optional diversity filter: an alternative whose edges overlap > 80 % with
the winner is dropped before the list is truncated to k−1 alternatives.
"""

from __future__ import annotations

import math
from typing import List, Optional

import networkx as nx

from shared.types import FloorGraph, NavigationNode

# Fraction of shared edges above which an alternative is considered a duplicate
_DIVERSITY_THRESHOLD = 0.80


def build_nx_graph(graph: FloorGraph) -> nx.Graph:
    """
    Convert a FloorGraph into an undirected weighted networkx Graph.

    Edge attributes stored:
        weight  — combined_cost   (used by all search algorithms)
        data    — NavigationEdge  (carried through for engine assembly)
    """
    G = nx.Graph()

    for node in graph.nodes:
        G.add_node(node.node_id, data=node)

    for edge in graph.edges:
        # networkx picks the lightest parallel edge automatically for simple
        # graphs; we use add_edge which overwrites — edges should be unique.
        G.add_edge(
            edge.source_id,
            edge.target_id,
            weight=edge.combined_cost,
            data=edge,
        )

    return G


def optimal_path(
    G: nx.Graph,
    start_id: str,
    dest_id: str,
) -> Optional[List[str]]:
    """
    Dijkstra shortest path on combined_cost.

    Returns list of node-id strings (start → dest), or None if unreachable /
    either endpoint is missing.
    """
    if start_id not in G or dest_id not in G:
        return None
    try:
        return nx.dijkstra_path(G, start_id, dest_id, weight="weight")
    except nx.NetworkXNoPath:
        return None


def optimal_path_astar(
    G: nx.Graph,
    start_id: str,
    dest_id: str,
) -> Optional[List[str]]:
    """
    A* shortest path using Euclidean distance as the admissible heuristic.

    The heuristic reads node positions from the 'data' attribute stored by
    build_nx_graph.  Falls back gracefully to zero heuristic (≡ Dijkstra)
    when position data is absent.

    Returns list of node-id strings, or None if unreachable / missing.
    """
    if start_id not in G or dest_id not in G:
        return None

    def heuristic(u: str, v: str) -> float:
        try:
            pu = G.nodes[u]["data"].position
            pv = G.nodes[v]["data"].position
            return math.sqrt((pu[0] - pv[0]) ** 2 + (pu[1] - pv[1]) ** 2)
        except (KeyError, AttributeError, TypeError):
            return 0.0

    try:
        return nx.astar_path(G, start_id, dest_id, heuristic=heuristic, weight="weight")
    except nx.NetworkXNoPath:
        return None


def k_best_paths(
    G: nx.Graph,
    start_id: str,
    dest_id: str,
    k: int = 4,
) -> List[List[str]]:
    """
    Up to k shortest loopless paths (Yen's algorithm via networkx).

    Returns
    -------
    List of node-id lists ordered by combined_cost, with a diversity filter
    applied: any alternative sharing > 80 % of its edges with the winner is
    dropped before truncation to k paths.

    Returns [] when endpoints are missing or no path exists.
    """
    if start_id not in G or dest_id not in G:
        return []

    try:
        gen = nx.shortest_simple_paths(G, start_id, dest_id, weight="weight")
        raw: List[List[str]] = []
        for path in gen:
            raw.append(path)
            if len(raw) >= k * 3:   # over-generate then filter
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []

    if not raw:
        return []

    winner = raw[0]
    winner_edges = _edge_set(winner)

    filtered = [winner]
    for candidate in raw[1:]:
        if len(filtered) >= k:
            break
        cand_edges = _edge_set(candidate)
        if not cand_edges:
            continue
        overlap = len(winner_edges & cand_edges) / len(cand_edges)
        if overlap <= _DIVERSITY_THRESHOLD:
            filtered.append(candidate)

    return filtered


# ── Helpers ───────────────────────────────────────────────────────────────────

def _edge_set(path: List[str]) -> set:
    """Convert an ordered node list into a frozenset of (u,v) edge pairs."""
    return {frozenset((path[i], path[i + 1])) for i in range(len(path) - 1)}
