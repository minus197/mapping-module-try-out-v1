"""
pathfinding/engine.py  —  Sprint 5–6
--------------------------------------
Pathfinding engine that consumes a FloorGraph (from map_extraction)
and produces ranked, annotated navigation paths for a visually impaired user.

Sprint 5 — Node resolution + K-shortest paths
    • Resolve natural language destination query → destination node
      using MiniLM sentence embeddings + fuzzy string fallback.
    • Run Yen's K-shortest paths algorithm (k=15) from start to destination.

Sprint 6 — Composite quality scoring + path selection
    • Score each candidate path on three criteria:
        safety_score    — weighted average of edge safety_scores
        shore_score     — fraction of edges that are shore_linable
        landmark_score  — density of landmark nodes along the path
    • Select the shortest path among those in the top quality tier.
    • Return a PathResult with the winning path and runner-up alternatives.

NOTE: This file is a Sprint 5–6 stub.
      All interfaces are fully defined; implement the TODO sections
      in Sprint 5–6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from shared.types import FloorGraph, NavigationEdge, NavigationNode, Point2D


# ── Result types ─────────────────────────────────────────────────────────────

@dataclass
class PathStep:
    """One step in a navigation path."""
    from_node:    NavigationNode
    to_node:      NavigationNode
    edge:         NavigationEdge
    bearing:      float           # degrees clockwise from north (0–360)
    distance:     float           # metres for this step
    instruction:  str             # raw instruction text, e.g. "turn left, walk 10 m"


@dataclass
class PathResult:
    """
    Output of PathfindingEngine.find_path().
    Consumed by the Feedback Module to generate voice/haptic instructions.
    """
    start_node:       NavigationNode
    destination_node: NavigationNode
    query:            str                  # original user query
    steps:            List[PathStep]       = field(default_factory=list)
    total_distance:   float                = 0.0
    safety_score:     float                = 0.0
    shore_score:      float                = 0.0
    landmark_score:   float                = 0.0
    alternatives:     List["PathResult"]   = field(default_factory=list)
    found:            bool                 = True

    def summary(self) -> str:
        return (
            f"PathResult: {self.start_node.label!r} → {self.destination_node.label!r}\n"
            f"  Steps      : {len(self.steps)}\n"
            f"  Distance   : {self.total_distance:.1f} m\n"
            f"  Safety     : {self.safety_score:.2f}\n"
            f"  Shore      : {self.shore_score:.2f}\n"
            f"  Landmark   : {self.landmark_score:.2f}\n"
            f"  Alternatives: {len(self.alternatives)}\n"
        )


# ── Engine ────────────────────────────────────────────────────────────────────

class PathfindingEngine:
    """
    Finds optimal paths through a FloorGraph for visually impaired navigation.

    Parameters
    ----------
    graph : FloorGraph
        Navigation graph from MapExtractionPipeline.run().
    k : int
        Number of candidate paths to generate (Yen's algorithm). Default 15.
    embedding_model : str
        Sentence-transformers model name for destination resolution.
        Default 'all-MiniLM-L6-v2'.
    """

    def __init__(self,
                 graph:           FloorGraph,
                 k:               int = 15,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.graph           = graph
        self.k               = k
        self.embedding_model = embedding_model
        self._embeddings_ready = False

    # ── Public API ────────────────────────────────────────────────────────────

    def find_path(self,
                  start_node_id: str,
                  destination_query: str) -> PathResult:
        """
        Find the optimal path from start_node to a destination described
        in natural language (e.g. "I want to go to Nike", "food court", "exit").

        Parameters
        ----------
        start_node_id     : str  — node_id of current user position
        destination_query : str  — free-text destination description

        Returns
        -------
        PathResult with steps, scores, and alternatives.
        """
        start = self.graph.node(start_node_id)
        if start is None:
            raise ValueError(f"Start node '{start_node_id}' not found in graph.")

        # Sprint 5: resolve destination
        dest = self._resolve_destination(destination_query)
        if dest is None:
            return PathResult(
                start_node       = start,
                destination_node = start,
                query            = destination_query,
                found            = False,
            )

        # Sprint 5: generate K candidate paths
        candidates = self._k_shortest_paths(start.node_id, dest.node_id)

        # Sprint 6: score and select
        scored     = self._score_paths(candidates)
        best, rest = self._select_path(scored)

        return best

    # ── Sprint 5 stubs ────────────────────────────────────────────────────────

    def _resolve_destination(self, query: str) -> Optional[NavigationNode]:
        """
        [Sprint 5 — TODO]
        Steps:
          1. Run Named Entity Extraction on the query to get a location name.
          2. Encode the name with sentence-transformers MiniLM.
          3. Compute cosine similarity against pre-encoded node labels.
          4. If top similarity < threshold, fall back to fuzzy string match
             (rapidfuzz or difflib).
          5. Return the best-matching NavigationNode.

        Stub: simple substring match for now.
        """
        q_lower = query.lower()
        best_node: Optional[NavigationNode] = None
        best_score = 0.0

        for node in self.graph.nodes:
            label_lower = node.label.lower()
            # Count matching words
            words = q_lower.split()
            score = sum(1 for w in words if w in label_lower) / max(len(words), 1)
            # Boost zone centroid nodes with matching category tag
            category = node.tags.get("category", "")
            if category and category in q_lower:
                score += 0.5
            if score > best_score:
                best_score = score
                best_node  = node

        return best_node if best_score > 0 else None

    def _k_shortest_paths(self,
                          source_id: str,
                          target_id: str) -> List[List[NavigationEdge]]:
        """
        [Sprint 5 — TODO]
        Implement Yen's K-shortest loopless paths algorithm using networkx.
        Return up to self.k paths as lists of NavigationEdges.

        Stub: returns a single greedy path.
        """
        # Build adjacency for greedy stub
        adj: Dict[str, List[Tuple[float, str, NavigationEdge]]] = {}
        for e in self.graph.edges:
            adj.setdefault(e.source_id, []).append((e.distance, e.target_id, e))
            adj.setdefault(e.target_id, []).append((e.distance, e.source_id, e))

        # Dijkstra stub
        import heapq
        dist   = {source_id: 0.0}
        prev:   Dict[str, Optional[Tuple[str, NavigationEdge]]] = {source_id: None}
        heap   = [(0.0, source_id)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float("inf")):
                continue
            for w, v, edge in adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v]  = (u, edge)
                    heapq.heappush(heap, (nd, v))

        # Reconstruct
        if target_id not in dist:
            return []
        path_edges = []
        cur = target_id
        while prev.get(cur) is not None:
            u, edge = prev[cur]
            path_edges.append(edge)
            cur = u
        path_edges.reverse()
        return [path_edges] if path_edges else []

    # ── Sprint 6 stubs ────────────────────────────────────────────────────────

    def _score_paths(self,
                     candidates: List[List[NavigationEdge]]
                     ) -> List[Tuple[float, float, float, List[NavigationEdge]]]:
        """
        [Sprint 6 — TODO]
        For each candidate path compute (safety_score, shore_score, landmark_score).
        Return list of (safety, shore, landmark, edges) tuples.
        """
        scored = []
        for path in candidates:
            if not path:
                continue
            safety    = _avg(e.safety_score   for e in path)
            shore     = _avg(float(e.shore_linable) for e in path)
            landmark  = _avg(e.landmark_score  for e in path)
            scored.append((safety, shore, landmark, path))
        return scored

    def _select_path(self,
                     scored: List[Tuple[float, float, float, List[NavigationEdge]]]
                     ) -> Tuple[PathResult, List[PathResult]]:
        """
        [Sprint 6 — TODO]
        Select the shortest path among those in the top quality tier.
        Current stub: pick the first (and only) scored path.
        """
        if not scored:
            # Return an empty result — caller handles found=False
            dummy_node = self.graph.nodes[0] if self.graph.nodes else NavigationNode(
                "UNKNOWN", "Unknown", (0.0, 0.0), "zone_centroid", None
            )
            result = PathResult(
                start_node       = dummy_node,
                destination_node = dummy_node,
                query            = "",
                found            = False,
            )
            return result, []

        safety, shore, landmark, best_edges = scored[0]
        steps  = self._edges_to_steps(best_edges)
        total_d = sum(e.distance for e in best_edges)

        start_node = self.graph.node(best_edges[0].source_id) if best_edges else self.graph.nodes[0]
        dest_node  = self.graph.node(best_edges[-1].target_id) if best_edges else self.graph.nodes[-1]

        best = PathResult(
            start_node       = start_node,
            destination_node = dest_node,
            query            = "",
            steps            = steps,
            total_distance   = round(total_d, 2),
            safety_score     = round(safety, 3),
            shore_score      = round(shore, 3),
            landmark_score   = round(landmark, 3),
            found            = True,
        )
        return best, []

    # ── Instruction generation ────────────────────────────────────────────────

    def _edges_to_steps(self,
                        edges: List[NavigationEdge]) -> List[PathStep]:
        """
        Convert a sequence of edges to PathStep objects with bearings and
        raw instruction text.  The Feedback Module converts these to voice.
        """
        steps = []
        prev_bearing: Optional[float] = None

        for edge in edges:
            src  = self.graph.node(edge.source_id)
            tgt  = self.graph.node(edge.target_id)
            if src is None or tgt is None:
                continue

            bearing = _bearing(src.position, tgt.position)
            turn    = _turn_description(prev_bearing, bearing)
            dist_str = f"{edge.distance:.0f} m" if edge.distance >= 1 else \
                       f"{edge.distance*100:.0f} cm"

            landmark_nearby = tgt.node_type in ("elevator", "escalator",
                                                 "stair", "landmark", "door")
            landmark_hint = f" You will reach {tgt.label}." if landmark_nearby else ""

            instruction = f"{turn} Walk {dist_str}.{landmark_hint}"

            steps.append(PathStep(
                from_node   = src,
                to_node     = tgt,
                edge        = edge,
                bearing     = bearing,
                distance    = edge.distance,
                instruction = instruction.strip(),
            ))
            prev_bearing = bearing

        return steps


# ── Geometry utilities ────────────────────────────────────────────────────────

def _bearing(a: Point2D, b: Point2D) -> float:
    """Clockwise bearing from north in degrees (0–360)."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    angle = math.degrees(math.atan2(dx, dy))
    return angle % 360


def _turn_description(prev_bearing: Optional[float], new_bearing: float) -> str:
    if prev_bearing is None:
        return "Start walking."
    delta = (new_bearing - prev_bearing + 360) % 360
    if delta < 20 or delta > 340:
        return "Continue straight."
    elif delta < 90:
        return "Bear right."
    elif delta < 180:
        return "Turn right."
    elif delta < 270:
        return "Turn left."
    else:
        return "Bear left."


def _avg(iterable) -> float:
    items = list(iterable)
    return sum(items) / len(items) if items else 0.0
