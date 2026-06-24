"""
pathfinding/node_resolver.py  —  WP5  (Stage 1)
-------------------------------------------------
Resolves a user's true world-coordinate position (wx, wy) to the nearest
navigable NavigationNode that is not separated from the user by a wall.

Algorithm
---------
1.  Build a KD-tree over all *navigable* nodes (junction | door | stair |
    lift | elevator | escalator).  Zone centroids are excluded — they are
    destinations, not waypoints a user can stand at.
2.  Query the k nearest neighbours (default k=8).
3.  Walk the neighbours in distance order; return the first one whose
    straight-line segment to (wx, wy) does NOT cross a wall according to
    the injected wall_checker callable.
4.  If every neighbour within k is wall-blocked, fall back to the
    geometrically nearest node (neighbour[0]) — documented behaviour, not
    a crash.

Parameters
----------
graph        : FloorGraph   — source of nodes; must have rebuild_index() called
wall_checker : callable(x1, y1, x2, y2) -> bool
               Returns True when the segment (x1,y1)→(x2,y2) crosses a wall.
               Injected so the resolver can be tested without perception_map.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree

from shared.types import FloorGraph, NavigationNode, Point2D

# Node types that are valid start-resolution candidates
_NAVIGABLE = {"junction", "door", "stair", "lift", "elevator", "escalator"}

WallChecker = Callable[[float, float, float, float], bool]


class StartNodeResolver:
    """
    Resolves a user's world position to the nearest wall-clear navigable node.

    Parameters
    ----------
    graph        : FloorGraph
    wall_checker : WallChecker
    """

    def __init__(self, graph: FloorGraph, wall_checker: WallChecker) -> None:
        self._wall_checker = wall_checker
        self._nodes: List[NavigationNode] = [
            n for n in graph.nodes if n.node_type in _NAVIGABLE
        ]
        if not self._nodes:
            raise ValueError("Graph contains no navigable nodes for StartNodeResolver.")

        coords = np.array([n.position for n in self._nodes], dtype=float)
        self._tree = KDTree(coords)

    def resolve(
        self,
        wx: float,
        wy: float,
        k: int = 8,
    ) -> Tuple[NavigationNode, Point2D]:
        """
        Find the nearest wall-clear navigable node to the user position.

        Parameters
        ----------
        wx, wy : user world coordinates (metres)
        k      : number of neighbours to check before falling back

        Returns
        -------
        (NavigationNode, (wx, wy))
            The resolved start node and the user's unchanged true coordinates.
        """
        k_actual = min(k, len(self._nodes))
        _, indices = self._tree.query([wx, wy], k=k_actual)

        # indices may be a scalar when k=1 — normalise to list
        if np.isscalar(indices):
            indices = [int(indices)]
        else:
            indices = list(indices)

        for idx in indices:
            candidate = self._nodes[idx]
            cx, cy = candidate.position
            if not self._wall_checker(wx, wy, cx, cy):
                return candidate, (wx, wy)

        # Fallback: all k neighbours are wall-blocked — return geometric nearest
        nearest = self._nodes[indices[0]]
        return nearest, (wx, wy)
