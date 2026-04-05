"""
pathfinding/__init__.py
-----------------------
Public API for the Pathfinding module.

Usage
-----
    from pathfinding import PathfindingEngine

    engine = PathfindingEngine(floor_graph)
    result = engine.find_path("ZONE-ENTRANCE", "I want to go to Nike")
"""

from pathfinding.engine import PathfindingEngine

__all__ = ["PathfindingEngine"]
