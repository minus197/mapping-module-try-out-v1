"""
shared/__init__.py
------------------
Shared types and utilities used by both map_extraction and pathfinding.
"""
from shared.types import NavigationNode, NavigationEdge, FloorGraph, Point2D

__all__ = ["NavigationNode", "NavigationEdge", "FloorGraph", "Point2D"]
