"""
map_extraction/__init__.py
--------------------------
Public API for the Map Extraction module.

Usage
-----
    from map_extraction import MapExtractionPipeline

    pipeline = MapExtractionPipeline("building.ifc")
    floor_graph = pipeline.run()          # -> shared.types.FloorGraph
    pipeline.save("data/outputs/graph.json")
"""

from map_extraction.pipeline import MapExtractionPipeline

__all__ = ["MapExtractionPipeline"]
