"""
map_extraction/__init__.py
--------------------------
Public API for the Map Extraction module.

Single-floor usage:
    from map_extraction import MapExtractionPipeline
    pipeline = MapExtractionPipeline("building.ifc", floor_label="L1")
    graph    = pipeline.run()          # -> FloorGraph
    pipeline.save("data/outputs/")
    # Produces: _graph.json, _sfm.json, _occupancy.json

Multi-floor usage:
    from map_extraction import MapExtractionPipeline, InterFloorLinker
    pipeline = MapExtractionPipeline.multi_floor(
        floors=[("L1.ifc","L1",{}), ("L2.ifc","L2",{})],
        building_name="My Mall",
    )
    building = pipeline.run_multi()    # -> BuildingGraph
    pipeline.save_multi("data/outputs/")

Standalone occupancy grid (if you already have a SemanticFloorMap):
    from map_extraction import OccupancyGridExporter
    OccupancyGridExporter(sfm).build().save("data/outputs/floor_occupancy.json")
"""

from map_extraction.pipeline import MapExtractionPipeline
from map_extraction.inter_floor_linker import InterFloorLinker
from map_extraction.occupancy_grid import OccupancyGridExporter

__all__ = [
    "MapExtractionPipeline",
    "InterFloorLinker",
    "OccupancyGridExporter",
]
