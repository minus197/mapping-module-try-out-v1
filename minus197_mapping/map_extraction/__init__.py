"""
map_extraction/__init__.py
--------------------------
Public API for the Map Extraction module.

Single-floor usage:
    from map_extraction import MapExtractionPipeline
    pipeline = MapExtractionPipeline("building.ifc", floor_label="L1")
    graph    = pipeline.run()          # → FloorGraph
    pipeline.save("data/outputs/")

Multi-floor usage:
    from map_extraction import MapExtractionPipeline, InterFloorLinker
    pipeline = MapExtractionPipeline.multi_floor(
        floors=[("L1.ifc","L1",{}), ("L2.ifc","L2",{})],
        building_name="My Mall",
    )
    building = pipeline.run_multi()    # → BuildingGraph
"""

from map_extraction.pipeline import MapExtractionPipeline
from map_extraction.inter_floor_linker import InterFloorLinker

__all__ = ["MapExtractionPipeline", "InterFloorLinker"]
