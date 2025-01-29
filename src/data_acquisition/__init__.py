"""
Data acquisition module for fetching and processing geospatial datasets.
"""

from .data_manager import DataManager
from .sources import (
    PlanetaryCompute,
    SentinelAPI,
    LandsatAPI,
    OvertureAPI,
    OSMDataAPI
)
from .processors import (
    ImageProcessor,
    VectorProcessor,
    DataFusion
)

__all__ = [
    'DataManager',
    'PlanetaryCompute',
    'SentinelAPI',
    'LandsatAPI',
    'OvertureAPI',
    'OSMDataAPI',
    'ImageProcessor',
    'VectorProcessor',
    'DataFusion'
] 