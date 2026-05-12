"""
Data loading, downloading, reprojection, and preprocessing utilities.

Available modules:
    utils_general      - Lightweight helper utilities.
    era5_download      - ERA5 download utilities.
    file_io            - Robust NetCDF I/O with retry handling.
    reprojection       - Reprojection utilities for xarray datasets.
    conus404           - CONUS404 dataset download and processing workflow.
    preprocessing      - High-level preprocessing routines.
    dataset            - Dataset classes for 3D wind data.
    regridder          - Spatial regridding utilities.

This package exposes only the high-level preprocessing and dataset classes
via the __all__ list for public API usage. All other modules remain accessible
via direct import (e.g., `from cosmos_wind_cnn.data.utils_general import ...`).
"""

# Public API exports (unchanged)
from .preprocessing import NetCDFPreprocessor
from .dataset import WindDataset3D, WindDatasetInMemory
from .regridder import Regridder

# New modules are NOT imported here to avoid unnecessary heavy imports,
# but they remain accessible:
#
#   from cosmos_wind_cnn.data.utils_general import ...
#   from cosmos_wind_cnn.data.era5_download import ...
#   from cosmos_wind_cnn.data.file_io import ...
#   from cosmos_wind_cnn.data.reprojection import ...
#   from cosmos_wind_cnn.data.conus404 import ...
#

__all__ = [
    "NetCDFPreprocessor",
    "WindDataset3D",
    "WindDatasetInMemory",
    "Regridder",
]
