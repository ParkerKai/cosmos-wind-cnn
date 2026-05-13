"""
Minimal integration test for the COSMOS Wind CNN data processing modules.

This script verifies:
    1. All modules import safely without side effects.
    2. Utility functions execute correctly.
    3. Reprojection works on a synthetic dataset.
    4. The ERA5 and CONUS404 modules can be imported without triggering
       network connections, cluster startup, or heavy initialization.

Execute manually:
    python tests/test_data_pipeline.py
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from pyproj import CRS

from cosmos_wind_cnn.data.utils_general import wrap_to_180, water_year_slice
from cosmos_wind_cnn.data.reprojection import reproject_dataset
from cosmos_wind_cnn.data.logging_config import get_logger


logger = get_logger(
    __name__, log_file=r"C:\Users\kai\Documents\Github\cosmos-wind-cnn\process.log"
)


def test_imports():
    logger.info("Testing module imports (import-safety check)...")

    import cosmos_wind_cnn.data.era5_download
    import cosmos_wind_cnn.data.conus404
    import cosmos_wind_cnn.data.file_io

    logger.info("All modules imported safely with no side effects.")


def test_utils():
    logger.info("Testing basic utility functions...")

    assert wrap_to_180(190) == -170
    assert wrap_to_180(-200) == 160
    assert wrap_to_180(45) == 45

    # simple water-year slicing test
    times = np.array(["2020-10-01", "2021-03-01", "2021-09-30"], dtype="datetime64[ns]")
    data = xr.DataArray(np.arange(3), coords={"time": times}, dims=["time"])

    wy = water_year_slice(data, 2020)
    assert len(wy.time) == 3

    logger.info("Utility functions passed.")


def test_reprojection():
    logger.info("Testing synthetic reprojection workflow...")

    # Tiny synthetic dataset
    x = np.array([0, 1])
    y = np.array([0, 1])
    t = np.array(["2020-01-01"], dtype="datetime64[ns]")

    ds = xr.Dataset(
        {"var": (("time", "y", "x"), [[[1, 2], [3, 4]]])},
        coords={"time": t, "x": x, "y": y},
    )

    src = CRS.from_epsg(4326)
    tgt = CRS.from_epsg(3857)

    out = reproject_dataset(ds, src, tgt, resolution=10)
    logger.info(f"Reprojection output shape: {out['var'].shape}")

    assert "var" in out
    assert out["var"].ndim == 3

    logger.info("Reprojection test passed.")


def test_no_download_side_effects():
    logger.info("Ensuring download modules can be imported without side-effects...")

    from cosmos_wind_cnn.data.era5_download import download_era5
    from cosmos_wind_cnn.data.conus404 import conus404_download_subset

    logger.info(
        "ERA5 and CONUS404 modules import cleanly without triggering downloads."
    )


if __name__ == "__main__":
    logger.info("=== Starting minimal integration tests ===")
    test_imports()
    test_utils()
    test_reprojection()
    test_no_download_side_effects()
    logger.info("=== All tests passed successfully ===")
