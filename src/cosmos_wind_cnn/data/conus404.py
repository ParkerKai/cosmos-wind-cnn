"""
High-level workflow for downloading, subsetting, optionally reprojecting,
and exporting CONUS404 dataset subsets as annual water-year NetCDF files.

This module is import-safe: heavy dependencies (intake, dask.distributed,
cartopy, metpy, pyproj) are imported lazily inside the main function.

Logging is integrated at INFO level to clearly track each processing stage.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from .utils_general import wrap_to_180, parse_date, water_year_slice
from .reprojection import add_lat_lon_from_crs, reproject_dataset
from .file_io import write_netcdf_with_retries

import logging
logger = logging.getLogger('DownloadLogger')

def conus404_download_subset(
    output_dir: str,
    lim_lon: list[float],
    lim_lat: list[float],
    dataset: str = "conus404-hourly-ba-osn",
    variables: list[str] = ["T2", "TD2", "U10", "V10", "PSFC"],
    time_lims: list = [
        pd.Timestamp(1979, 10, 1),
        pd.Timestamp(2022, 10, 1)],
    reproject: bool = False,
    reproject_crs: int = 26910,
    n_workers: int = 2,
):
    """
    Download and process a spatial & temporal subset of the CONUS404 dataset.

    This function performs:
        1. Lazy initialization of a Dask LocalCluster.
        2. Loading the CONUS404 dataset via the HyTEST Intake catalog.
        3. Adding lat/lon coordinates if missing.
        4. Spatial and temporal subsetting.
        5. Optional reprojection to a new CRS.
        6. Exporting one NetCDF file per water year.

    Parameters
    ----------
    output_dir : str
        Directory for output NetCDF files.
    lim_lon : [west, east]
        Longitude bounds (will be wrapped to [-180, 180]).
    lim_lat : [south, north]
        Latitude bounds.
    dataset : str
        HyTEST dataset name.
    variables : list[str], optional
        Variables to extract.
    time_lims : [start, end], optional
        Datetime bounds.
    reproject : bool
        Whether to reproject the subset before exporting.
    reproject_crs : int
        EPSG target CRS (default: 26910).
    n_workers : int
        Number of Dask workers.

    Returns
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Starting CONUS404 subset extraction.")
    logger.info(f"Output directory: {output_dir}")

    # Lazy imports for heavy dependencies.
    import intake
    from distributed import Client, LocalCluster
    import metpy
    import cartopy.crs as ccrs  # Needed behind the scenes by metpy.parse_cf
    from pyproj import CRS

    # Special-case variable list for BA dataset
    if dataset == "conus404-hourly-ba-osn":
        variables = ["RAINRATE", "T2D"]

    # -----------------------------
    # 1. Start Dask LocalCluster
    # -----------------------------
    logger.info(f"Initializing Dask LocalCluster (workers={n_workers})...")
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        processes=True,
        silence_logs=False,
    )
    client = Client(cluster)
    logger.info(f"Dask dashboard available at: {cluster.dashboard_link}")

    # -----------------------------
    # 2. Load CONUS404 dataset
    # -----------------------------
    logger.info("Loading HyTEST Intake catalog...")
    cat = intake.open_catalog(
        "https://raw.githubusercontent.com/hytest-org/hytest/main/dataset_catalog/hytest_intake_catalog.yml"
    )
    conus_cat = cat["conus404-catalog"]

    logger.info(f"Loading dataset: {dataset}")
    ds = conus_cat[dataset].to_dask().metpy.parse_cf()

    # -----------------------------
    # 3. Ensure lat/lon present and remove duplicate timestamps
    # -----------------------------
    if "lon" not in ds:
        logger.info("Computing lat/lon coordinates from CRS...")
        src = CRS.from_wkt(ds["crs"].attrs["crs_wkt"])
        tgt = CRS.from_epsg(4326)
        ds = add_lat_lon_from_crs(ds, src, tgt)


    # Get rid of duplicate timestamps
    ds = ds.sortby("time")
    _, unique_index = np.unique(ds["time"], return_index=True)
    ds = ds.isel(time=unique_index)

    # -----------------------------
    # 4. Spatial/temporal subset
    # -----------------------------
    logger.info("Performing spatial subset...")
    lon_vals = ds["lon"].values
    lat_vals = ds["lat"].values

    wrapped_lons = wrap_to_180(np.array(lim_lon))

    mask = (
        (lon_vals >= wrapped_lons[0])
        & (lon_vals <= wrapped_lons[1])
        & (lat_vals >= lim_lat[0])
        & (lat_vals <= lim_lat[1])
    )

    idx = np.where(mask)
    if idx[0].size == 0:
        logger.warning("No grid points found within the specified spatial bounds.")

    logger.info("Extracting spatial slice...")
    var_list = variables + ["lat", "lon"]

    ds_sub = (
        ds[var_list]
        .isel(
            x=slice(np.min(idx[1]), np.max(idx[1])),
            y=slice(np.min(idx[0]), np.max(idx[0])),
        )
        .sel(time=slice(time_lims[0], time_lims[1]))
    )

    # -----------------------------
    # 5. Optional reprojection
    # -----------------------------
    if reproject:
        logger.info(f"Reprojecting dataset to EPSG:{reproject_crs}...")
        src = CRS.from_wkt(ds["crs"].attrs["crs_wkt"])
        tgt = CRS.from_epsg(reproject_crs)
        resolution = 1000 if dataset == "conus404-hourly-ba-osn" else 4000

        ds_sub = reproject_dataset(
            ds_sub,
            source_crs=src,
            target_crs=tgt,
            resolution=resolution,
            method="linear",
        )

    # -----------------------------
    # 6. Export water years
    # -----------------------------
    logger.info("Exporting water-year files...")

    years = np.unique(parse_date(ds_sub["time"].values, "year"))

    for yr in years:
        logger.info(f"Processing water year {yr}...")
        wy = water_year_slice(ds_sub, yr)

        out_file = os.path.join(output_dir, f"CONUS404_WY{yr}.nc")

        write_netcdf_with_retries(
            wy,
            out_file,
            engine="netcdf4",
            exist_ok=True,
            retries=6,
            initial_delay=2.0,
            backoff=1.7,
        )

    # -----------------------------
    # 7. Cleanup
    # -----------------------------
    logger.info("Shutting down Dask cluster...")
    client.close()
    cluster.close()

    logger.info("CONUS404 subset extraction complete.")
