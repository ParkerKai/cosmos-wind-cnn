"""
High-level workflow for downloading, subsetting, and exporting HRRR subsets
as monthly or water-year NetCDF files using the Herbie package.

Import-safe: heavy/optional dependencies (cfgrib/eccodes, wgrib2) are imported
lazily inside the main function or when needed.

Logging is integrated at INFO level to clearly track each processing stage.

References:
- Herbie data sources & priority: https://herbie.readthedocs.io/... (see module docs)
- Herbie search/inventory (regex): https://herbie.readthedocs.io/... (see module docs)
- HRRR gallery/lat-lon 2-D curvilinear: https://herbie.readthedocs.io/... (see module docs)
"""

from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
import xarray as xr
import logging
from typing import Sequence

from .utils_general import wrap_to_180, parse_date, water_year_slice
from .file_io import write_netcdf_with_retries

logger = logging.getLogger("DownloadLogger")

# -----------------------------
# Light dependencies at import
# -----------------------------
from herbie import Herbie  # core

# Tenacity for robust retries (transient errors)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)
from requests.exceptions import ConnectionError, ReadTimeout


# -----------------------------
# Retry wrappers
# -----------------------------
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(exp_initial=1, exp_max=20),
    retry=retry_if_exception_type((ConnectionError, ReadTimeout)),
    reraise=True,
)
def _xarray_with_retry(H: Herbie, search: str) -> xr.Dataset:
    """Open GRIB subset into xarray with retry on transient network errors."""
    return H.xarray(search)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(exp_initial=1, exp_max=20),
    retry=retry_if_exception_type((ConnectionError, ReadTimeout)),
    reraise=True,
)
def _download_with_retry(H: Herbie, search: str):
    """Download full or subset GRIB with retry."""
    return H.download(search)


# -----------------------------
# Helpers
# -----------------------------
def _subset_bbox_curvilinear(
    ds: xr.Dataset, lon_min: float, lon_max: float, lat_min: float, lat_max: float
) -> xr.Dataset:
    """
    Subset an HRRR curvilinear grid to a lon/lat bounding box.
    Finds rectangular index bounds from a boolean mask and slices y,x.
    """
    lon = ds["longitude"]
    lat = ds["latitude"]
    mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
    idx = np.where(mask.values)
    if idx[0].size == 0:
        logger.warning("No grid points found within the specified spatial bounds.")
        # Return empty slice to preserve coords and schema
        return ds.isel(x=slice(0, 0), y=slice(0, 0))
    y_min, y_max = int(np.min(idx[0])), int(np.max(idx[0]))
    x_min, x_max = int(np.min(idx[1])), int(np.max(idx[1]))
    return ds.isel(y=slice(y_min, y_max + 1), x=slice(x_min, x_max + 1))


def _build_search_regex(search_terms: Sequence[str]) -> str:
    """
    Combine multiple raw regex search terms into one alternation group that
    Herbie understands (e.g., r'(?:TMP:2 m|SPFH:2 m|[UV]GRD:10 m)').
    Each term should be a raw string that matches the 'search_this' inventory.
    """
    # Strip leading/trailing whitespace, ensure raw-like terms remain intact.
    cleaned = [term.strip() for term in search_terms]
    # Avoid accidental regex anchors mismatches; caller provides them explicitly.
    return r"(?:{})".format("|".join(cleaned))


# -----------------------------
# Main function
# -----------------------------
def hrrr_download_subset(
    output_dir: str,
    lim_lon: list[float],
    lim_lat: list[float],
    variables: list[str] = (  # raw regex terms compatible with Herbie inventory
        [r":TMP:2 m"]  # 2-m temperature
    ),
    time_lims: list = [pd.Timestamp(2014, 7, 1), pd.Timestamp(2026, 8, 1)],
    product: str = "sfc",
    fxx: int = 0,  # analysis; for forecast hours, set e.g., 0..18 or 0..48 per cycle
    priority: list[str] = None,
    save_cache_dir: str | None = None,
    pre_crop_with_wgrib2: bool = False,
    write_calendar: str = "monthly",  # "monthly" or "water_year"
    compression_level: int = 4,
    netcdf_engine: str = "netcdf4",
) -> None:
    """
    Download and process a spatial & temporal subset of HRRR via Herbie.

    This function performs:
        1) Herbie object creation per timestep (using source priority & local cache).
        2) Regex-based GRIB message subsetting for selected variables.
        3) Optional regional crop with wgrib2 for performance.
        4) Spatial subset on HRRR's curvilinear grid.
        5) Export to monthly or water-year NetCDF with compression.

    Parameters
    ----------
    output_dir : str
        Directory for output NetCDF files.
    lim_lon : [west, east]
        Longitude bounds (will be wrapped to [0, 360) for HRRR).
    lim_lat : [south, north]
        Latitude bounds.
    variables : list[str]
        Raw regex strings matching GRIB inventory "search_this" (e.g., r":TMP:2 m").
    time_lims : [start, end]
        Datetime bounds (inclusive start, inclusive end).
    product : str
        HRRR product, e.g., "sfc", "prs", "nat", "subh".
    fxx : int
        Forecast lead time in hours (0=analysis).
    priority : list[str]
        Herbie data source search order. If None, defaults to open mirrors first.
    save_cache_dir : str | None
        Local directory for GRIB cache (Herbie's save_dir).
    pre_crop_with_wgrib2 : bool
        If True and wgrib2 is available, crop GRIB to bbox before reading.
    write_calendar : {"monthly", "water_year"}
        Output split pattern.
    compression_level : int
        NetCDF zlib compression level (0–9).
    netcdf_engine : str
        Engine for writing NetCDF ("netcdf4" recommended).

    Returns
    -------
    None
    """
    # -----------------------------
    # Setup
    # -----------------------------
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Starting HRRR subset extraction.")
    logger.info(f"Output directory: {output_dir}")

    # Normalize longitudes: HRRR lat/lon arrays are in [0, 360)
    wrapped = wrap_to_180(np.array(lim_lon))  # reuse helper if you prefer [-180, 180]

    # Convert to [0, 360) for HRRR grids
    def _to_360(lon):
        return lon + 360 if lon < 0 else lon

    lon_bounds = np.array([_to_360(wrapped[0]), _to_360(wrapped[1])], dtype=float)
    lat_bounds = np.array(lim_lat, dtype=float)

    # Default source priority: prefer open mirrors before NOMADS (rate limits)
    # See Herbie docs for data sources & archive characteristics.
    if priority is None:
        priority = ["aws", "google", "azure", "nomads", "pando", "pando2"]

    # Optional local GRIB cache
    if save_cache_dir is None:
        save_cache_dir = os.path.join(output_dir, "grib_cache")
    os.makedirs(save_cache_dir, exist_ok=True)
    logger.info(f"Local GRIB cache: {save_cache_dir}")

    # Build time vector
    t_vec = pd.date_range(start=time_lims[0], end=time_lims[1], freq="H")

    # Multi-variable search regex
    search_regex = _build_search_regex(variables)
    logger.info(f"Search regex: {search_regex}")

    # Heavy/optional imports done lazily
    HAVE_WGRIB2 = False
    if pre_crop_with_wgrib2:
        try:
            from herbie import wgrib2  # lazy

            HAVE_WGRIB2 = True
            logger.info("wgrib2 detected; will pre-crop GRIBs to bbox.")
        except Exception:
            HAVE_WGRIB2 = False
            logger.warning("wgrib2 not available. Proceeding without pre-crop.")

    # -----------------------------
    # Process times
    # -----------------------------
    ds_list = []
    template_ds = None

    for t in t_vec:
        logger.info(f"Processing time: {t}")

        # Create Herbie object; Herbie searches sources in given priority
        H = Herbie(
            str(t),
            model="hrrr",
            product=product,
            fxx=fxx,
            priority=priority,
            save_dir=save_cache_dir,
            verbose=False,
        )

        # If requested & available, pre-crop GRIB by bbox, then open locally
        if HAVE_WGRIB2:
            # Download only the messages we need (partial GRIB via byte-range)
            try:
                local_subset_grib = _download_with_retry(H, search_regex)
            except Exception as e:
                logger.error(f"Download failed at {t}: {e}")
                local_subset_grib = None

            if local_subset_grib is not None:
                try:
                    subset_file = wgrib2.region(
                        str(local_subset_grib),
                        (lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]),
                        name="bbox",
                    )
                    # Read locally via cfgrib for speed
                    ds_t = xr.open_dataset(str(subset_file), engine="cfgrib")

                except Exception as e:
                    logger.error(f"wgrib2 crop/open failed at {t}: {e}")
                    ds_t = None
            else:
                ds_t = None

            if ds_t is None:
                # Fallback to direct xarray read via Herbie
                try:
                    ds_t = _xarray_with_retry(H, search_regex)
                except Exception as e:
                    logger.error(f"xarray read failed at {t}: {e}")
                    ds_t = None

        else:
            # No pre-crop: read subset directly with Herbie (regex)
            try:
                ds_t = _xarray_with_retry(H, search_regex)
            except Exception as e:
                logger.error(f"xarray read failed at {t}: {e}")
                ds_t = None

        # Handle missing by filling NaNs using template slice
        if ds_t is None:
            if template_ds is not None:
                ds_nan = template_ds.copy(deep=True)
                for var in ds_nan.data_vars:
                    ds_nan[var].data = np.full(
                        ds_nan[var].shape, np.nan, dtype=ds_nan[var].dtype
                    )
                ds_nan = ds_nan.assign_coords(
                    time=("time", [np.datetime64(pd.Timestamp(t))])
                )
                ds_list.append(ds_nan)
            else:
                logger.warning(f"No template yet; skipping time {t}.")
            continue

        # Spatial subset (curvilinear grid)
        ds_t = _subset_bbox_curvilinear(
            ds_t, lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]
        )

        # Save template for NaN fill
        if (
            template_ds is None
            and ds_t.sizes.get("x", 0) > 0
            and ds_t.sizes.get("y", 0) > 0
        ):
            template_ds = ds_t.isel(x=slice(None), y=slice(None))

        # Standardize time coordinate
        ds_t = ds_t.assign_coords(time=("time", [np.datetime64(pd.Timestamp(t))]))

        ds_list.append(ds_t)

    if len(ds_list) == 0:
        logger.warning("No datasets collected; nothing to export.")
        return

    # -----------------------------
    # Concatenate & export
    # -----------------------------
    ds_all = xr.concat(
        ds_list, dim="time", data_vars="minimal", coords="minimal", compat="override"
    )

    # NetCDF compression
    encoding = {
        var: {"zlib": True, "complevel": compression_level} for var in ds_all.data_vars
    }

    if write_calendar == "monthly":
        logger.info("Exporting monthly files...")
        ym = np.vstack((t_vec.year, t_vec.month))
        chunks = np.unique(ym, axis=1)
        for i in range(chunks.shape[1] - 1):
            start = f"{chunks[0, i]:04d}-{chunks[1, i]:02d}-01 00:00"
            end = f"{chunks[0, i + 1]:04d}-{chunks[1, i + 1]:02d}-01 00:00"
            ds_month = ds_all.sel(time=slice(start, end))
            if ds_month.sizes.get("time", 0) == 0:
                continue
            out_file = os.path.join(
                output_dir, f"HRRR_{chunks[0, i]:04d}{chunks[1, i]:02d}.nc"
            )
            write_netcdf_with_retries(
                ds_month,
                out_file,
                engine=netcdf_engine,
                exist_ok=True,
                retries=6,
                initial_delay=2.0,
                backoff=1.7,
                encoding=encoding,
            )
            logger.info(f"File Saved: {out_file}")

    elif write_calendar == "water_year":
        logger.info("Exporting water-year files...")
        years = np.unique(parse_date(ds_all["time"].values, "year"))
        for yr in years:
            wy = water_year_slice(ds_all, yr)
            if wy.sizes.get("time", 0) == 0:
                continue
            out_file = os.path.join(output_dir, f"HRRR_WY{yr}.nc")
            write_netcdf_with_retries(
                wy,
                out_file,
                engine=netcdf_engine,
                exist_ok=True,
                retries=6,
                initial_delay=2.0,
                backoff=1.7,
                encoding=encoding,
            )
            logger.info(f"File Saved: {out_file}")
    else:
        raise ValueError("write_calendar must be 'monthly' or 'water_year'.")

    logger.info("HRRR subset extraction complete.")
