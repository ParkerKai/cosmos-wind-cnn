"""
Reprojection utilities for xarray datasets.

This module provides:
- A robust time-aware reprojection function for (time, y, x) gridded data.
- A helper to compute geographic latitude/longitude arrays from projected grids.

All heavy operations and imports (e.g., scipy.interpolate) occur lazily inside
functions to keep this module import-safe.

Logging is integrated at INFO level to track reprojection steps.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from typing import Optional

from pyproj import CRS, Transformer
from .logging_config import get_logger

logger = get_logger(__name__)


def reproject_dataset(
    ds: xr.Dataset,
    source_crs: CRS,
    target_crs: CRS,
    resolution: float = 1000.0,
    method: str = "linear",
    variables: Optional[list[str]] = None,
) -> xr.Dataset:
    """
    Reproject a dataset with grid-aligned (time, y, x) variables into a new CRS.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing variables on a regular or curvilinear grid.
    source_crs : pyproj.CRS
        CRS describing (x, y) coordinates of the input dataset.
    target_crs : pyproj.CRS
        CRS for the reprojected output grid.
    resolution : float, default 1000
        Output grid resolution in units of the target CRS.
    method : {"linear", "nearest", "cubic"}
        Interpolation method passed to scipy.interpolate.griddata.
    variables : list[str], optional
        Variables to reproject. Defaults to all (time, y, x) arrays.

    Returns
    -------
    xarray.Dataset
        Reprojected dataset containing variables in the target CRS.
    """
    logger.info("Starting dataset reprojection...")
    logger.info(f"Interpolation method: {method}")
    logger.debug(f"Using output resolution: {resolution}")

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    # Extract input grid
    x = ds["x"].values
    y = ds["y"].values

    if x.ndim == 1 and y.ndim == 1:
        x2d, y2d = np.meshgrid(x, y)
        logger.debug("Input grid is regular (1D x and y).")
    else:
        x2d, y2d = x, y
        logger.debug("Input grid is curvilinear (2D x and y).")

    # Transform coordinates
    xp, yp = transformer.transform(x2d, y2d)
    pts = np.column_stack((xp.ravel(), yp.ravel()))

    # Build target grid
    x_min, x_max = np.nanmin(xp), np.nanmax(xp)
    y_min, y_max = np.nanmin(yp), np.nanmax(yp)

    xi = np.arange(x_min, x_max + resolution, resolution)
    yi = np.arange(y_min, y_max + resolution, resolution)
    xi2d, yi2d = np.meshgrid(xi, yi)

    time_vals = ds["time"].values

    # Lazy import for heavy computation
    from scipy.interpolate import griddata

    def _reproject_var(da: xr.DataArray):
        logger.info(f"Reprojecting variable: {da.name}")
        frames = []
        for t in time_vals:
            values = da.sel(time=t).values.ravel()
            result = griddata(pts, values, (xi2d, yi2d), method=method)
            frames.append(result)

        data = np.stack(frames, axis=0)
        return xr.DataArray(
            data,
            dims=("time", "y", "x"),
            coords={"time": time_vals, "y": yi, "x": xi},
            attrs=da.attrs,
        )

    selected = variables or list(ds.data_vars)
    out_vars = {}

    for name in selected:
        da = ds[name]
        if set(da.dims) == {"time", "y", "x"}:
            out_vars[name] = _reproject_var(da.transpose("time", "y", "x"))
        else:
            logger.debug(f"Skipping non-gridded variable: {name}")

    out_ds = xr.Dataset(out_vars, coords={"time": time_vals, "x": xi, "y": yi})
    out_ds.attrs["target_crs_epsg"] = target_crs.to_epsg()
    out_ds.attrs["target_crs_wkt"] = target_crs.to_wkt()

    logger.info("Reprojection complete.")
    return out_ds


def add_lat_lon_from_crs(
    ds: xr.Dataset,
    source_crs: CRS | str | int,
    target_crs: CRS | str | int,
    x_name: str = "x",
    y_name: str = "y",
    lat_name: str = "lat",
    lon_name: str = "lon",
    set_coords: bool = False,
    overwrite: bool = False,
) -> xr.Dataset:
    """
    Compute latitude and longitude arrays for a projected x/y grid.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing x and y coordinate variables.
    source_crs : CRS or str or int
        CRS describing the input x/y coordinates.
    target_crs : CRS or str or int
        CRS describing desired lat/lon output (usually EPSG:4326).
    x_name, y_name : str
        Names of x and y coordinate variables in the dataset.
    lat_name, lon_name : str
        Output latitude and longitude variable names.
    set_coords : bool
        If True, register lat/lon as dataset coordinates.
    overwrite : bool
        If False and lat/lon already exist, raise an error.

    Returns
    -------
    xarray.Dataset
        New dataset including lat lon.
    """
    logger.info("Adding latitude and longitude arrays from CRS definitions.")

    src = CRS.from_user_input(source_crs)
    tgt = CRS.from_user_input(target_crs)

    x = ds[x_name]
    y = ds[y_name]

    if x.ndim == 1 and y.ndim == 1:
        x2d, y2d = np.meshgrid(x.values, y.values)
        dims = (y.dims[0], x.dims[0])
        logger.debug("Detected regular (1D) input grid.")
    else:
        x2d, y2d = x.values, y.values
        dims = x.dims
        logger.debug("Detected curvilinear (2D) input grid.")

    transformer = Transformer.from_crs(src, tgt, always_xy=True)

    lon2d, lat2d = transformer.transform(x2d, y2d)

    if not overwrite and (lat_name in ds or lon_name in ds):
        raise ValueError(
            f"Dataset already contains '{lat_name}' or '{lon_name}'. "
            "Pass overwrite=True to replace."
        )

    ds_out = ds.copy()
    ds_out[lat_name] = xr.DataArray(lon2d, dims=dims)
    ds_out[lon_name] = xr.DataArray(lat2d, dims=dims)

    if set_coords:
        ds_out = ds_out.set_coords([lat_name, lon_name])

    logger.info("Lat/lon coordinate computation complete.")
    return ds_out
