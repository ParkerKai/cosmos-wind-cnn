# -*- coding: utf-8 -*-

"""
Data Download utilities

Author: Kai Parker
Email: kaparker@usgs.gov
Updated: 2026-05-11
"""


import cdsapi
import os
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed



import os

os.environ["USE_PYGEOS"] = "0"

import fsspec
import xarray as xr

import intake
import metpy
import cartopy.crs as ccrs
import numpy as np

import pandas as pd
import time
import uuid
import warnings
from contextlib import contextmanager
import random
from typing import Optional, Dict, Callable
from dask.distributed import Client, LocalCluster
from pyproj import CRS, Transformer
from scipy.interpolate import griddata

from typing import Union


# Figure out the fsspec situation on this environment
try:
    import fsspec

    HAS_FSSPEC = True

except ImportError:
    HAS_FSSPEC = False


#######################################################################
# General Helper Functions 
#######################################################################

def wrapTo180(lon):
    lon = np.mod(lon - 180.0, 360.0) - 180.0
    return lon


def wrapTo360(lon):
    lon = lon % 360
    return lon

def parseDate(date, type):
    # parses numpy datetime64
    if type == "year":
        out = date.astype("datetime64[Y]").astype(int) + 1970
    elif type == "month":
        out = date.astype("datetime64[M]").astype(int) % 12 + 1
    elif type == "days":
        out = date - date.astype("datetime64[M]") + 1
    else:
        print("user type not found")
        out = []

    return out

def water_year_slice(da: xr.DataArray | xr.Dataset, yr: int):
    start = pd.Timestamp(yr, 10, 1, 0, 0).to_datetime64()
    end = pd.Timestamp(yr + 1, 9, 30, 23, 59, 59).to_datetime64()
    return da.sel(time=slice(start, end))


def expand_year_months(years, months):
    """Expand year and month ranges into lists of ['YYYY', 'MM'] pairs."""
    year_list = [str(y) for y in range(years[0], years[1] + 1)]
    month_list = [f"{m:02d}" for m in range(months[0], months[1] + 1)]

    return [[yy, mm] for yy in year_list for mm in month_list]


def build_day_list(days_option):
    """Return list of day strings."""
    if days_option == "all":
        return [f"{d:02d}" for d in range(1, 32)]
    else:
        return [f"{int(d):02d}" for d in days_option]


def build_hour_list(hours_option):
    """Return list of hour strings."""
    if hours_option == "all":
        return [f"{h:02d}:00" for h in range(24)]
    else:
        return [f"{int(h):02d}:00" for h in hours_option]


#######################################################################
# Functions for ERA5 download
#######################################################################

def retrieve_single_month(dataset, client_args, request, date, dir_out):
    """Download a single month of ERA5 data."""
    yy, mm = date

    print(f"Requesting year {yy}, month {mm}")

    # Each thread gets its own CDS client for safety
    client = cdsapi.Client(**client_args)

    # Build target filename
    target = os.path.join(dir_out, f"ERA5_PNW_{yy}_{mm}.nc")

    # Update request
    request_local = request.copy()
    request_local.update({"year": yy, "month": mm})

    # Execute download
    client.retrieve(dataset, request_local, target)

    return f"Completed {yy}-{mm}: {target}"


def download_era5(
    dir_out,
    area_lims,
    years,
    months,
    variables,
    dataset="reanalysis-era5-land",
    max_threads=10,
    days="all",
    hours="all",
    cds_url=None,
    cds_key=None,
):
    """
    Parallelized CDS API downloader for ERA5/ERA5-Land datasets.

    Parameters
    ----------
    dir_out : str
        Output directory.
    area_lims : list [N, W, S, E]
        Geographic box.
    years : [start_year, end_year]
        Inclusive year range.
    months : [start_month, end_month]
        Inclusive month range.
    variables : list
        ERA5 variable names.
    dataset : str
        CDS dataset name.
    max_threads : int
        Number of parallel requests.
    days : "all" or list of days
    hours : "all" or list of hours
    cds_url : str
        Optional custom CDS API URL.
    cds_key : str
        Optional CDS API key/token.
    """

    # Ensure directory exists
    os.makedirs(dir_out, exist_ok=True)

    # Build request template
    request = {
        "data_format": "netcdf",
        "download_format": "unarchived",
        "variable": variables,
        "area": area_lims,
        "day": build_day_list(days),
        "time": build_hour_list(hours),
    }

    # Thread-safe CDS client args
    client_args = {}
    if cds_url:
        client_args["url"] = cds_url
    if cds_key:
        client_args["key"] = cds_key

    # Expand year–month combinations
    date_list = expand_year_months(years, months)

    # Launch parallel downloads
    print(
        f"Submitting {len(date_list)} monthly downloads using {max_threads} threads...\n"
    )

    results = []
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [
            executor.submit(
                retrieve_single_month, dataset, client_args, request, date, dir_out
            )
            for date in date_list
        ]

        for f in as_completed(futures):
            try:
                result = f.result()
                print(result)
                results.append(result)
            except Exception as e:
                print("Download failed:", e)

    print("\nAll downloads completed.")
    return results


# Example usage:
# if __name__ == "__main__":
#     download_era5(
#         dir_out=r"D:\Kai\ERA5\PNW_Meteo",
#         area_lims=[49, -126.5, 41.5, -122],
#         years=[2024, 2026],
#         months=[1, 12],
#         variables=[
#             "10m_u_component_of_wind",
#             "10m_v_component_of_wind",
#             "2m_temperature",
#             "2m_dewpoint_temperature",
#             "mean_sea_level_pressure",
#             "total_precipitation",
#         ],
#         dataset="reanalysis-era5-land",
#         max_threads=10,
#     )



#######################################################################
# Functions for Conus404 download
#######################################################################


def _is_remote_path(path: str) -> bool:
    if not isinstance(path, str):
        return False
    if not HAS_FSSPEC:
        return any(
            path.startswith(pfx)
            for pfx in (
                "s3://",
                "gs://",
                "abfs://",
                "adl://",
                "az://",
                "oss://",
                "hdfs://",
                "http://",
                "https://",
                "r2://",
                "sftp://",
                "ftp://",
            )
        )
    try:
        fs, _, _ = fsspec.core.get_fs_token_paths(path)
        protocol = getattr(fs, "protocol", "file")
        if isinstance(protocol, (list, tuple)):
            return "file" not in protocol
        return protocol != "file"
    except Exception:
        return any(
            path.startswith(pfx)
            for pfx in (
                "s3://",
                "gs://",
                "abfs://",
                "adl://",
                "az://",
                "oss://",
                "hdfs://",
                "http://",
                "https://",
                "r2://",
                "sftp://",
                "ftp://",
            )
        )


@contextmanager
def _temporary_target(path: str, *, storage_options: Optional[Dict] = None):
    storage_options = storage_options or {}
    dirname = os.path.dirname(path.rstrip("/"))
    base = os.path.basename(path)
    temp_name = f".{base}.part.{uuid.uuid4().hex}"
    temp_path = (
        f"{dirname}/{temp_name}"
        if _is_remote_path(path)
        else os.path.join(dirname, temp_name)
    )

    if not _is_remote_path(path):
        os.makedirs(dirname or ".", exist_ok=True)

    try:
        yield temp_path
        if _is_remote_path(path) and HAS_FSSPEC:
            fs, _, _ = fsspec.core.get_fs_token_paths(path)
            fs.rename(temp_path, path)
            try:
                fs.invalidate_cache(path)
            except Exception:
                pass
        else:
            os.replace(temp_path, path)
    finally:
        try:
            if _is_remote_path(temp_path) and HAS_FSSPEC:
                fs, _, _ = fsspec.core.get_fs_token_paths(temp_path)
                if fs.exists(temp_path):
                    fs.rm(temp_path, recursive=False)
            else:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception:
            pass


def _is_zstd_corruption_error(e: Exception) -> bool:
    msg = str(e)
    return (
        "Zstd decompression error" in msg
        or "Data corruption detected" in msg
        or ("zstd" in msg.lower() and "corrupt" in msg.lower())
    )


def write_netcdf_with_retries(
    ds_or_da: xr.Dataset | xr.DataArray,
    path: str,
    *,
    engine: str = "netcdf4",
    mode: str = "w",
    format: Optional[str] = "NETCDF4",
    encoding: Optional[dict] = None,
    retries: int = 5,
    initial_delay: float = 2.0,
    backoff: float = 1.7,
    preload: bool = False,
    exist_ok: bool = True,
    storage_options: Optional[Dict] = None,
    upload_via_local_temp: bool = True,
    max_delay: float = 60.0,
    handle_zstd: bool = True,
    refresh_on_zstd: Optional[Callable[[], xr.Dataset | xr.DataArray]] = None,
    **to_netcdf_kwargs,
):
    """
    Robust writer for xarray objects with retries, safe temp files, and built-in Zstd recovery.

    Parameters
    ----------
    ds_or_da : xr.Dataset | xr.DataArray
    path : str
    engine : str
    mode : str
    format : str | None
    encoding : dict | None
    retries : int
    initial_delay : float
    backoff : float
    preload : bool
        If True, `.load()` before writing.
    exist_ok : bool
    storage_options : dict | None
        Passed to fsspec.
    upload_via_local_temp : bool
        If True, write locally then PUT to final for object stores.
    max_delay : float
    handle_zstd : bool
        If True, detect Zstd errors and refresh data before continuing retries.
    refresh_on_zstd : Callable[[], xr.Dataset | xr.DataArray] | None
        Called when a Zstd error occurs to re-materialize the object to write.
        If None, falls back to `obj.load()`.
    """
    storage_options = storage_options or {}

    # Pre-load if requested
    obj = ds_or_da.load() if preload else ds_or_da

    # Existence check
    try:
        already_exists = False
        if _is_remote_path(path) and HAS_FSSPEC:
            fs, _, _ = fsspec.core.get_fs_token_paths(path)
            already_exists = fs.exists(path)
        else:
            already_exists = os.path.exists(path)
        if already_exists and not exist_ok and mode == "w":
            raise FileExistsError(f"Target already exists: {path}")
    except Exception:
        pass

    if mode == "a" and _is_remote_path(path):
        warnings.warn(
            "Append mode on remote storage requires download+local append+re-upload; "
            "this function rewrites from local temp.",
            RuntimeWarning,
        )

    delay = initial_delay

    for attempt in range(1, retries + 1):
        try:
            if _is_remote_path(path) and HAS_FSSPEC and upload_via_local_temp:
                # Strategy A: local temp → single PUT
                dirname = os.path.dirname(path.rstrip("/"))
                base = os.path.basename(path)
                local_temp = os.path.join(
                    os.getcwd(), f".{base}.part.{uuid.uuid4().hex}"
                )
                try:
                    obj.to_netcdf(
                        local_temp,
                        engine=engine,
                        mode="w" if mode != "a" else "a",
                        format=format,
                        encoding=encoding,
                        **to_netcdf_kwargs,
                    )
                    fs, _, _ = fsspec.core.get_fs_token_paths(path)
                    fs.put(local_temp, path, **storage_options)
                    try:
                        fs.invalidate_cache(path)
                    except Exception:
                        pass
                finally:
                    try:
                        if os.path.exists(local_temp):
                            os.remove(local_temp)
                    except Exception:
                        pass

            else:
                # Strategy B: remote temp key + rename
                with _temporary_target(
                    path, storage_options=storage_options
                ) as temp_path:
                    if _is_remote_path(temp_path) and HAS_FSSPEC:
                        fs, _, _ = fsspec.core.get_fs_token_paths(temp_path)
                        try:
                            fobj = fs.open(
                                temp_path, "wb", autocommit=False, **storage_options
                            )
                        except TypeError:
                            fobj = fs.open(temp_path, "wb", **storage_options)

                        with fobj as fh:
                            obj.to_netcdf(
                                fh,
                                engine=engine,
                                mode="w" if mode != "a" else "a",
                                format=format,
                                encoding=encoding,
                                **to_netcdf_kwargs,
                            )
                            if hasattr(fh, "commit"):
                                fh.commit()
                    else:
                        os.makedirs(os.path.dirname(temp_path) or ".", exist_ok=True)
                        obj.to_netcdf(
                            temp_path,
                            engine=engine,
                            mode=mode,
                            format=format,
                            encoding=encoding,
                            **to_netcdf_kwargs,
                        )

            # Quick verification for remote
            if _is_remote_path(path) and HAS_FSSPEC:
                fs, _, _ = fsspec.core.get_fs_token_paths(path)
                size = fs.size(path)
                if size is None or size <= 0:
                    raise IOError(f"Wrote zero-byte object: {path}")

            return  # Success

        except (OSError, TimeoutError, IOError, RuntimeError) as exc:
            # Zstd-specific recovery
            if handle_zstd and _is_zstd_corruption_error(exc):
                warnings.warn(
                    f"ZSTD corruption detected on attempt {attempt}/{retries}: {exc}. "
                    f"Refreshing data and retrying…",
                    RuntimeWarning,
                )
                try:
                    # Prefer caller-supplied refresh; else force load
                    if refresh_on_zstd is not None:
                        obj = refresh_on_zstd()
                    else:
                        # Force materialization to avoid flaky remote reads
                        obj = obj.load()
                except Exception as refresh_exc:
                    warnings.warn(
                        f"Refresh after ZSTD failed: {refresh_exc}", RuntimeWarning
                    )

                # Continue to retry logic below

            # Transient retry path
            if attempt < retries:
                jitter = random.uniform(0, delay * 0.25)
                wait = min(delay + jitter, max_delay)
                warnings.warn(
                    f"Write failed on attempt {attempt}/{retries}: {exc}. "
                    f"Retrying in {wait:.1f}s…",
                    RuntimeWarning,
                )
                time.sleep(wait)
                delay = min(delay * backoff, max_delay)
                continue

            # Exhausted retries
            raise RuntimeError(
                f"Failed to write NetCDF after {retries} attempts: {exc}"
            ) from exc

        except Exception:
            # Non-transient error—surface immediately
            raise


def reproject_dataset_with_time(
    ds: xr.Dataset,
    source_crs: CRS,
    target_crs: CRS,
    resolution: float = 1000.0,
    method: str = "linear",
    variables: list[str] | None = None,
) -> xr.Dataset:
    """
    Reprojects all dataset variables with dimensions (time, y, x) onto a new grid in the target CRS.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing coords 'time', 'x', 'y'. 'x' and 'y' may be 1D or 2D.
    source_crs : pyproj.CRS
        Source CRS (e.g., CRS.from_wkt(ds['crs'].attrs['crs_wkt'])).
    target_crs : pyproj.CRS
        Target CRS (e.g., CRS.from_epsg(26910)).
    resolution : float, optional
        Spacing of the output grid (units of target CRS). Default is 1000.
    method : {"linear", "nearest", "cubic"}, optional
        Interpolation method passed to scipy.interpolate.griddata. Default "linear".
        Note: "cubic" requires more points and is slower; "nearest" fills holes better.
    variables : list[str] | None, optional
        If provided, only these variables will be reprojected. Otherwise all eligible variables.

    Returns
    -------
    xr.Dataset
        Reprojected dataset with coords ('time', 'y', 'x') in the target CRS. Variables not matching
        (time, y, x) are omitted.
    """
    # Build transformer
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    # Get original coordinates
    x = ds["x"].values
    y = ds["y"].values
    if x.ndim == 1 and y.ndim == 1:
        x2d, y2d = np.meshgrid(x, y)
    else:
        # Assume already 2D curvilinear grid
        x2d, y2d = x, y

    # Transform grid to target CRS
    x_proj, y_proj = transformer.transform(x2d, y2d)

    # Create points array for griddata (flattened)
    points = np.column_stack((x_proj.ravel(), y_proj.ravel()))

    # Define target grid bounds (include max bound by adding resolution)
    x_min, x_max = np.nanmin(x_proj), np.nanmax(x_proj)
    y_min, y_max = np.nanmin(y_proj), np.nanmax(y_proj)
    xi = np.arange(x_min, x_max + resolution, resolution)
    yi = np.arange(y_min, y_max + resolution, resolution)
    xi2d, yi2d = np.meshgrid(xi, yi)

    # Prepare time coordinates
    time_coords = ds["time"].values

    # Helper to reproject a single (time, y, x) DataArray
    def _reproject_da(da: xr.DataArray) -> xr.DataArray:
        interpolated_list = []
        # Iterate per time slice
        for t in time_coords:
            values = da.sel(time=t).values.ravel()
            interpolated = griddata(points, values, (xi2d, yi2d), method=method)
            interpolated_list.append(interpolated)
        # Stack back into 3D array
        data = np.stack(interpolated_list, axis=0)
        return xr.DataArray(
            data,
            dims=("time", "y", "x"),
            coords={"time": time_coords, "x": xi, "y": yi},
            attrs=da.attrs,
        )

    # Select which variables to process
    if variables is None:
        var_items = list(ds.data_vars.items())
    else:
        var_items = [(name, ds[name]) for name in variables if name in ds.data_vars]

    # Process all eligible vars
    out_vars = {}
    skipped = []
    for name, da in var_items:
        # Check dims are exactly (time, y, x) regardless of order
        required_dims = {"time", "y", "x"}
        if set(da.dims) == required_dims:
            # Put in canonical order before processing
            da_tyx = da.transpose("time", "y", "x")
            out_vars[name] = _reproject_da(da_tyx)
        else:
            skipped.append(name)

    # Build output dataset
    out_ds = xr.Dataset(out_vars, coords={"time": time_coords, "x": xi, "y": yi})

    # Attach CRS metadata for the target grid (if source ds has CRS attrs)
    # This is optional but helpful if you store WKT or EPSG
    try:
        out_ds.attrs.update(ds.attrs)
    except Exception:
        pass

    # Store target CRS in an attribute block
    out_ds.attrs["target_crs_epsg"] = getattr(target_crs, "to_epsg", lambda: None)()
    out_ds.attrs["target_crs_wkt"] = target_crs.to_wkt()

    # Inform about skipped variables
    if skipped:
        print(
            f"Skipped {len(skipped)} variable(s) without dims exactly ('time','y','x'): {skipped}"
        )

    return out_ds

def add_lat_lon_from_crs(
    ds: xr.Dataset,
    source_crs: Union[CRS, str, int],
    target_crs: Union[CRS, str, int],
    x_name: str = "x",
    y_name: str = "y",
    lat_name: str = "lat",
    lon_name: str = "lon",
    set_as_coords: bool = False,
    overwrite: bool = False,
) -> xr.Dataset:
    """
    Add latitude/longitude variables to an xarray.Dataset based on projected x/y coordinates,
    using a user-supplied source CRS.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing x and y coordinate variables. These may be 1D (regular grid) or 2D
        (curvilinear grid).
    source_crs : Union[pyproj.CRS, str, int]
        Source CRS describing the x/y coordinates. Accepts:
        - a pyproj.CRS object (recommended),
        - a WKT string (e.g., ds["crs"].attrs["crs_wkt"]),
        - an EPSG integer (e.g., 3857), or
        - any input supported by pyproj.CRS.from_user_input.
    target_crs : Union[pyproj.CRS, str, int]
        Source CRS describing the x/y coordinates. Accepts:
        - a pyproj.CRS object (recommended),
        - a WKT string (e.g., ds["crs"].attrs["crs_wkt"]),
        - an EPSG integer (e.g., 3857), or
        - any input supported by pyproj.CRS.from_user_input.
    x_name : str, default "x"
        Name of the x coordinate variable in the dataset.
    y_name : str, default "y"
        Name of the y coordinate variable in the dataset.
    lat_name : str, default "lat"
        Name of the latitude variable to create.
    lon_name : str, default "lon"
        Name of the longitude variable to create.
    set_as_coords : bool, default False
        If True, add lat/lon as coordinates.
    overwrite : bool, default False
        If False and lat/lon already exist, raises a ValueError. If True, replaces them.

    Returns
    -------
    xr.Dataset
        Dataset with added lat/lon variables having dims (y, x).

    Notes
    -----
    - Uses pyproj with `always_xy=True` to treat input order as (x, y).
    - Works with both 1D and 2D x/y coordinates.
    - For very large dask-backed datasets, consider using `xr.apply_ufunc` for lazy transformation.
    """
    # --- Validate presence of x/y ---
    if x_name not in ds or y_name not in ds:
        raise KeyError(f"Dataset must contain '{x_name}' and '{y_name}' variables.")

    x_var = ds[x_name]
    y_var = ds[y_name]

    # --- Normalize source/target CRS ---
    src_crs = CRS.from_user_input(source_crs)
    tgt_crs = CRS.from_user_input(target_crs)

    # --- Prepare 2D coordinate arrays and dims ---
    if x_var.ndim == 1 and y_var.ndim == 1:
        x_dim = x_var.dims[0]
        y_dim = y_var.dims[0]
        # Meshgrid with 'xy' indexing -> X shape (ny, nx), Y shape (ny, nx)
        x2d, y2d = np.meshgrid(x_var.values, y_var.values, indexing="xy")
        dims = (y_dim, x_dim)
    elif x_var.ndim == 2 and y_var.ndim == 2:
        x2d = np.asarray(x_var.values)
        y2d = np.asarray(y_var.values)
        # Prefer y_var dims if available; otherwise x_var dims
        dims = y_var.dims if len(y_var.dims) == 2 else x_var.dims
    else:
        raise ValueError(
            f"Inconsistent dimensionality: x.ndim={x_var.ndim}, y.ndim={y_var.ndim}. "
            "Both must be 1D or both 2D."
        )

    # --- Transform to lon/lat ---
    transformer = Transformer.from_crs(src_crs, tgt_crs, always_xy=True)
    lon2d, lat2d = transformer.transform(x2d, y2d)

    # --- Build DataArray with attributes ---
    lat_da = xr.DataArray(
        lat2d,
        dims=dims,
        name=lat_name,
        attrs={
            "standard_name": "latitude",
            "long_name": "latitude",
            "units": "degrees_north",
            "crs": f"EPSG:{tgt_crs.to_epsg()}",
        },
    )
    lon_da = xr.DataArray(
        lon2d,
        dims=dims,
        name=lon_name,
        attrs={
            "standard_name": "longitude",
            "long_name": "longitude",
            "units": "degrees_east",
            "crs": f"EPSG:{tgt_crs.to_epsg()}",
        },
    )

    # --- Insert into dataset ---
    if not overwrite and (lat_name in ds or lon_name in ds):
        raise ValueError(
            f"Dataset already contains '{lat_name}' or '{lon_name}'. "
            "Set overwrite=True to replace."
        )

    ds_out = ds.copy()
    ds_out[lat_name] = lat_da
    ds_out[lon_name] = lon_da

    if set_as_coords:
        ds_out = ds_out.set_coords([lat_name, lon_name])

    return ds_out



def download_conus404_subset(
    dir_out,
    lim_lon,
    lim_lat,
    dataset="conus404-hourly-ba-osn",
    var=None,
    t_lims=None,
    reproject=False,
    reproject_crs=26910,
    n_workers=2,
):
    """
    Download, subset, optionally reproject, and export CONUS404 data as annual
    water-year NetCDF files.

    This function automates the full workflow for retrieving CONUS404 data from
    the HyTest intake catalog, spatially/temporally subsetting it, optionally
    reprojecting it to a new CRS, and exporting the results as annual (water year)
    NetCDF files. It also sets up a Dask cluster for parallelized data access and
    processing.

    Parameters
    ----------
    dir_out : str
        Directory path where the output NetCDF files will be written.

    lim_lon : array-like of length 2
        Longitude boundaries of the region of interest given as
        [west_lon, east_lon]. Will be wrapped to -180:180.

    lim_lat : array-like of length 2
        Latitude boundaries of the region of interest given as
        [south_lat, north_lat].

    dataset : str, optional
        Name of the CONUS404 dataset to load from the HyTest intake catalog.
        Defaults to "conus404-hourly-ba-osn".

    var : list of str, optional
        List of variable names to retrieve from the dataset.  
        If None, defaults to a standard meteorological set.  
        If dataset == "conus404-hourly-ba-osn", this list is overridden to:
        ["RAINRATE", "T2D"].

    t_lims : list-like, optional
        Two-element sequence of time bounds. If None, defaults to:
        [1979-10-01, 2022-10-01].

    reproject : bool, optional
        Whether to reproject the data to a new coordinate system.

    reproject_crs : int or str, optional
        EPSG code for the target coordinate system (used only if reproject=True).
        Defaults to 26910 (NAD83 / UTM Zone 10N).

    n_workers : int, optional
        Number of Dask workers to start in the LocalCluster.

    Returns
    -------
    None
        The function writes NetCDF files to disk and prints progress.

    Workflow Summary
    ----------------
    1. Start a Dask LocalCluster to speed up distributed data access.
    2. Open the HyTest intake catalog and load the requested CONUS404 dataset.
    3. Add latitude/longitude coordinates if the dataset lacks them.
    4. Subset the dataset spatially based on the provided lon/lat bounds.
    5. Subset the dataset temporally based on `t_lims`.
    6. Optionally reproject to a new coordinate reference system.
    7. Slice the dataset into individual water years.
    8. Export each water year to a separate NetCDF file using robust retry logic.
    9. Cleanly shut down the Dask cluster.
    """

    import os
    import numpy as np
    import pandas as pd
    import intake
    from distributed import Client, LocalCluster
    from pyproj import CRS

    # ------------------------------------------------------------------------------
    # Input defaults: variable list and time limits
    # ------------------------------------------------------------------------------

    if var is None:
        var = [
            "T2", "TD2", "U10", "V10", "PSFC", "LANDMASK",
            "HGT", "PREC_ACC_NC", "Q2", "SMOIS", "ACSWDNB", "ACLWDNB",
        ]

    if dataset == "conus404-hourly-ba-osn":
        var = ["RAINRATE", "T2D"]

    if t_lims is None:
        t_lims = [
            pd.Timestamp(1979, 10, 1).to_datetime64(),
            pd.Timestamp(2022, 10, 1).to_datetime64(),
        ]

    # ------------------------------------------------------------------------------
    # Start Dask Cluster
    # ------------------------------------------------------------------------------

    print("Starting Dask LocalCluster...")

    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        processes=True,
        silence_logs=False,
    )
    client = Client(cluster)

    print("Dashboard:", cluster.dashboard_link)

    # ------------------------------------------------------------------------------
    # Load HyTest Intake Dataset
    # ------------------------------------------------------------------------------

    print("Opening HyTEST intake catalog...")
    cat_root = intake.open_catalog(
        "https://raw.githubusercontent.com/hytest-org/hytest/main/dataset_catalog/hytest_intake_catalog.yml"
    )
    cat = cat_root["conus404-catalog"]

    print(f"Loading dataset: {dataset}")
    ds = cat[dataset].to_dask().metpy.parse_cf()

    # ------------------------------------------------------------------------------
    # Add lat/lon if dataset requires it
    # ------------------------------------------------------------------------------

    if "lon" not in ds:
        print("Computing lat/lon from CRS definitions...")
        source_crs = CRS.from_wkt(ds["crs"].attrs["crs_wkt"])
        target_crs = CRS.from_epsg(4326)
        ds = add_lat_lon_from_crs(ds, source_crs, target_crs)

    # ------------------------------------------------------------------------------
    # Subset spatially and temporally
    # ------------------------------------------------------------------------------

    print("Subsetting spatial region...")
    lon = ds["lon"].values
    lat = ds["lat"].values

    lim_lon = wrapTo180(lim_lon)

    # Boolean region mask
    region_mask = (
        (lon >= lim_lon[0])
        & (lon <= lim_lon[1])
        & (lat >= lim_lat[0])
        & (lat <= lim_lat[1])
    )

    idx = np.where(region_mask)

    var_subset = var + ["lat", "lon"]

    da = ds[var_subset].isel(
        x=slice(np.min(idx[1]), np.max(idx[1])),
        y=slice(np.min(idx[0]), np.max(idx[0])),
    ).sel(time=slice(t_lims[0], t_lims[1]))

    # ------------------------------------------------------------------------------
    # Optional reprojection
    # ------------------------------------------------------------------------------

    if reproject:
        print("Reprojecting dataset...")

        if dataset == "conus404-hourly-ba-osn":
            res = 1000
        else:
            res = 4000

        source_crs = CRS.from_wkt(ds["crs"].attrs["crs_wkt"])
        tgt_crs = CRS.from_epsg(reproject_crs)

        da = reproject_dataset_with_time(
            da,
            source_crs=source_crs,
            target_crs=tgt_crs,
            resolution=res,
            method="linear",
        )

    # ------------------------------------------------------------------------------
    # Write out water-year files
    # ------------------------------------------------------------------------------

    print("Exporting water years...")

    crs_obj = ds[var[0]].metpy.cartopy_crs
    da = da.drop_vars("metpy_crs")
    da.attrs["projection"] = crs_obj.to_json()

    years = np.unique(parseDate(da["time"].values, "year"))

    for yr in years:
        print(f"Saving water year {yr}...")

        wy = water_year_slice(da, yr)

        write_netcdf_with_retries(
            wy,
            os.path.join(dir_out, f"conus404BA_SFbay_WY{yr}.nc"),
            engine="netcdf4",
            format="NETCDF4",
            retries=8,
            initial_delay=2.0,
            backoff=1.7,
            preload=False,
            exist_ok=True,
            handle_zstd=True,
            refresh_on_zstd=lambda: water_year_slice(da, yr),
            upload_via_local_temp=True,
        )

    # ------------------------------------------------------------------------------
    # Cleanup Dask cluster
    # ------------------------------------------------------------------------------

    print("Shutting down cluster...")
    client.close()
    client.shutdown()

    print("Finished processing CONUS404 data.")