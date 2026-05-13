"""
ERA5 download utilities using the Copernicus Climate Data Store (CDS) API.

This module is fully import-safe: all heavy imports (e.g., cdsapi) occur inside
the functions that need them. Logging is integrated at INFO level by default.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List
from datetime import datetime

from .utils_general import (
    expand_year_months,
    build_day_list,
    build_hour_list,
)
from download_data import logger



def retrieve_month(
    dataset: str,
    client_args: dict,
    base_request: dict,
    date_pair: list[str],
    output_dir: str,
):
    """
    Retrieve a single year–month ERA5 dataset from the CDS API.

    All network-related setup and API client creation occur inside this function
    to keep the module import-safe.

    Parameters
    ----------
    dataset : str
        ERA5 dataset name (e.g., "reanalysis-era5-land").
    client_args : dict
        Arguments passed to cdsapi.Client().
    base_request : dict
        Template request dictionary containing variables, area, time list, etc.
    date_pair : [year_str, month_str]
        Specific year and month to download.
    output_dir : str
        Local directory where NetCDF files will be stored.

    Returns
    -------
    str
        Confirmation message containing the completed filename.
    """
    # Lazy import for import-safety
    import cdsapi

    year, month = date_pair
    logger.info(f"Requesting ERA5 {year}-{month}")

    client = cdsapi.Client(**client_args)

    target = os.path.join(output_dir, f"ERA5_PNW_{year}_{month}.nc")
    request = base_request.copy()
    request.update({"year": year, "month": month})

    try:
        client.retrieve(dataset, request, target)
        logger.info(f"Completed ERA5 {year}-{month}: {target}")
        return f"Completed ERA5 {year}-{month}: {target}"
    except Exception as exc:
        logger.error(f"Failed ERA5 request {year}-{month}: {exc}", exc_info=True)
        raise


def download_era5(
    output_dir: str,
    area: list[float],
    years: tuple[int, int] = [1950, datetime.now().year],
    months: tuple[int, int] = [1, 12],
    variables: List[str] = ["10m_u_component_of_wind", "10m_v_component_of_wind"],
    dataset: str = "reanalysis-era5-land",
    max_threads: int = (os.cpu_count() - 1),
    days: str | list[int] = "all",
    hours: str | list[int] = "all",
    cds_url: Optional[str] = None,
    cds_key: Optional[str] = None,
):
    """
    Download multiple months of ERA5/ERA5-Land data in parallel.

    This function safely builds the request template, expands year/month ranges,
    initializes CDS API clients lazily, and uses a thread pool to download
    multiple months concurrently.

    Parameters
    ----------
    output_dir : str
        Directory to store output NetCDF files.
    area : list[float]
        Geographic bounding box [North, West, South, East].
    years : (start, end)
        Year range, inclusive.
    months : (start, end)
        Month range, inclusive.
    variables : list[str]
        ERA5 variable names.
    dataset : str
        Name of the dataset in CDS.
    max_threads : int
        Number of parallel threads.
    days : "all" or list[int]
        Days of month to request.
    hours : "all" or list[int]
        Hours of day to request.
    cds_url : str, optional
        Custom CDS URL.
    cds_key : str, optional
        Custom CDS authentication key.

    Returns
    -------
    list[str]
        A list of completion messages for each downloaded month.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output area: {area}")
    logger.info(f"Output years: {years}")
    logger.info(f"Output months: {months}")

    base_request = {
        "data_format": "netcdf",
        "download_format": "unarchived",
        "variable": variables,
        "area": area,
        "day": build_day_list(days),
        "time": build_hour_list(hours),
    }

    client_args = {}
    if cds_url:
        client_args["url"] = cds_url
    if cds_key:
        client_args["key"] = cds_key

    date_list = expand_year_months(years, months)

    logger.info(
        f"Submitting {len(date_list)} ERA5 monthly downloads using {max_threads} threads."
    )

    results: List[str] = []
    log_errors = []

    with ThreadPoolExecutor(max_workers=max_threads) as pool:
        futures = [
            pool.submit(
                retrieve_month,
                dataset,
                client_args,
                base_request,
                date,
                output_dir,
            )
            for date in date_list
        ]

        for fut in as_completed(futures):
            try:
                result = fut.result()
                results.append(result)
            except Exception as exc:
                logger.error(f"ERA5 download failed: {exc}", exc_info=True)
                log_errors.append(str(exc))

    logger.info("ERA5 batch download completed.")

    if log_errors:
        logger.warning(f"{len(log_errors)} download errors occurred.")

    return results
