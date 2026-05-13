"""
General-purpose utilities for geospatial and time-indexed datasets.

This module contains only lightweight, import-safe helper functions with
no side effects and no heavy dependencies at import time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .logging_config import get_logger

logger = get_logger(__name__)


def wrap_to_180(lon: float | np.ndarray) -> float | np.ndarray:
    """
    Wrap longitude values into the [-180, 180) range.

    Parameters
    ----------
    lon : float or array-like
        Input longitude(s).

    Returns
    -------
    float or ndarray
        Wrapped longitude values.
    """
    logger.debug("Wrapping longitude(s) into [-180, 180) range.")
    return np.mod(lon - 180.0, 360.0) - 180.0


def wrap_to_360(lon: float | np.ndarray) -> float | np.ndarray:
    """
    Wrap longitude values into the [0, 360) range.

    Parameters
    ----------
    lon : float or array-like

    Returns
    -------
    float or ndarray
    """
    logger.debug("Wrapping longitude(s) into [0, 360) range.")
    return lon % 360


def parse_date(date: np.ndarray, component: str) -> np.ndarray:
    """
    Extract a time component from numpy.datetime64 arrays.

    Parameters
    ----------
    date : ndarray of datetime64
        Input dates.
    component : {"year", "month", "day"}
        Desired component.

    Returns
    -------
    ndarray
        Component values.

    Raises
    ------
    ValueError
        If the component is invalid.
    """
    logger.debug(f"Parsing datetime component: {component}")

    if component == "year":
        return date.astype("datetime64[Y]").astype(int) + 1970
    if component == "month":
        return date.astype("datetime64[M]").astype(int) % 12 + 1
    if component == "day":
        # day-of-month
        return date - date.astype("datetime64[M]") + 1

    raise ValueError(f"Unknown component '{component}'")


def water_year_slice(da: xr.DataArray | xr.Dataset, year: int):
    """
    Select a single water year (Oct 1 to Sep 30) from a dataset.

    Parameters
    ----------
    da : xarray object
        Dataset or DataArray with a time coordinate.
    year : int
        Starting water-year (Oct 1 of this year).

    Returns
    -------
    xarray object
        Subset over the water year.
    """
    logger.info(f"Selecting water year: {year}")

    start = pd.Timestamp(year, 10, 1)
    end = pd.Timestamp(year + 1, 9, 30, 23, 59, 59)
    return da.sel(time=slice(start, end))


def expand_year_months(
    years: tuple[int, int], months: tuple[int, int]
) -> list[list[str]]:
    """
    Expand year and month ranges into a list of [YYYY, MM] strings.

    Parameters
    ----------
    years : (start, end)
    months : (start, end)

    Returns
    -------
    list of [year_str, month_str]
    """
    logger.debug(f"Expanding year-month ranges: {years}, {months}")

    return [
        [str(y), f"{m:02d}"]
        for y in range(years[0], years[1] + 1)
        for m in range(months[0], months[1] + 1)
    ]


def build_day_list(days: str | list[int]) -> list:
    """
    Build a list of day strings (DD).

    Parameters
    ----------
    days : "all" or list of integers

    Returns
    -------
    list of strings
    """
    logger.debug(f"Building day list: {days}")

    if days == "all":
        return [f"{d:02d}" for d in range(1, 32)]
    return [f"{int(d):02d}" for d in days]


def build_hour_list(hours: str | list[int]) -> list:
    """
    Build a list of hour strings (HH:00).

    Parameters
    ----------
    hours : "all" or list of integers

    Returns
    -------
    list of strings
    """
    logger.debug(f"Building hour list: {hours}")

    if hours == "all":
        return [f"{h:02d}:00" for h in range(24)]
    return [f"{int(h):02d}:00" for h in hours]
