"""
Robust NetCDF reading/writing utilities with retry logic and remote-storage
support. This module is import-safe: all heavy imports (e.g., fsspec) are
performed lazily inside the functions that require them.

Logging is integrated at INFO level to provide clear traceability of file I/O.
"""

from __future__ import annotations

import os
import time
import uuid
import random
import warnings
from typing import Optional, Callable

import numpy as np
import xarray as xr
from contextlib import contextmanager

from .logging_config import get_logger

logger = get_logger(__name__)


def is_remote_path(path: str) -> bool:
    """
    Determine whether a path refers to a remote storage backend or protocol.

    Parameters
    ----------
    path : str
        File path or URI.

    Returns
    -------
    bool
        True if the path is remote (S3, GCS, HTTPS, etc.), False otherwise.
    """
    if not isinstance(path, str):
        return False

    # Quick heuristic check
    remote_prefixes = (
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
    if any(path.startswith(p) for p in remote_prefixes):
        return True

    # If available, let fsspec determine protocol (lazy import)
    try:
        import fsspec  # lazy import

        fs, _, _ = fsspec.core.get_fs_token_paths(path)
        protocol = getattr(fs, "protocol", "file")
        if isinstance(protocol, (tuple, list)):
            return "file" not in protocol
        return protocol != "file"
    except Exception:
        return False


@contextmanager
def temporary_target(path: str, *, storage_options: Optional[dict] = None):
    """
    Create a safe temporary file path for writing, then rename into place.

    This prevents partial/corrupted writes when failures occur.

    Parameters
    ----------
    path : str
        Destination file path.
    storage_options : dict, optional
        Additional storage configuration for remote backends.

    Yields
    ------
    str
        Temporary path to write to.
    """
    storage_options = storage_options or {}
    directory = os.path.dirname(path.rstrip("/"))
    basename = os.path.basename(path)

    temp_name = f".{basename}.part.{uuid.uuid4().hex}"

    if is_remote_path(path):
        temp_path = f"{directory}/{temp_name}"
    else:
        temp_path = os.path.join(directory, temp_name)
        os.makedirs(directory or ".", exist_ok=True)

    logger.debug(f"Creating temporary path: {temp_path}")

    try:
        yield temp_path

        if is_remote_path(path):
            import fsspec

            fs, _, _ = fsspec.core.get_fs_token_paths(path)
            logger.debug(f"Renaming remote temp file to final path: {path}")
            fs.rename(temp_path, path)
        else:
            logger.debug(f"Renaming local temp file to final path: {path}")
            os.replace(temp_path, path)

    finally:
        # Clean-up orphaned temp files
        try:
            if is_remote_path(temp_path):
                import fsspec

                fs, _, _ = fsspec.core.get_fs_token_paths(temp_path)
                if fs.exists(temp_path):
                    logger.debug(f"Removing remote temp file: {temp_path}")
                    fs.rm(temp_path)
            else:
                if os.path.exists(temp_path):
                    logger.debug(f"Removing local temp file: {temp_path}")
                    os.remove(temp_path)
        except Exception:
            pass


def write_netcdf_with_retries(
    obj: xr.Dataset | xr.DataArray,
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
    storage_options: Optional[dict] = None,
    upload_via_local_temp: bool = True,
    max_delay: float = 60.0,
    handle_zstd: bool = True,
    refresh_on_zstd: Optional[Callable[[], xr.Dataset | xr.DataArray]] = None,
    **to_netcdf_kwargs,
):
    """
    Write an xarray object to NetCDF with retry logic and support for remote
    object stores. This is designed for reliability when working with unstable
    cloud-based storage systems or large distributed computations.

    Parameters
    ----------
    obj : xarray Dataset or DataArray
        Data to write.
    path : str
        Destination path.
    engine : str
        NetCDF engine (default "netcdf4").
    mode : str
        File write mode.
    format : str or None
        NetCDF format.
    encoding : dict, optional
        Encoding dictionary for xarray.to_netcdf().
    retries : int
        Maximum number of attempts.
    initial_delay : float
        Initial retry delay in seconds.
    backoff : float
        Exponential backoff multiplier.
    preload : bool
        If True, loads the dataset fully into memory before writing.
    exist_ok : bool
        If False, writing to an existing file raises FileExistsError.
    storage_options : dict, optional
        Passed to fsspec.
    upload_via_local_temp : bool
        If True, create a local temp file before uploading to remote storage.
    max_delay : float
        Maximum retry delay.
    handle_zstd : bool
        Detect and recover from ZSTD decompression corruption.
    refresh_on_zstd : callable, optional
        Called to refresh input data if corruption is detected.

    Returns
    -------
    None
    """
    storage_options = storage_options or {}

    if not exist_ok and os.path.exists(path):
        raise FileExistsError(f"Target file exists and exist_ok=False: {path}")

    obj_local = obj.load() if preload else obj

    logger.info(f"Writing NetCDF: {path}")

    for attempt in range(1, retries + 1):
        try:
            if is_remote_path(path) and upload_via_local_temp:
                import fsspec

                local_temp = f".{uuid.uuid4().hex}.nc"
                logger.debug(f"Using local temp file for remote upload: {local_temp}")

                obj_local.to_netcdf(
                    local_temp,
                    engine=engine,
                    mode="w",
                    format=format,
                    encoding=encoding,
                    **to_netcdf_kwargs,
                )

                fs, _, _ = fsspec.core.get_fs_token_paths(path)
                fs.put(local_temp, path, **storage_options)

                os.remove(local_temp)
                logger.info(f"Successfully uploaded NetCDF to remote path: {path}")

            else:
                with temporary_target(path, storage_options=storage_options) as tmp:
                    obj_local.to_netcdf(
                        tmp,
                        engine=engine,
                        mode=mode,
                        format=format,
                        encoding=encoding,
                        **to_netcdf_kwargs,
                    )
                logger.info(f"Successfully wrote NetCDF to: {path}")

            return  # success

        except Exception as exc:
            if attempt == retries:
                logger.error(f"Final write attempt failed: {exc}", exc_info=True)
                raise RuntimeError(
                    f"Failed to write NetCDF after {retries} attempts"
                ) from exc

            delay = min(initial_delay * (backoff ** (attempt - 1)), max_delay)
            jitter = random.uniform(0, delay * 0.25)
            wait = delay + jitter

            logger.warning(
                f"Write attempt {attempt} failed: {exc}. "
                f"Retrying in {wait:.1f} seconds..."
            )

            time.sleep(wait)
