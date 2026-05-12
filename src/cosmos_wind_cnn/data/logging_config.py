"""
Central logging configuration for COSMOS Wind CNN data utilities.

This module provides a helper function that initializes loggers with a
standardized format. Logging defaults to INFO level and can optionally
write logs to a file in addition to the console.
"""

import logging
from typing import Optional


def get_logger(
    name: str,
    level: int = logging.INFO,
    propagate: bool = False,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Return a configured logger for the given module.

    Parameters
    ----------
    name : str
        Logger name (usually __name__).
    level : int, default INFO
        Logging verbosity.
    propagate : bool, default False
        Avoids double logging in notebook/Dask contexts.
    log_file : str, optional
        If provided, log messages will also be written to this file.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Avoid duplicate handlers
    if not logger.handlers:
        # Console handler
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        # Optional file handler
        if log_file is not None:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    logger.propagate = propagate
    return logger
