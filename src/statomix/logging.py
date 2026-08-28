"""Project-wide logging backed by Fileverse.

Domain modules import :func:`get_logger` from here instead of constructing
Fileverse loggers directly.  The small registry guard mirrors Multiomix and
prevents duplicate handlers during notebook autoreloading.
"""

from __future__ import annotations

import logging
from pathlib import Path
from threading import RLock

from fileverse.logger import Logger as FileverseLogger

_CONFIGURED_ATTRIBUTE = "_statomix_fileverse_configured"
_CONFIGURATION_LOCK = RLock()


def _qualified_name(*, name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("Logger name must be a string.")

    cleaned_name = name.strip()
    if not cleaned_name:
        raise ValueError("Logger name must not be empty.")
    if cleaned_name == "statomix" or cleaned_name.startswith("statomix."):
        return cleaned_name
    return f"statomix.{cleaned_name}"


def get_logger(
    name: str,
    *,
    level: int = logging.INFO,
    log_to_console: bool = True,
    log_to_txt: bool = False,
    log_to_csv: bool = False,
    log_folder: str | Path = "logs",
) -> logging.Logger:
    """Return one consistently configured Fileverse-backed logger."""

    qualified_name = _qualified_name(name=name)
    if isinstance(level, bool) or not isinstance(level, int):
        raise TypeError("level must be a logging level integer.")

    with _CONFIGURATION_LOCK:
        existing_logger = logging.getLogger(qualified_name)
        if getattr(existing_logger, _CONFIGURED_ATTRIBUTE, False):
            existing_logger.setLevel(level)
            return existing_logger

        fileverse_logger = FileverseLogger(
            name=qualified_name,
            log_folder=str(log_folder),
            log_to_console=log_to_console,
            log_to_txt=log_to_txt,
            log_to_csv=log_to_csv,
        ).get_logger()
        fileverse_logger.setLevel(level)
        setattr(fileverse_logger, _CONFIGURED_ATTRIBUTE, True)
        return fileverse_logger


__all__ = ["get_logger"]
