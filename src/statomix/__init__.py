"""Statomix: modular human-in-the-loop statistical workflows."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

try:
    __version__ = version("statomix")
except PackageNotFoundError:
    __version__ = "0+unknown"

_MODULE_EXPORTS = {
    "analysis",
    "analytics",
    "core",
    "curation",
    "dataset",
    "pipelines",
    "project",
    "reporting",
    "storage",
    "workflows",
}


def __getattr__(name: str) -> Any:
    if name in _MODULE_EXPORTS:
        return import_module(f"statomix.{name}")
    if name == "Project":
        from statomix.workflows.project import Project

        return Project
    if name == "Dataset":
        from statomix.workflows.dataset import Dataset

        return Dataset
    raise AttributeError(f"module 'statomix' has no attribute {name!r}")


__all__ = [
    "Dataset",
    "Project",
    "__version__",
    *_MODULE_EXPORTS,
]
