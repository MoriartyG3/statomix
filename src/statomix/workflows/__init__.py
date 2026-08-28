"""Composable project, dataset, Cleaner, and Analyzer workflows."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BaseDataset": ("statomix.workflows.base_dataset", "BaseDataset"),
    "Cleaner": ("statomix.workflows.cleaner", "Cleaner"),
    "Dataset": ("statomix.workflows.dataset", "Dataset"),
    "DatasetAnalyzer": ("statomix.workflows.dataset_analyzer", "Analyzer"),
    "GroupAnalyzer": ("statomix.workflows.group_analyzer", "GroupAnalyzer"),
    "Project": ("statomix.workflows.project", "Project"),
    "ProjectAnalyzer": ("statomix.workflows.project_analyzer", "Analyzer"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = (
    "BaseDataset",
    "Cleaner",
    "Dataset",
    "DatasetAnalyzer",
    "GroupAnalyzer",
    "Project",
    "ProjectAnalyzer",
)
