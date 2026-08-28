"""Kaplan-Meier, log-rank, and Cox proportional-hazards analyses."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BinaryClassSurv": ("statomix.analysis.survival.binary", "BinaryClassSurv"),
    "MultiClassSurv": ("statomix.analysis.survival.multi", "MultiClassSurv"),
    "PreparedSurvivalData": (
        "statomix.analysis.survival.data",
        "PreparedSurvivalData",
    ),
    "SingleClassSurv": ("statomix.analysis.survival.single", "SingleClassSurv"),
    "prepare_survival_data": (
        "statomix.analysis.survival.data",
        "prepare_survival_data",
    ),
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
    "BinaryClassSurv",
    "MultiClassSurv",
    "PreparedSurvivalData",
    "SingleClassSurv",
    "prepare_survival_data",
)
