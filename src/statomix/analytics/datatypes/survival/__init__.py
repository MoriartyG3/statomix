"""Kaplan--Meier, log-rank, Cox, and threshold survival analyses."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BinaryClassSurv": (
        "statomix.analytics.datatypes.survival.binary_class_surv",
        "BinaryClassSurv",
    ),
    "MaximallySelectedLogRank": (
        "statomix.analytics.datatypes.survival.thresholds.mpv.maxstat",
        "MaximallySelectedLogRank",
    ),
    "MaxstatResult": (
        "statomix.analytics.datatypes.survival.thresholds.mpv.maxstat",
        "MaxstatResult",
    ),
    "MultiClassSurv": (
        "statomix.analytics.datatypes.survival.multi_class_surv",
        "MultiClassSurv",
    ),
    "PreparedSurvivalData": (
        "statomix.analytics.datatypes.survival.data",
        "PreparedSurvivalData",
    ),
    "SingleClassSurv": (
        "statomix.analytics.datatypes.survival.single_class_surv",
        "SingleClassSurv",
    ),
    "ThresholdScan": (
        "statomix.analytics.datatypes.survival.thresholds.mpv.scan",
        "ThresholdScan",
    ),
    "prepare_survival_data": (
        "statomix.analytics.datatypes.survival.data",
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


__all__ = tuple(_EXPORTS)
