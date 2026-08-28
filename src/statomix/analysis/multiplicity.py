"""Multiplicity corrections for families of statistical tests."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Return Holm step-down adjusted p-values.

    Holm's procedure controls the family-wise error rate under arbitrary
    dependence, which is important for overlapping threshold splits whose
    tests are strongly correlated.
    """

    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if values.size == 0:
        return values
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("p_values must be finite and in [0, 1]")

    order = np.argsort(values, kind="stable")
    ordered = values[order]
    multipliers = values.size - np.arange(values.size)
    adjusted_ordered = np.maximum.accumulate(ordered * multipliers)
    adjusted_ordered = np.clip(adjusted_ordered, 0.0, 1.0)

    adjusted = np.empty_like(adjusted_ordered)
    adjusted[order] = adjusted_ordered
    return adjusted


def holm_adjust_with_missing(p_values: Iterable[float]) -> np.ndarray:
    """Apply Holm correction to finite p-values while retaining NaNs."""

    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if valid.any():
        adjusted[valid] = holm_adjust(values[valid])
    return adjusted
