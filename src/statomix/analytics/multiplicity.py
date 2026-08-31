"""Multiplicity corrections for families of statistical tests."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from statsmodels.stats.multitest import multipletests


@dataclass(frozen=True, slots=True)
class CorrectionSpec:
    """Description and backend mapping for one p-value correction."""

    name: str
    controls: str
    statsmodels_method: str | None
    assumption: str


CORRECTION_REGISTRY: dict[str, CorrectionSpec] = {
    "none": CorrectionSpec(
        name="none",
        controls="none",
        statsmodels_method=None,
        assumption="Exploratory raw p-values; no multiplicity control.",
    ),
    "bonferroni": CorrectionSpec(
        name="bonferroni",
        controls="FWER",
        statsmodels_method="bonferroni",
        assumption="Valid under arbitrary dependence.",
    ),
    "holm": CorrectionSpec(
        name="holm",
        controls="FWER",
        statsmodels_method=None,
        assumption="Valid under arbitrary dependence.",
    ),
    "holm_sidak": CorrectionSpec(
        name="holm_sidak",
        controls="FWER",
        statsmodels_method="holm-sidak",
        assumption="Typically used under independence or suitable dependence.",
    ),
    "hochberg": CorrectionSpec(
        name="hochberg",
        controls="FWER",
        statsmodels_method="simes-hochberg",
        assumption="Requires independence or suitable positive dependence.",
    ),
    "fdr_bh": CorrectionSpec(
        name="fdr_bh",
        controls="FDR",
        statsmodels_method="fdr_bh",
        assumption="Valid under independence or positive regression dependence.",
    ),
    "fdr_by": CorrectionSpec(
        name="fdr_by",
        controls="FDR",
        statsmodels_method="fdr_by",
        assumption="Valid under arbitrary dependence.",
    ),
}

SUPPORTED_CORRECTIONS = tuple(CORRECTION_REGISTRY)
_P_VALUE_BOUNDARY_TOLERANCE = 1e-12


def normalize_correction_methods(
    correction_methods: str | Sequence[str] | None,
    *,
    selection_method: str = "none",
) -> tuple[str, ...]:
    """Return validated methods, always including raw and the selector."""

    if correction_methods is None:
        requested: list[str] = []
    elif isinstance(correction_methods, str):
        requested = [correction_methods]
    else:
        requested = list(correction_methods)

    ordered = ["none", *requested]
    if selection_method not in ordered:
        ordered.append(selection_method)

    normalized: list[str] = []
    for method in ordered:
        if not isinstance(method, str):
            raise TypeError(
                "correction methods must be strings, got " f"{type(method).__name__}."
            )
        if method not in CORRECTION_REGISTRY:
            raise ValueError(
                f"Unknown correction method {method!r}. Supported methods: "
                f"{list(SUPPORTED_CORRECTIONS)}"
            )
        if method not in normalized:
            normalized.append(method)

    return tuple(normalized)


def _validated_p_values(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    outside_tolerance = (values < -_P_VALUE_BOUNDARY_TOLERANCE) | (
        values > 1 + _P_VALUE_BOUNDARY_TOLERANCE
    )
    if not np.isfinite(values).all() or outside_tolerance.any():
        raise ValueError("p_values must be finite and in [0, 1]")
    return np.clip(values, 0.0, 1.0)


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Return Holm step-down adjusted p-values.

    This implementation remains local to preserve the established Statomix
    numerical contract. Holm controls familywise error under arbitrary
    dependence.
    """

    values = _validated_p_values(p_values)
    if values.size == 0:
        return values

    order = np.argsort(values, kind="stable")
    ordered = values[order]
    multipliers = values.size - np.arange(values.size)
    adjusted_ordered = np.maximum.accumulate(ordered * multipliers)
    adjusted_ordered = np.clip(adjusted_ordered, 0.0, 1.0)

    adjusted = np.empty_like(adjusted_ordered)
    adjusted[order] = adjusted_ordered
    return adjusted


def adjust_p_values(
    p_values: Iterable[float],
    *,
    method: str,
) -> np.ndarray:
    """Adjust one complete, finite p-value family with a registered method."""

    if method not in CORRECTION_REGISTRY:
        raise ValueError(
            f"Unknown correction method {method!r}. Supported methods: "
            f"{list(SUPPORTED_CORRECTIONS)}"
        )

    values = _validated_p_values(p_values)
    if values.size == 0 or method == "none":
        return values.copy()
    if method == "holm":
        return holm_adjust(values)

    statsmodels_method = CORRECTION_REGISTRY[method].statsmodels_method
    if statsmodels_method is None:
        raise RuntimeError(f"Correction method {method!r} has no implementation.")
    _, adjusted, _, _ = multipletests(
        values,
        method=statsmodels_method,
        is_sorted=False,
        returnsorted=False,
    )
    return np.asarray(adjusted, dtype=float)


def adjust_p_values_with_missing(
    p_values: Iterable[float],
    *,
    method: str,
) -> np.ndarray:
    """Adjust finite p-values while retaining non-finite positions as NaN."""

    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")

    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if finite.any():
        adjusted[finite] = adjust_p_values(values[finite], method=method)
    return adjusted


def holm_adjust_with_missing(p_values: Iterable[float]) -> np.ndarray:
    """Apply Holm correction to finite p-values while retaining NaNs."""

    return adjust_p_values_with_missing(p_values, method="holm")
