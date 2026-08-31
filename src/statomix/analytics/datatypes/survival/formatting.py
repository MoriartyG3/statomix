"""Formatting and interpretation helpers for survival results."""

from __future__ import annotations

import math
from numbers import Real


def get_p_value_label(p_value: Real) -> str:
    """Format an unrounded p-value without changing its significance class."""

    value = float(p_value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"p_value must be finite and in [0, 1], got {p_value!r}")
    if value < 0.001:
        return "P<0.001"
    if round(value, 2) == 0.05 and value < 0.05:
        return f"P={value:.4f}"
    return f"P={round(value, 4)}"


def interpret_hazard_ratio(
    hazard_ratio: Real,
    baseline_group_name: str,
    other_group_name: str,
) -> str:
    """Interpret a positive hazard ratio against its reference group."""

    value = float(hazard_ratio)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"hazard_ratio must be finite and greater than 0, got {hazard_ratio!r}"
        )
    if math.isclose(value, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        return (
            "The hazard ratio is 1. This means there is no difference in "
            f"risk between the {other_group_name} group and the "
            f"{baseline_group_name} group."
        )
    if value > 1:
        risk_increase = (value - 1) * 100
        return (
            f"The hazard ratio is {value:.2f}. This means the "
            f"{other_group_name} group has {risk_increase:.0f}% higher risk "
            f"of the event compared to the {baseline_group_name} group."
        )

    risk_reduction = (1 - value) * 100
    return (
        f"The hazard ratio is {value:.2f}. This means the "
        f"{other_group_name} group has {risk_reduction:.0f}% lower risk of "
        f"the event compared to the {baseline_group_name} group."
    )
