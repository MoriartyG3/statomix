"""Regression tests for unit-aware survival plot ticks."""

from __future__ import annotations

import numpy as np
import pytest

from statomix.pipelines.analyzer.artifact_survival import (
    _survival_axis_ticks,
)


@pytest.mark.parametrize(
    ("maximum", "expected"),
    [
        (0.0, [0]),
        (3.0, [0]),
        (12.0, [0, 12]),
        (24.0, [0, 12, 24]),
        (45.1, [0, 12, 24, 36]),
        (534.0, list(range(0, 529, 12))),
    ],
)
def test_month_ticks_use_twelve_month_intervals(maximum, expected):
    ticks = _survival_axis_ticks(
        maximum=maximum,
        unit_name="months",
    )

    assert ticks == expected
    assert all(type(tick) is int for tick in ticks)
    assert all(tick <= maximum for tick in ticks)


@pytest.mark.parametrize("unit_name", ["month", "Months", " months "])
def test_month_unit_normalization(unit_name):
    ticks = _survival_axis_ticks(
        maximum=36.0,
        unit_name=unit_name,
    )

    assert ticks == [0, 12, 24, 36]


def test_other_units_retain_existing_spacing():
    ticks = _survival_axis_ticks(
        maximum=100.0,
        unit_name="days",
    )

    np.testing.assert_allclose(
        ticks,
        [0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
    )


def test_zero_maximum_for_other_units():
    ticks = _survival_axis_ticks(
        maximum=0.0,
        unit_name="days",
    )

    assert ticks == [0.0]


@pytest.mark.parametrize(
    "maximum",
    [-1.0, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_maximum_is_rejected(maximum):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _survival_axis_ticks(
            maximum=maximum,
            unit_name="months",
        )
