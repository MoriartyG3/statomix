from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statomix.analysis.descriptive.categorical import BaseCategorical
from statomix.analysis.multiplicity import holm_adjust, holm_adjust_with_missing
from statomix.analysis.survival.data import prepare_survival_data
from statomix.analysis.survival.formatting import (
    get_p_value_label,
    interpret_hazard_ratio,
)


def test_p_value_label_uses_unrounded_threshold() -> None:
    assert get_p_value_label(0.0009) == "P<0.001"
    assert get_p_value_label(0.004) == "P=0.004"
    assert get_p_value_label(0.04996) == "P=0.0500"


def test_p_value_and_hazard_ratio_domains_are_validated() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        get_p_value_label(1.1)
    with pytest.raises(ValueError, match="greater than 0"):
        interpret_hazard_ratio(
            hazard_ratio=0,
            baseline_group_name="A",
            other_group_name="B",
        )


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

    with_missing = holm_adjust_with_missing([0.01, np.nan, 0.04])
    np.testing.assert_allclose(with_missing[[0, 2]], [0.02, 0.04])
    assert np.isnan(with_missing[1])


def test_mpv_multiplicity_metadata_uses_separate_finite_family_counts() -> None:
    from statomix.analysis.survival.thresholds.minimum_p_value import (
        MinimumPValue,
    )

    mpv_df = pd.DataFrame(
        {
            "cox_ph.p_value": [0.01, np.nan, np.inf, 0.04],
            "log_rank.p_value": [0.01, 0.04, 0.03, np.nan],
        }
    )
    expected_cox = holm_adjust_with_missing(mpv_df["cox_ph.p_value"])
    expected_log_rank = holm_adjust_with_missing(mpv_df["log_rank.p_value"])

    mpv = MinimumPValue.__new__(MinimumPValue)
    mpv.correction_methods = ("none", "holm")
    mpv.selection_method = "holm"
    mpv.multiplicity_method = "holm"
    mpv._add_multiplicity_columns(mpv_df=mpv_df)

    assert mpv_df["cox_ph.multiplicity.n_tests"].eq(2).all()
    assert mpv_df["log_rank.multiplicity.n_tests"].eq(3).all()
    assert "multiplicity.n_tests" not in mpv_df.columns
    assert mpv_df["multiplicity.methods"].eq("none|holm").all()
    assert mpv_df["multiplicity.selection_method"].eq("holm").all()
    np.testing.assert_allclose(
        mpv_df["cox_ph.p_value_holm"], expected_cox, equal_nan=True
    )
    np.testing.assert_allclose(
        mpv_df["log_rank.p_value_holm"], expected_log_rank, equal_nan=True
    )


def test_mpv_multiplicity_metadata_has_stable_empty_schema() -> None:
    from statomix.analysis.survival.thresholds.minimum_p_value import (
        MinimumPValue,
    )

    mpv_df = pd.DataFrame()
    mpv = MinimumPValue.__new__(MinimumPValue)
    mpv.correction_methods = ("none", "holm")
    mpv.selection_method = "holm"
    mpv.multiplicity_method = "holm"
    mpv._add_multiplicity_columns(mpv_df=mpv_df)

    assert {
        "cox_ph.multiplicity.n_tests",
        "log_rank.multiplicity.n_tests",
        "multiplicity.methods",
        "multiplicity.selection_method",
    } <= set(mpv_df.columns)
    assert mpv_df["cox_ph.multiplicity.n_tests"].empty
    assert mpv_df["log_rank.multiplicity.n_tests"].empty


def test_survival_data_is_validated_once_with_a_drop_audit() -> None:
    prepared = prepare_survival_data(
        frame=pd.DataFrame(
            {
                "time": [3.0, 4.0, np.nan],
                "event": [1, 0, 1],
                "group": ["A", "B", "A"],
            }
        ),
        grouping_column="group",
    )

    assert prepared.dropped_rows == 1
    assert prepared.frame["event"].dtype == bool
    assert prepared.frame["event"].tolist() == [True, False]


def test_categorical_distribution_preserves_native_order_when_possible() -> None:
    numeric = BaseCategorical.get_distribution_df(series=pd.Series([10, 2, 10, 2, 1]))
    mixed = BaseCategorical.get_distribution_df(
        series=pd.Series(["2", 1, "10", 2], dtype="object")
    )

    assert numeric["category"].tolist() == [1, 2, 10]
    assert mixed["category"].tolist() == [1, 2, "10", "2"]


@pytest.mark.parametrize(
    "frame, message",
    [
        (
            pd.DataFrame({"time": [-1.0], "event": [1]}),
            "negative durations",
        ),
        (
            pd.DataFrame({"time": [1.0], "event": [2]}),
            "only 0/1",
        ),
        (
            pd.DataFrame({"time": [np.inf], "event": [1]}),
            "finite durations",
        ),
        (
            pd.DataFrame({"time": [True], "event": [1]}),
            "must be numeric",
        ),
    ],
)
def test_invalid_survival_inputs_fail_explicitly(
    frame: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_survival_data(frame=frame)


def test_raw_fitted_parameter_ks_is_excluded_from_consensus() -> None:
    pytest.importorskip("statsmodels")
    from statomix.analysis.normality import Normality

    normality = Normality(series=pd.Series(np.linspace(-2, 2, 60)))
    report = normality.get_normality_report(test_type="ks")
    consensus = normality.get_consensus()

    assert report["p_value_reliable"] is False
    assert consensus["votes_total"] == 5
