from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from statomix.analytics.datatypes.survival.thresholds.mpv.mpv import MinimumPValue
from statomix.analytics.multiplicity import (
    SUPPORTED_CORRECTIONS,
    adjust_p_values,
    adjust_p_values_with_missing,
    normalize_correction_methods,
)


def _mpv_for_plotting() -> MinimumPValue:
    mpv = MinimumPValue.__new__(MinimumPValue)
    mpv.alpha = 0.05
    mpv.correction_methods = ("none", "bonferroni", "fdr_bh")
    mpv.selection_method = "none"
    mpv.selection_family = "cox_ph"
    mpv.multiplicity_method = "none"
    mpv.mode = "exploratory"
    mpv.maxstat_result = None
    mpv.surv_label = "OS"
    mpv.target_col_stats = {"name": "biomarker", "median": 2.1}
    mpv.mpv_df = pd.DataFrame(
        {
            "threshold": [1.0, 2.0, 3.0],
            "valid_split": [True, True, True],
            "cox_ph.p_value": [0.01, 0.04, 0.20],
            "log_rank.p_value": [0.20, 0.03, 0.01],
            "cox_ph.hr.raw.hr": [0.8, 1.2, 2.0],
            "split_ratio": [1.0, 1.2, 0.8],
        }
    )
    mpv._add_multiplicity_columns(mpv_df=mpv.mpv_df)
    mpv._refresh_marker_cache()
    return mpv


def test_mpv_defaults_to_raw_p_values() -> None:
    signature = inspect.signature(MinimumPValue)

    assert signature.parameters["multiplicity_method"].default is None
    assert signature.parameters["correction_methods"].default is None
    assert signature.parameters["selection_method"].default == "none"
    assert signature.parameters["mode"].default is None
    assert normalize_correction_methods(None) == ("none",)


def test_correction_configuration_is_ordered_deduplicated_and_validated() -> None:
    assert normalize_correction_methods(
        ["holm", "fdr_bh", "holm"],
        selection_method="fdr_by",
    ) == ("none", "holm", "fdr_bh", "fdr_by")

    with pytest.raises(ValueError, match="Unknown correction method"):
        normalize_correction_methods(["not-a-method"])
    with pytest.raises(TypeError, match="must be strings"):
        normalize_correction_methods(["holm", 1])  # type: ignore[list-item]


@pytest.mark.parametrize("method", SUPPORTED_CORRECTIONS)
def test_registered_corrections_preserve_missing_positions(method: str) -> None:
    raw = np.array([0.01, np.nan, 0.04, np.inf, 0.20])
    adjusted = adjust_p_values_with_missing(raw, method=method)

    assert np.isnan(adjusted[[1, 3]]).all()
    assert np.isfinite(adjusted[[0, 2, 4]]).all()
    assert ((adjusted[[0, 2, 4]] >= 0) & (adjusted[[0, 2, 4]] <= 1)).all()


def test_known_adjustments_and_raw_identity() -> None:
    raw = np.array([0.01, 0.04, 0.03])

    np.testing.assert_allclose(adjust_p_values(raw, method="none"), raw)
    np.testing.assert_allclose(
        adjust_p_values(raw, method="bonferroni"),
        [0.03, 0.12, 0.09],
    )
    np.testing.assert_allclose(
        adjust_p_values(raw, method="holm"),
        [0.03, 0.06, 0.06],
    )
    np.testing.assert_allclose(
        adjust_p_values(raw, method="fdr_bh"),
        [0.03, 0.04, 0.04],
    )


def test_numerical_boundary_noise_is_clipped_but_real_domain_errors_fail() -> None:
    adjusted = adjust_p_values([1 + 5e-13, -5e-13], method="none")
    np.testing.assert_allclose(adjusted, [1.0, 0.0])

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        adjust_p_values([1 + 2e-12], method="none")


def test_no_correction_does_not_create_duplicate_raw_columns() -> None:
    mpv = MinimumPValue.__new__(MinimumPValue)
    mpv.correction_methods = ("none",)
    mpv.selection_method = "none"
    mpv_df = pd.DataFrame(
        {
            "cox_ph.p_value": [0.01, 0.02],
            "log_rank.p_value": [0.03, 0.04],
        }
    )

    mpv._add_multiplicity_columns(mpv_df=mpv_df)

    assert not any(column.endswith("_none") for column in mpv_df)
    assert mpv_df["multiplicity.methods"].eq("none").all()
    assert mpv_df["multiplicity.selection_method"].eq("none").all()


def test_threshold_markers_are_computed_per_correction() -> None:
    mpv = _mpv_for_plotting()
    raw_markers = {
        marker["label"]: marker["idx"]
        for marker in mpv.marked_thresholds_by_correction["none"]
    }
    bonferroni_markers = {
        marker["label"]: marker["idx"]
        for marker in mpv.marked_thresholds_by_correction["bonferroni"]
    }

    assert raw_markers["Closest significant Cox-Wald (descriptive)"] == 1
    assert bonferroni_markers["Closest significant Cox-Wald (descriptive)"] == 0


def test_threshold_markers_are_computed_per_test_family() -> None:
    mpv = _mpv_for_plotting()
    cox_markers = {
        marker["label"]: marker["idx"]
        for marker in mpv.marked_thresholds_by_family["cox_ph"]["none"]
    }
    log_rank_markers = {
        marker["label"]: marker["idx"]
        for marker in mpv.marked_thresholds_by_family["log_rank"]["none"]
    }

    assert cox_markers["Min Cox-Wald p (descriptive)"] == 0
    assert log_rank_markers["Min log-rank p (descriptive)"] == 2


def test_inferential_marker_is_the_log_rank_maxstat_optimum() -> None:
    mpv = _mpv_for_plotting()
    mpv.mode = "inferential"
    mpv.selection_family = "log_rank"
    mpv.maxstat_result = SimpleNamespace(optimal_threshold=3.0)
    mpv._refresh_marker_cache()

    selected = {marker["label"]: marker["idx"] for marker in mpv.marked_threshold_dicts}

    assert selected["Maxstat optimum (log-rank)"] == 2
    assert not any("Cox" in label for label in selected)


def test_correction_specific_and_combined_plots() -> None:
    mpv = _mpv_for_plotting()

    raw_figure = mpv.plot_p_values(correction="none")
    adjusted_figure = mpv.plot_p_values(correction="fdr_bh")
    combined_figure = mpv.plot_p_values_all_corrections()

    raw_labels = {line.get_label() for line in raw_figure.axes[0].get_lines()}
    adjusted_labels = {line.get_label() for line in adjusted_figure.axes[0].get_lines()}
    combined_labels = {line.get_label() for line in combined_figure.axes[0].get_lines()}

    assert "Cox PH p-value" in raw_labels
    assert raw_figure.axes[0].get_lines()[0].get_markersize() == 2.5
    assert "Cox PH (fdr_bh-adjusted)" in adjusted_labels
    assert {"Raw (none)", "bonferroni", "fdr_bh"} <= combined_labels
    assert len(combined_figure.axes) == 2

    plt.close(raw_figure)
    plt.close(adjusted_figure)
    plt.close(combined_figure)


def test_dashboard_always_delegates_to_raw_view() -> None:
    mpv = _mpv_for_plotting()

    with (
        patch.object(MinimumPValue, "plot_p_values") as plot_p_values,
        patch.object(MinimumPValue, "plot_hr_with_ci") as plot_hr_with_ci,
        patch.object(MinimumPValue, "plot_group_sizes") as plot_group_sizes,
        patch.object(MinimumPValue, "plot_split_ratio") as plot_split_ratio,
        patch.object(MinimumPValue, "plot_median_survival") as plot_median_survival,
    ):
        figure = mpv.plot_dashboard(correction="fdr_bh")

    for mocked_method in (
        plot_p_values,
        plot_hr_with_ci,
        plot_group_sizes,
        plot_split_ratio,
        plot_median_survival,
    ):
        assert mocked_method.call_args.kwargs["correction"] == "none"

    plt.close(figure)


def test_scatter_uses_raw_p_values_by_default() -> None:
    mpv = _mpv_for_plotting()

    raw_figure = mpv.plot_hr_vs_pvalue_scatter()
    corrected_figure = mpv.plot_hr_vs_pvalue_scatter(correction="fdr_bh")

    assert "raw" in raw_figure.axes[0].get_ylabel()
    assert "fdr_bh-adjusted" in corrected_figure.axes[0].get_ylabel()

    plt.close(raw_figure)
    plt.close(corrected_figure)


def test_missing_requested_correction_requires_regeneration() -> None:
    mpv = MinimumPValue.__new__(MinimumPValue)
    mpv.mpv_df = pd.DataFrame({"cox_ph.p_value": [0.01]})

    with pytest.raises(RuntimeError, match="replace=True"):
        mpv._p_value_column(family="cox_ph", correction="holm")
