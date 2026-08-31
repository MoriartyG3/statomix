from __future__ import annotations

import itertools
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from statomix.analytics.datatypes.survival.binary_class_surv import BinaryClassSurv
from statomix.analytics.datatypes.survival.thresholds.mpv.maxstat import (
    MaximallySelectedLogRank,
    lausen_schumacher_p_value,
    logrank_scores,
)
from statomix.analytics.datatypes.survival.thresholds.mpv.mpv import MinimumPValue
from statomix.analytics.datatypes.survival.thresholds.mpv.scan import ThresholdScan


def _survival_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictor = np.asarray([1, 2, 3, 4, 5, 6], dtype=float)
    time = np.asarray([1, 4, 2, 6, 3, 5], dtype=float)
    event = np.asarray([1, 1, 0, 1, 0, 1], dtype=int)
    return predictor, time, event


def test_unique_partitions_collapse_ties_and_synthetic_grid_points() -> None:
    predictor = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=float)
    observed = ThresholdScan(predictor, minprop=0.1, maxprop=0.9)
    synthetic = ThresholdScan(
        predictor,
        minprop=0.1,
        maxprop=0.9,
        use_synthetic_cutoffs=True,
        search_resolution=0.25,
    )

    np.testing.assert_array_equal(observed.partition_indices, [2, 4, 6])
    np.testing.assert_array_equal(synthetic.partition_indices, [2, 4, 6])
    np.testing.assert_allclose(observed.thresholds, [0, 1, 2])
    np.testing.assert_allclose(synthetic.thresholds, [0, 1, 2])


def test_logrank_scores_match_exact_rank_tests_tie_convention() -> None:
    scores = logrank_scores(
        time=np.asarray([1, 1, 2], dtype=float),
        event=np.asarray([1, 0, 1], dtype=int),
    )

    np.testing.assert_allclose(scores, [0.5, -0.5, -0.5])


@pytest.mark.parametrize(
    ("statistic", "minprop", "maxprop"),
    [
        (2.829, 0.25, 0.75),
        (2.560, 0.40, 0.60),
        (3.054, 0.10, 0.90),
        (2.883, 0.40, 0.90),
    ],
)
def test_lausen_1992_matches_pinned_r_maxstat_table_2_quantiles(
    statistic: float,
    minprop: float,
    maxprop: float,
) -> None:
    # CRAN maxstat 0.7-26 reproduces Lausen & Schumacher (1992), Table 2.
    # These rounded statistics are the package's published alpha=0.05
    # quantiles, so the small tolerance accounts for the three-decimal input.
    p_value = lausen_schumacher_p_value(
        statistic,
        minprop=minprop,
        maxprop=maxprop,
    )

    assert p_value == pytest.approx(0.05, abs=5e-4)


def test_exhaustive_permutation_matches_independent_enumeration() -> None:
    predictor, time, event = _survival_fixture()
    analysis = MaximallySelectedLogRank(
        predictor,
        time,
        event,
        minprop=0.33,
        maxprop=0.67,
    )
    result = analysis.fit(
        method="conditional_monte_carlo",
        exhaustive=True,
        exhaustive_max_n=6,
    )

    scores = analysis.scores
    split_sizes = analysis.scan.partition_indices
    n = scores.size
    score_sum = scores.sum()
    expectation = split_sizes / n * score_sum
    variance = (
        split_sizes
        * (n - split_sizes)
        / (n**2 * (n - 1))
        * (n * np.square(scores).sum() - score_sum**2)
    )
    tolerance = np.finfo(float).eps * max(1.0, result.statistic) * 8
    extreme = 0
    total = 0
    for permutation in itertools.permutations(scores):
        cumulative = np.cumsum(np.asarray(permutation, dtype=float))
        maximum = np.max(
            np.abs(cumulative[split_sizes - 1] - expectation) / np.sqrt(variance)
        )
        extreme += int(maximum >= result.statistic - tolerance)
        total += 1

    assert total == 720
    assert result.n_permutations == total
    assert result.extreme_count == extreme
    assert result.p_value == pytest.approx(extreme / total)
    assert result.exhaustive is True


def test_monte_carlo_p_value_uses_plus_one_correction() -> None:
    predictor, time, event = _survival_fixture()
    result = MaximallySelectedLogRank(
        predictor,
        time,
        event,
        minprop=0.33,
        maxprop=0.67,
    ).fit(
        method="conditional_monte_carlo",
        n_permutations=17,
        random_state=7,
        batch_size=4,
    )

    assert result.extreme_count is not None
    assert result.p_value == (1 + result.extreme_count) / 18
    assert result.p_value > 0
    assert result.monte_carlo_standard_error is not None
    assert result.monte_carlo_ci_lower is not None
    assert result.monte_carlo_ci_upper is not None


def test_maxstat_rejects_an_outcome_with_no_score_variance() -> None:
    predictor = np.arange(1, 7, dtype=float)
    time = np.arange(1, 7, dtype=float)
    event = np.zeros(6, dtype=int)

    with pytest.raises(ValueError, match="non-positive variance"):
        MaximallySelectedLogRank(predictor, time, event)


def test_mode_is_required_before_artifact_access() -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 1, 0],
            "marker": [1.0, 2.0, 3.0, 4.0],
        }
    )

    with pytest.raises(ValueError, match="mode must be explicitly set"):
        MinimumPValue("OS", frame, root_group=None)


def test_invalid_family_rows_are_excluded_from_every_correction() -> None:
    table = pd.DataFrame(
        {
            "cox_ph.p_value": [0.01, 0.02, 0.03],
            "log_rank.p_value": [0.04, 0.05, 0.06],
            "cox_ph_valid_split": [True, False, True],
            "log_rank_valid_split": [False, True, True],
        }
    )
    scan = MinimumPValue.__new__(MinimumPValue)
    scan.correction_methods = ("none", "holm")
    scan.selection_method = "none"

    scan._add_multiplicity_columns(mpv_df=table)

    assert table["cox_ph.multiplicity.n_tests"].eq(2).all()
    assert table["log_rank.multiplicity.n_tests"].eq(2).all()
    assert np.isnan(table.loc[1, "cox_ph.p_value_holm"])
    assert np.isnan(table.loc[0, "log_rank.p_value_holm"])
    assert table["cox_ph.multiplicity.eligible"].tolist() == [
        True,
        False,
        True,
    ]


def test_cox_failure_does_not_erase_valid_log_rank_result() -> None:
    analysis = BinaryClassSurv.__new__(BinaryClassSurv)
    analysis.surv_df0 = pd.DataFrame({"time": [1.0, 2.0], "event": [1, 0]})
    analysis.surv_df1 = pd.DataFrame({"time": [3.0, 4.0], "event": [1, 0]})

    with (
        patch.object(
            BinaryClassSurv,
            "get_log_rank_dict",
            return_value={"valid": True, "p_value": 0.04},
        ),
        patch.object(
            BinaryClassSurv,
            "get_cox_ph_dict",
            side_effect=RuntimeError("singular fit"),
        ),
    ):
        results = analysis.get_tests_dict()

    assert results["log_rank"] == {"valid": True, "p_value": 0.04}
    assert results["cox_ph"]["split_invalid_reason"] == "analysis_error"
    assert np.isnan(results["cox_ph"]["p_value"])


def test_scan_provenance_mismatch_is_rejected(tmp_path) -> None:
    scan = MinimumPValue.__new__(MinimumPValue)
    scan.paths = {"scan_config": tmp_path / "scan_config.json"}
    scan.scan_fingerprint = "fingerprint-a"
    scan.scan_config = {"schema_version": 2, "alpha": 0.05}
    scan._write_scan_config(replace=True)

    scan.scan_config = {"schema_version": 2, "alpha": 0.01}
    with pytest.raises(RuntimeError, match="provenance does not match"):
        scan._write_scan_config(replace=False)


def test_scan_provenance_captures_every_result_changing_setting() -> None:
    scan = MinimumPValue.__new__(MinimumPValue)
    scan.mode = "inferential"
    scan.surv_label = "OS"
    scan.target_col_stats = {"name": "marker"}
    scan.surv_df_mpv = pd.DataFrame(
        {
            "time": [1.0, 2.0],
            "event": [True, False],
            "marker": [3.0, 4.0],
        }
    )
    scan.minprop = 0.1
    scan.maxprop = 0.9
    scan.use_synthetic_cutoffs = False
    scan.search_resolution = 0.5
    scan.alpha = 0.05
    scan.correction_methods = ("none", "holm")
    scan.selection_method = "none"
    scan.selection_family = "log_rank"
    scan.maxstat_method = "conditional_monte_carlo"
    scan.n_permutations = 9_999
    scan.random_state = 17
    scan.permutation_batch_size = 128
    scan.permutation_confidence_level = 0.95
    scan.trunc_pct = None
    scan.iqr_multiplier = None

    config = scan._build_scan_config()

    assert {
        "input_fingerprint",
        "mode",
        "minprop",
        "maxprop",
        "use_synthetic_cutoffs",
        "search_resolution",
        "alpha",
        "correction_methods",
        "selection_method",
        "selection_family",
        "maxstat_method",
        "n_permutations",
        "random_state",
        "permutation_batch_size",
        "permutation_confidence_level",
    } <= set(config)


def test_legacy_artifact_load_is_read_only(tmp_path) -> None:
    path = tmp_path / "legacy_mpv_df.parquet"
    expected = pd.DataFrame({"threshold": [1.0], "cox_ph.p_value": [0.2]})
    expected.to_parquet(path, index=False)
    before = path.read_bytes()

    loaded = MinimumPValue.load_legacy_artifact(path)

    pd.testing.assert_frame_equal(loaded, expected)
    assert path.read_bytes() == before
