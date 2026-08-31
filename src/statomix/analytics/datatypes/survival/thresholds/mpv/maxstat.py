"""Maximally selected log-rank statistics for one numerical predictor.

This implementation follows the score process used by the CRAN ``maxstat``
package.  It deliberately returns one scan-level p-value.  Per-cutoff Cox-Wald
or ordinary log-rank p-values are descriptive outputs and are not substituted
for this global test.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import beta, norm

from .scan import ThresholdScan

MaxstatMethod = Literal["lausen_1992", "conditional_monte_carlo"]

_RESULT_SCHEMA_VERSION = 2
_POST_SELECTION_WARNING = (
    "The cutoff was selected on these data. Hazard ratios, confidence "
    "intervals, and cutoff-specific p-values at the selected split are "
    "post-selection descriptive estimates and can be optimistic."
)


@dataclass(frozen=True, slots=True)
class MaxstatResult:
    """Serializable result of one maximally selected log-rank test."""

    schema_version: int
    statistic_family: str
    p_value_method: str
    statistic: float
    p_value: float
    optimal_threshold: float
    optimal_partition_index: int
    lower_n: int
    upper_n: int
    lower_proportion: float
    n_observations: int
    n_events: int
    n_unique_predictor_values: int
    has_predictor_ties: bool
    n_candidates: int
    minprop: float
    maxprop: float
    candidate_min_lower_proportion: float
    candidate_max_lower_proportion: float
    n_permutations: int | None
    extreme_count: int | None
    random_state: int | None
    exhaustive: bool
    monte_carlo_standard_error: float | None
    monte_carlo_confidence_level: float | None
    monte_carlo_ci_lower: float | None
    monte_carlo_ci_upper: float | None
    assumptions: tuple[str, ...]
    post_selection_warning: str
    reference_implementation: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe result dictionary."""

        result = asdict(self)
        result["assumptions"] = list(self.assumptions)
        return result


def logrank_scores(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Return CRAN ``exactRankTests::cscores.Surv`` log-rank scores.

    Integer ranks count observations with time less than or equal to each
    observed time, matching ``exactRankTests::irank``.  The formula therefore
    also reproduces the reference implementation's handling of tied times.
    """

    time_values = np.asarray(time, dtype=float)
    event_values = np.asarray(event, dtype=float)
    if time_values.ndim != 1 or event_values.ndim != 1:
        raise ValueError("time and event must be one-dimensional.")
    if time_values.size != event_values.size:
        raise ValueError("time and event must have the same length.")
    if time_values.size < 2:
        raise ValueError("At least two survival observations are required.")
    if not np.isfinite(time_values).all():
        raise ValueError("time must contain only finite values.")
    if (time_values < 0).any():
        raise ValueError("time must be non-negative.")
    if (
        not np.isfinite(event_values).all()
        or not np.isin(event_values, (0.0, 1.0)).all()
    ):
        raise ValueError("event must contain only exact 0/1 values.")

    order = np.argsort(time_values, kind="stable")
    sorted_time = time_values[order]
    integer_ranks = np.searchsorted(
        sorted_time,
        time_values,
        side="right",
    )
    risk_denominator = time_values.size - integer_ranks + 1
    hazard_terms = event_values / risk_denominator
    cumulative_hazard = np.cumsum(hazard_terms[order])
    return event_values - cumulative_hazard[integer_ranks - 1]


def lausen_schumacher_p_value(
    statistic: float,
    *,
    minprop: float = 0.1,
    maxprop: float = 0.9,
) -> float:
    """Return the Lausen--Schumacher (1992) Brownian-bridge approximation.

    The expression matches ``maxstat::pLausen92``.  It is a large-sample
    approximation, not a row-wise multiple-testing correction.
    """

    if not np.isfinite(statistic) or statistic < 0:
        raise ValueError("statistic must be finite and non-negative.")
    if not 0 < minprop < maxprop < 1:
        raise ValueError(
            "Lausen--Schumacher inference requires " "0 < minprop < maxprop < 1."
        )
    if statistic < 1:
        return 1.0

    density = float(norm.pdf(statistic))
    odds_ratio = (maxprop * (1 - minprop)) / ((1 - maxprop) * minprop)
    p_value = 4 * density / statistic + density * (statistic - 1 / statistic) * np.log(
        odds_ratio
    )
    return float(np.clip(p_value, 0.0, 1.0))


class MaximallySelectedLogRank:
    """Fit a maximally selected log-rank test across unique partitions."""

    def __init__(
        self,
        predictor: pd.Series | np.ndarray,
        time: pd.Series | np.ndarray,
        event: pd.Series | np.ndarray,
        *,
        minprop: float = 0.1,
        maxprop: float = 0.9,
    ) -> None:
        self.predictor = np.asarray(predictor, dtype=float)
        self.time = np.asarray(time, dtype=float)
        self.event = np.asarray(event, dtype=float)

        if not (self.predictor.ndim == self.time.ndim == self.event.ndim == 1):
            raise ValueError("predictor, time, and event must be one-dimensional.")
        if not (self.predictor.size == self.time.size == self.event.size):
            raise ValueError("predictor, time, and event must have the same length.")
        if not np.isfinite(self.predictor).all():
            raise ValueError("predictor must contain only finite values.")

        self.scan = ThresholdScan(
            self.predictor,
            minprop=minprop,
            maxprop=maxprop,
            use_synthetic_cutoffs=False,
        )
        self.minprop = float(minprop)
        self.maxprop = float(maxprop)
        self.scores = logrank_scores(self.time, self.event)
        self.process = self._build_standardized_process()
        self._observed_statistic = float(self.process["standardized_statistic"].max())
        first_maximum_position = int(
            np.argmax(self.process["standardized_statistic"].to_numpy(dtype=float))
        )
        self._optimal_row = self.process.iloc[first_maximum_position]

    def _build_standardized_process(self) -> pd.DataFrame:
        order = np.argsort(self.predictor, kind="stable")
        ordered_scores = self.scores[order]
        cumulative_scores = np.cumsum(ordered_scores)
        split_sizes = self.scan.partition_indices
        n = self.predictor.size
        score_sum = float(self.scores.sum())
        score_sum_squares = float(np.square(self.scores).sum())

        expectation = split_sizes / n * score_sum
        variance = (
            split_sizes
            * (n - split_sizes)
            / (n**2 * (n - 1))
            * (n * score_sum_squares - score_sum**2)
        )
        if not np.isfinite(variance).all() or (variance <= 0).any():
            raise ValueError(
                "The log-rank score process has non-positive variance; "
                "maxstat inference is undefined for these outcomes."
            )

        observed_sum = cumulative_scores[split_sizes - 1]
        standardized = np.abs(observed_sum - expectation) / np.sqrt(variance)
        process = self.scan.to_frame()
        process["score_sum"] = observed_sum
        process["expected_score_sum"] = expectation
        process["score_variance"] = variance
        process["standardized_statistic"] = standardized
        return process

    @property
    def process_df(self) -> pd.DataFrame:
        """Return a defensive copy of the standardized score process."""

        return self.process.copy()

    def fit(
        self,
        *,
        method: MaxstatMethod = "lausen_1992",
        n_permutations: int = 9_999,
        random_state: int | None = None,
        batch_size: int = 256,
        confidence_level: float = 0.95,
        exhaustive: bool = False,
        exhaustive_max_n: int = 9,
    ) -> MaxstatResult:
        """Calculate the selected global p-value and return a result record."""

        if method == "lausen_1992":
            if exhaustive:
                raise ValueError("exhaustive is only available for permutation fits.")
            p_value = lausen_schumacher_p_value(
                self._observed_statistic,
                minprop=self.minprop,
                maxprop=self.maxprop,
            )
            return self._result(
                method=method,
                p_value=p_value,
                assumptions=(
                    "Independent observations.",
                    "Right-censored outcomes and non-informative censoring.",
                    "Large-sample Brownian-bridge approximation.",
                    "A continuously ordered predictor under the null approximation.",
                ),
            )
        if method != "conditional_monte_carlo":
            raise ValueError(
                "method must be 'lausen_1992' or " "'conditional_monte_carlo'."
            )

        permutation = self._permutation_p_value(
            n_permutations=n_permutations,
            random_state=random_state,
            batch_size=batch_size,
            confidence_level=confidence_level,
            exhaustive=exhaustive,
            exhaustive_max_n=exhaustive_max_n,
        )
        return self._result(
            method=method,
            p_value=permutation["p_value"],
            n_permutations=permutation["n_permutations"],
            extreme_count=permutation["extreme_count"],
            random_state=random_state,
            exhaustive=exhaustive,
            monte_carlo_standard_error=permutation["standard_error"],
            monte_carlo_confidence_level=(None if exhaustive else confidence_level),
            monte_carlo_ci_lower=permutation["ci_lower"],
            monte_carlo_ci_upper=permutation["ci_upper"],
            assumptions=(
                "Independent observations.",
                "Right-censored outcomes and non-informative censoring.",
                "Exchangeability of survival scores and predictor values "
                "under the global null.",
                (
                    "Exhaustive enumeration of all subject-label permutations."
                    if exhaustive
                    else "Monte Carlo approximation; uncertainty is reported."
                ),
            ),
        )

    def _max_statistics_for_score_matrix(
        self,
        score_matrix: np.ndarray,
    ) -> np.ndarray:
        split_sizes = self.scan.partition_indices
        cumulative = np.cumsum(score_matrix, axis=1)[:, split_sizes - 1]
        expectation = self.process["expected_score_sum"].to_numpy(dtype=float)
        standard_deviation = np.sqrt(
            self.process["score_variance"].to_numpy(dtype=float)
        )
        standardized = np.abs(cumulative - expectation) / standard_deviation
        return standardized.max(axis=1)

    def _permutation_p_value(
        self,
        *,
        n_permutations: int,
        random_state: int | None,
        batch_size: int,
        confidence_level: float,
        exhaustive: bool,
        exhaustive_max_n: int,
    ) -> dict[str, float | int | None]:
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1).")
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

        tolerance = np.finfo(float).eps * max(1.0, self._observed_statistic) * 8
        extreme_count = 0

        if exhaustive:
            n = self.predictor.size
            if n > exhaustive_max_n:
                raise ValueError(
                    f"Exhaustive permutation is limited to n <= "
                    f"{exhaustive_max_n}; got n={n}."
                )
            total = 0
            for permutation in itertools.permutations(range(n)):
                permuted = self.scores[np.asarray(permutation, dtype=int)][None, :]
                maximum = self._max_statistics_for_score_matrix(permuted)[0]
                extreme_count += int(maximum >= self._observed_statistic - tolerance)
                total += 1
            return {
                "p_value": extreme_count / total,
                "n_permutations": total,
                "extreme_count": extreme_count,
                "standard_error": None,
                "ci_lower": None,
                "ci_upper": None,
            }

        if not isinstance(n_permutations, int) or n_permutations < 1:
            raise ValueError("n_permutations must be a positive integer.")
        rng = np.random.default_rng(random_state)
        completed = 0
        n = self.predictor.size
        while completed < n_permutations:
            current_batch = min(batch_size, n_permutations - completed)
            permuted_scores = np.empty((current_batch, n), dtype=float)
            for row in range(current_batch):
                permuted_scores[row] = rng.permutation(self.scores)
            maxima = self._max_statistics_for_score_matrix(permuted_scores)
            extreme_count += int(
                np.count_nonzero(maxima >= self._observed_statistic - tolerance)
            )
            completed += current_batch

        corrected_p_value = (1 + extreme_count) / (n_permutations + 1)
        standard_error = float(
            np.sqrt(corrected_p_value * (1 - corrected_p_value) / (n_permutations + 1))
        )
        ci_lower, ci_upper = self._corrected_clopper_pearson_interval(
            extreme_count=extreme_count,
            n_permutations=n_permutations,
            confidence_level=confidence_level,
        )
        return {
            "p_value": corrected_p_value,
            "n_permutations": n_permutations,
            "extreme_count": extreme_count,
            "standard_error": standard_error,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    @staticmethod
    def _corrected_clopper_pearson_interval(
        *,
        extreme_count: int,
        n_permutations: int,
        confidence_level: float,
    ) -> tuple[float, float]:
        alpha = 1 - confidence_level
        if extreme_count == 0:
            lower_tail = 0.0
        else:
            lower_tail = float(
                beta.ppf(
                    alpha / 2,
                    extreme_count,
                    n_permutations - extreme_count + 1,
                )
            )
        if extreme_count == n_permutations:
            upper_tail = 1.0
        else:
            upper_tail = float(
                beta.ppf(
                    1 - alpha / 2,
                    extreme_count + 1,
                    n_permutations - extreme_count,
                )
            )

        denominator = n_permutations + 1
        corrected_lower = (1 + n_permutations * lower_tail) / denominator
        corrected_upper = (1 + n_permutations * upper_tail) / denominator
        return (
            float(np.clip(corrected_lower, 0.0, 1.0)),
            float(np.clip(corrected_upper, 0.0, 1.0)),
        )

    def _result(
        self,
        *,
        method: MaxstatMethod,
        p_value: float,
        assumptions: tuple[str, ...],
        n_permutations: int | None = None,
        extreme_count: int | None = None,
        random_state: int | None = None,
        exhaustive: bool = False,
        monte_carlo_standard_error: float | None = None,
        monte_carlo_confidence_level: float | None = None,
        monte_carlo_ci_lower: float | None = None,
        monte_carlo_ci_upper: float | None = None,
    ) -> MaxstatResult:
        row = self._optimal_row
        return MaxstatResult(
            schema_version=_RESULT_SCHEMA_VERSION,
            statistic_family="maximally_selected_log_rank",
            p_value_method=method,
            statistic=self._observed_statistic,
            p_value=float(p_value),
            optimal_threshold=float(row["threshold"]),
            optimal_partition_index=int(row["partition_index"]),
            lower_n=int(row["lower_n"]),
            upper_n=int(row["upper_n"]),
            lower_proportion=float(row["lower_proportion"]),
            n_observations=int(self.predictor.size),
            n_events=int(self.event.sum()),
            n_unique_predictor_values=int(np.unique(self.predictor).size),
            has_predictor_ties=bool(
                np.unique(self.predictor).size < self.predictor.size
            ),
            n_candidates=int(self.process.shape[0]),
            minprop=self.minprop,
            maxprop=self.maxprop,
            candidate_min_lower_proportion=float(
                self.process["lower_proportion"].min()
            ),
            candidate_max_lower_proportion=float(
                self.process["lower_proportion"].max()
            ),
            n_permutations=n_permutations,
            extreme_count=extreme_count,
            random_state=random_state,
            exhaustive=exhaustive,
            monte_carlo_standard_error=monte_carlo_standard_error,
            monte_carlo_confidence_level=monte_carlo_confidence_level,
            monte_carlo_ci_lower=monte_carlo_ci_lower,
            monte_carlo_ci_upper=monte_carlo_ci_upper,
            assumptions=assumptions,
            post_selection_warning=_POST_SELECTION_WARNING,
            reference_implementation=(
                "CRAN maxstat 0.7-26 with exactRankTests 0.8-37 " "log-rank scores"
            ),
        )
