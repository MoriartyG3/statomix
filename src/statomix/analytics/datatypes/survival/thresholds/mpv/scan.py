"""Candidate construction for survival threshold scans.

The candidate unit is a patient partition, not a displayed numeric cutoff.
Several numeric cutoffs can induce the same ``<=``/``>`` allocation when the
predictor is tied or a synthetic grid is used.  This module canonicalizes such
cutoffs to one observed boundary and evaluates each partition at most once.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ThresholdCandidate:
    """One unique, ordered binary partition of a numerical predictor."""

    threshold: float
    partition_index: int
    lower_n: int
    upper_n: int
    lower_proportion: float
    upper_proportion: float
    source: str

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a serialization-safe representation."""

        return asdict(self)


class ThresholdScan:
    """Build unique predictor partitions within explicit size bounds.

    The lower group is defined as ``x <= threshold``.  Candidate-size bounds
    follow the CRAN ``maxstat`` indexing convention: the minimum and maximum
    allowed lower-group sizes are ``floor(n * minprop)`` and
    ``floor(n * maxprop)`` respectively, with a minimum size of one.  The
    realized lower proportion can therefore differ from ``minprop`` by less
    than one observation in small samples.

    Parameters
    ----------
    values:
        Finite numerical predictor values.
    minprop, maxprop:
        Search bounds for the lower-group proportion.
    use_synthetic_cutoffs:
        If true, start from a regular numeric grid.  Grid points that induce
        the same patient allocation are still collapsed to one partition.
    search_resolution:
        Positive grid spacing used only with synthetic cutoffs.
    """

    def __init__(
        self,
        values: pd.Series | np.ndarray,
        *,
        minprop: float = 0.1,
        maxprop: float = 0.9,
        use_synthetic_cutoffs: bool = False,
        search_resolution: float = 0.5,
    ) -> None:
        if not 0 <= minprop < maxprop <= 1:
            raise ValueError(
                "minprop and maxprop must satisfy "
                f"0 <= minprop < maxprop <= 1; got {minprop!r}, {maxprop!r}."
            )
        if use_synthetic_cutoffs and search_resolution <= 0:
            raise ValueError(
                "search_resolution must be positive when synthetic cutoffs "
                f"are enabled; got {search_resolution!r}."
            )

        array = np.asarray(values, dtype=float)
        if array.ndim != 1:
            raise ValueError("values must be one-dimensional.")
        if array.size < 2:
            raise ValueError("At least two observations are required.")
        if not np.isfinite(array).all():
            raise ValueError("values must contain only finite numbers.")
        if np.unique(array).size < 2:
            raise ValueError("At least two distinct predictor values are required.")

        self.values = array
        self.sorted_values = np.sort(array, kind="stable")
        self.n_observations = int(array.size)
        self.minprop = float(minprop)
        self.maxprop = float(maxprop)
        self.use_synthetic_cutoffs = bool(use_synthetic_cutoffs)
        self.search_resolution = float(search_resolution)
        self.candidates = self._build_candidates()

        if not self.candidates:
            raise ValueError(
                "No unique patient partitions satisfy minprop/maxprop. "
                "Widen the bounds or provide more distinct predictor values."
            )

    def _requested_thresholds(self) -> tuple[np.ndarray, str]:
        unique_values = np.unique(self.sorted_values)
        if not self.use_synthetic_cutoffs:
            return unique_values[:-1], "observed"

        lower = float(unique_values[0])
        upper = float(unique_values[-1])
        grid = np.arange(
            lower,
            upper + self.search_resolution,
            self.search_resolution,
            dtype=float,
        )
        grid = grid[grid < upper]
        return grid, "synthetic_grid"

    def _build_candidates(self) -> tuple[ThresholdCandidate, ...]:
        requested, source = self._requested_thresholds()
        n = self.n_observations
        minimum_size = max(1, int(np.floor(n * self.minprop)))
        maximum_size = min(n - 1, int(np.floor(n * self.maxprop)))

        partitions: dict[int, ThresholdCandidate] = {}
        for requested_threshold in requested:
            partition_index = int(
                np.searchsorted(
                    self.sorted_values,
                    requested_threshold,
                    side="right",
                )
            )
            if not minimum_size <= partition_index <= maximum_size:
                continue
            if partition_index <= 0 or partition_index >= n:
                continue

            canonical_threshold = float(self.sorted_values[partition_index - 1])
            partitions.setdefault(
                partition_index,
                ThresholdCandidate(
                    threshold=canonical_threshold,
                    partition_index=partition_index,
                    lower_n=partition_index,
                    upper_n=n - partition_index,
                    lower_proportion=partition_index / n,
                    upper_proportion=(n - partition_index) / n,
                    source=source,
                ),
            )

        return tuple(partitions[index] for index in sorted(partitions))

    @property
    def thresholds(self) -> np.ndarray:
        """Return canonical thresholds in partition order."""

        return np.asarray(
            [candidate.threshold for candidate in self.candidates],
            dtype=float,
        )

    @property
    def partition_indices(self) -> np.ndarray:
        """Return one-based cumulative group sizes for candidate partitions."""

        return np.asarray(
            [candidate.partition_index for candidate in self.candidates],
            dtype=int,
        )

    def to_frame(self) -> pd.DataFrame:
        """Return candidate metadata as an analysis-ready table."""

        return pd.DataFrame(candidate.to_dict() for candidate in self.candidates)
