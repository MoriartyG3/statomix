"""
Minimum p-value (optimal cutpoint) search for a continuous/ordinal
survival covariate.

Given a continuous or ordinal column plus "time"/"event", this sweeps a
grid of candidate thresholds, binarizes the column at each one
(<= threshold vs. > threshold), and runs a log-rank test + Cox model via
`BinaryClassSurv` at every threshold. The output is a DataFrame with one
row per threshold, which can then be inspected for the threshold that
best separates the two groups.

IMPORTANT -- the multiple testing problem
------------------------------------------
This is the classic "minimum p-value" / "optimal cutpoint" approach
(Miller & Siegmund 1982; Altman et al. 1994; Lausen & Schumacher 1992).
Scanning many thresholds and reporting the smallest log-rank p-value as
if it were a single planned test is a well-known source of false
positives: the minimum of many correlated test statistics is *not*
distributed the same way a single test statistic is, so comparing it
directly to `alpha` overstates significance, often substantially so
(the more thresholds scanned, the worse the inflation).

This class does NOT correct for that on its own. `get_mpv_df()` returns
the raw, unadjusted result of each individual test. Treat the "p_value"
and "significant" columns as descriptive/exploratory, not confirmatory,
unless you've applied an appropriate correction (e.g. a permutation test
on the minimum p-value itself, or the Lausen & Schumacher adjustment) on
top of this output -- see `statomix...mpv_corrections` for those, and the
`*_correction` kwargs on the plotting methods below for overlaying them.
`get_best_threshold()` will emit a warning every time it's called, as a
deliberate, hard-to-miss reminder of this.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from statomix.analytics.datatypes.time_to_event.binary_class_surv import BinaryClassSurv


class MinimumPValue:
    """Sweeps cutpoints over a single continuous/ordinal column and runs a
    binary log-rank + Cox comparison at each one.

    Parameters
    ----------
    surv_df_mpv : pandas.DataFrame
        Must contain "time", "event", and exactly one other column (the
        continuous/ordinal variable to search over).
    surv_label : str
        Passed through to `BinaryClassSurv` / `SingleClassSurv` at each
        threshold (legend label / fit name).
    use_synthetic_cutoffs : bool, default False
        If True, scan a regular grid from `floor(min)` to `ceil(max)` in
        steps of `search_resolution`, instead of using the observed data
        values as candidate cutoffs. Useful when the underlying variable
        is continuous and you want evenly spaced candidate thresholds
        rather than being tied to wherever data points happen to fall.
    search_resolution : float, default 0.5
        Step size for the synthetic grid. Ignored if
        `use_synthetic_cutoffs=False`. Must be > 0.
    show_progress : bool, default True
        Show a tqdm progress bar while sweeping thresholds.
    alpha : float, default 0.05
        Significance threshold used to populate the "significant" column.
        Must be in (0, 1). See the module docstring -- this is NOT
        multiplicity-adjusted.
    skip_invalid_thresholds : bool, default True
        If a threshold produces a degenerate split (all points on one
        side) it's always skipped gracefully (`valid_split=False`). This
        flag additionally controls whether *unexpected* errors raised
        while fitting `BinaryClassSurv` at a given threshold (e.g. a
        degenerate Cox fit, a convergence failure) are caught and
        recorded as `valid_split=False` rows with an "error" message
        (True), or re-raised immediately, aborting the whole sweep
        (False). Defaults to True because a single problematic threshold
        out of potentially hundreds should not silently kill the entire
        search.

    Notes
    -----
    Each row's `baseline_group` is always read from
    `BinaryClassSurv.baseline_label` (the resolved, authoritative value)
    rather than being independently reconstructed from the threshold --
    this guarantees the reported baseline always matches what the Cox
    model actually used as its reference level.
    """

    def __init__(
        self,
        surv_df_mpv: pd.DataFrame,
        surv_label: str,
        use_synthetic_cutoffs: bool = False,
        search_resolution: float = 0.5,
        show_progress: bool = True,
        alpha: float = 0.05,
        skip_invalid_thresholds: bool = True,
    ):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        if use_synthetic_cutoffs and search_resolution <= 0:
            raise ValueError(
                f"search_resolution must be positive, got {search_resolution!r}"
            )

        self.alpha = alpha
        self.surv_label = surv_label
        self.surv_df_mpv = surv_df_mpv
        self.show_progress = show_progress
        self.search_resolution = search_resolution
        self.use_synthetic_cutoffs = use_synthetic_cutoffs
        self.skip_invalid_thresholds = skip_invalid_thresholds

        required_cols = {"time", "event"}
        missing = required_cols - set(self.surv_df_mpv.columns)
        if missing:
            raise ValueError(
                f"surv_df_mpv is missing required column(s): {sorted(missing)}"
            )

        grouping_cols = [c for c in self.surv_df_mpv.columns if c not in ("time", "event")]
        if len(grouping_cols) != 1:
            raise ValueError(
                "Expected exactly one grouping column besides 'time'/'event', "
                f"found {len(grouping_cols)}: {grouping_cols}"
            )
        target_col_name = grouping_cols[0]
        self.target_col_name = target_col_name

        if not pd.api.types.is_numeric_dtype(self.surv_df_mpv[target_col_name]):
            raise ValueError(
                f"'{target_col_name}' must be numeric to search over "
                "thresholds with <=/> comparisons."
            )

    # ------------------------------------------------------------------
    # Core sweep
    # ------------------------------------------------------------------

    def _get_thresholds(self) -> np.ndarray:
        target_col = self.surv_df_mpv[self.target_col_name]

        if self.use_synthetic_cutoffs:
            # Use the true float bounds of the data (not int()-truncated)
            # so the grid always covers the full observed range, and add
            # one extra step so `target_col.max()` itself is included
            # even when it doesn't fall exactly on the grid.
            lo, hi = float(target_col.min()), float(target_col.max())
            thresholds = np.arange(lo, hi + self.search_resolution, self.search_resolution)
            thresholds = thresholds[thresholds <= hi + 1e-9]
        else:
            # Drop the maximum observed value: thresholding at the max
            # always puts every row in the "<=" group (a guaranteed
            # degenerate split), so it's never worth testing.
            sorted_vals = np.sort(target_col.unique())
            thresholds = sorted_vals[:-1] if len(sorted_vals) > 1 else sorted_vals

        return np.unique(thresholds)

    def get_mpv_df(self) -> pd.DataFrame:
        thresholds = self._get_thresholds()
        iterator = tqdm(thresholds) if self.show_progress else thresholds

        mpv_dicts = []
        for threshold in iterator:
            mpv_dict = self._get_mpv_dict(threshold=threshold)
            mpv_dicts.append(mpv_dict)

        return pd.DataFrame(mpv_dicts)

    def _get_mpv_dict(self, threshold: float) -> dict:
        """Run the binary split + log-rank/Cox comparison at a single
        threshold. Instance method (not static) so it can read
        `self.alpha`, `self.surv_label`, etc. directly instead of having
        them threaded through as parameters.
        """
        lo_label = f"<={threshold:.2f}"
        hi_label = f">{threshold:.2f}"

        mpv_dict = {
            "threshold": threshold,
            "group0_label": lo_label,
            "group1_label": hi_label,
            "valid_split": False,  # overwritten to True on success below
        }

        surv_df_binary = self.surv_df_mpv.copy()
        is_low = surv_df_binary[self.target_col_name] <= threshold
        surv_df_binary[self.target_col_name] = np.where(is_low, lo_label, hi_label)

        if surv_df_binary[self.target_col_name].nunique() != 2:
            # Degenerate split: every row landed on one side. Expected to
            # happen at the extremes of the threshold range -- not an
            # error, just not usable.
            return mpv_dict

        try:
            binary_class_surv = BinaryClassSurv(
                surv_df_binary=surv_df_binary,
                surv_label=self.surv_label,
                alpha=self.alpha,
                baseline_group=lo_label,
            )
        except Exception as exc:
            # A single bad threshold (e.g. a Cox convergence failure or a
            # column-naming edge case inside BinaryClassSurv) should not
            # take down a sweep over hundreds of thresholds.
            if self.skip_invalid_thresholds:
                mpv_dict["error"] = f"{type(exc).__name__}: {exc}"
                return mpv_dict
            raise

        # hazard_dict and log_rank_dict each carry a "p_value" key (Cox
        # Wald p-value vs. log-rank p-value respectively). The log-rank
        # p-value is the one this class reports as *the* p-value /
        # "significant" flag, so it's stripped out of hazard_dict before
        # merging to avoid a silent, order-dependent overwrite.
        hazard_dict = {k: v for k, v in binary_class_surv.hazard_dict.items() if k != "p_value"}
        log_rank_dict = binary_class_surv.log_rank_dict

        mpv_dict.update(log_rank_dict)
        mpv_dict["significant"] = mpv_dict["p_value"] < self.alpha
        mpv_dict.update(hazard_dict)

        # Single source of truth for baseline/comparison group identity:
        # read directly from the BinaryClassSurv instance that actually
        # ran the Cox fit, rather than re-deriving it from `threshold`.
        # (hazard_dict["baseline_group"] above is already this same
        # value -- this line is the canonical, readable name for it and
        # makes the "comparison_group" pairing explicit.)
        mpv_dict["baseline_group"] = binary_class_surv.baseline_label
        mpv_dict["comparison_group"] = (
            hi_label if binary_class_surv.baseline_label == lo_label else lo_label
        )
        mpv_dict["valid_split"] = True

        return mpv_dict

    def get_best_threshold(self, mpv_df: pd.DataFrame | None = None) -> pd.Series:
        """Return the row with the smallest (unadjusted) log-rank p-value
        among valid splits.

        Always emits a UserWarning about the multiple-testing problem --
        see the module docstring. This method is a convenience lookup,
        not a statistically corrected "best" threshold.

        Parameters
        ----------
        mpv_df : pandas.DataFrame, optional
            Reuse an already-computed result from `get_mpv_df()` instead
            of recomputing the full sweep.

        Raises
        ------
        ValueError
            If no threshold produced a valid split.
        """
        warnings.warn(
            "get_best_threshold() reports the minimum *unadjusted* "
            "log-rank p-value across all scanned thresholds. This is "
            "subject to the well-known 'minimum p-value' multiple "
            "testing problem (Altman et al. 1994) and will overstate "
            "significance. Treat this threshold as exploratory, and "
            "apply a correction (e.g. permutation testing on the "
            "minimum p-value) before treating it as confirmatory.",
            UserWarning,
            stacklevel=2,
        )

        if mpv_df is None:
            mpv_df = self.get_mpv_df()

        valid = mpv_df[mpv_df["valid_split"]]
        if valid.empty:
            raise ValueError("No threshold produced a valid 2-group split.")

        return valid.loc[valid["p_value"].idxmin()]

    # ------------------------------------------------------------------
    # Binary split inspection at an arbitrary (or best) threshold
    # ------------------------------------------------------------------

    def get_binary_class_at_cutoff(
        self,
        threshold: float,
        baseline_group: str = "largest",
        censoring: str = "right",
    ) -> BinaryClassSurv:
        """Build and fit a `BinaryClassSurv` at an arbitrary threshold,
        independent of the sweep -- e.g. to inspect or plot a specific
        cutoff in detail (full KM curves, at-risk table, HR annotation)
        rather than just reading its row out of `get_mpv_df()`.

        Unlike the sweep's internal `_get_mpv_dict`, which always pins
        `baseline_group` to the "<=threshold" side (so every row's
        hazard ratio is reported on a consistent, "low vs. high" basis
        across the whole sweep), this method lets you choose any
        `baseline_group` value `BinaryClassSurv` accepts ("largest",
        "smallest", "first", "second", or an explicit label) -- this is
        a one-off inspection, not a row that needs to stay comparable
        across a sweep.

        Parameters
        ----------
        threshold : float
            Cutoff to split on: <=threshold vs. >threshold.
        baseline_group : str, default "largest"
            Forwarded to `BinaryClassSurv`.
        censoring : str, default "right"
            Forwarded to `BinaryClassSurv`.

        Returns
        -------
        BinaryClassSurv
            Fitted instance; use `.plot_km_curves(...)`,
            `.hazard_dict`, `.log_rank_dict`, etc.

        Raises
        ------
        ValueError
            If the threshold produces a degenerate (one-sided) split.
        """
        lo_label = f"<={threshold:.2f}"
        hi_label = f">{threshold:.2f}"

        surv_df_binary = self.surv_df_mpv.copy()
        is_low = surv_df_binary[self.target_col_name] <= threshold
        surv_df_binary[self.target_col_name] = np.where(is_low, lo_label, hi_label)

        if surv_df_binary[self.target_col_name].nunique() != 2:
            raise ValueError(
                f"threshold={threshold!r} produces a degenerate split "
                f"(all rows fall on one side) -- choose a threshold "
                f"strictly between the column's min and max."
            )

        return BinaryClassSurv(
            surv_df_binary=surv_df_binary,
            surv_label=self.surv_label,
            alpha=self.alpha,
            baseline_group=baseline_group,
            censoring=censoring,
        )

    def get_binary_class_at_best_threshold(
        self,
        mpv_df: pd.DataFrame | None = None,
        baseline_group: str = "largest",
        censoring: str = "right",
    ) -> BinaryClassSurv:
        """Convenience wrapper: find the best (min raw p-value) threshold
        via `get_best_threshold()`, then build a `BinaryClassSurv` at it
        via `get_binary_class_at_cutoff()`. Inherits the same
        multiple-testing warning from `get_best_threshold()`.
        """
        best_row = self.get_best_threshold(mpv_df=mpv_df)
        return self.get_binary_class_at_cutoff(
            threshold=float(best_row["threshold"]),
            baseline_group=baseline_group,
            censoring=censoring,
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    #
    # All plotting methods below share the same interface as the old
    # module: `plot=True/False` to call `plt.show()`, and an optional
    # `save_path` to `plt.savefig(...)` -- but none of them read from or
    # write to a managed directory/CSV; they take `mpv_df` directly
    # (computing it via `get_mpv_df()` if not passed) and return
    # `(fig, ax)` so callers can keep customizing before showing/saving.
    #
    # None of these correct the multiple-testing problem on their own.
    # Pass a `CorrectionResult` (from `mpv_corrections.py`) via the
    # `correction=` kwarg where available to overlay the adjusted
    # significance threshold / null distribution alongside the raw
    # result, so the inflation is visible rather than implicit.

    @staticmethod
    def _valid_sorted(mpv_df: pd.DataFrame) -> pd.DataFrame:
        return mpv_df[mpv_df["valid_split"]].sort_values("threshold")

    @staticmethod
    def _unpack_hr_ci(valid: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        def _safe_get(ci, i):
            try:
                return float(ci[i])
            except (TypeError, IndexError):
                return np.nan

        lower = valid["hr_ci"].apply(lambda ci: _safe_get(ci, 0)).to_numpy()
        upper = valid["hr_ci"].apply(lambda ci: _safe_get(ci, 1)).to_numpy()
        return lower, upper

    def _finalize_plot(self, fig, save_path, plot):
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
        if plot:
            plt.show()
        else:
            plt.close(fig)

    def plot_p_value_curve(
        self,
        mpv_df: pd.DataFrame | None = None,
        correction=None,
        log_scale: bool = True,
        save_path: str | None = None,
        plot: bool = True,
    ):
        """Threshold vs. log-rank p-value. Replaces the old module's
        `_plot_p_values`: same idea, fixed to use `threshold` (not
        `cutoff`), `self.alpha` (not a hardcoded 0.05), guards against
        an exact-0 p-value breaking the log scale, and -- new -- if a
        `CorrectionResult` (from `mpv_corrections.py`) is passed via
        `correction=`, overlays its adjusted significance line and
        annotates the raw-vs-adjusted gap directly on the plot.
        """
        if mpv_df is None:
            mpv_df = self.get_mpv_df()
        valid = self._valid_sorted(mpv_df)
        if valid.empty:
            raise ValueError("No valid threshold rows to plot.")

        p = np.clip(valid["p_value"].to_numpy(), 1e-300, None) if log_scale else valid["p_value"].to_numpy()

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(valid["threshold"], p, color="tab:blue", lw=1.2, marker=".", label="log-rank p-value")
        ax.axhline(self.alpha, color="tab:red", ls="--", lw=1, label=f"alpha = {self.alpha}")

        best_idx = valid["p_value"].idxmin()
        best_threshold = valid.loc[best_idx, "threshold"]
        best_p = valid.loc[best_idx, "p_value"]
        ax.axvline(best_threshold, color="gray", ls=":", lw=1, label=f"min p @ threshold={best_threshold:.3g}")

        if correction is not None:
            ax.axhline(
                correction.adjusted_p_value, color="darkred", ls="-.", lw=1.4,
                label=f"{correction.method} adjusted p = {correction.adjusted_p_value:.3g}",
            )
            ax.annotate(
                f"raw min p = {best_p:.3g}\nadjusted = {correction.adjusted_p_value:.3g}",
                xy=(best_threshold, best_p), xytext=(0.02, 0.92), textcoords="axes fraction",
                fontsize=9, arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            )

        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("p-value" + (" (log scale)" if log_scale else ""))
        ax.set_title(f"Log-rank p-value across scanned thresholds ({self.target_col_name})")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

        self._finalize_plot(fig, save_path, plot)
        return fig, ax

    def plot_hazard_ratio_curve(
        self,
        mpv_df: pd.DataFrame | None = None,
        save_path: str | None = None,
        plot: bool = True,
    ):
        """Threshold vs. Cox hazard ratio with 95% CI ribbon. New plot --
        the old module never visualized the Cox side, even though a full
        Cox fit happens at every threshold. Unpacks the `hr_ci` list
        column into lower/upper bounds.
        """
        if mpv_df is None:
            mpv_df = self.get_mpv_df()
        valid = self._valid_sorted(mpv_df)
        if "hr" not in valid.columns:
            raise KeyError(f"'hr' not in mpv_df columns: {list(valid.columns)}")

        lower, upper = self._unpack_hr_ci(valid)

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(valid["threshold"], valid["hr"], color="tab:purple", lw=1.2, marker=".", label="Hazard ratio")
        if not np.all(np.isnan(lower)):
            ax.fill_between(valid["threshold"], lower, upper, color="tab:purple", alpha=0.15, label="95% CI")
        ax.axhline(1.0, color="black", lw=0.8, ls="--", label="HR = 1")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Hazard ratio (baseline vs. comparison)")
        ax.set_title(f"Cox hazard ratio across scanned thresholds ({self.target_col_name})")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        self._finalize_plot(fig, save_path, plot)
        return fig, ax

    def plot_group_sizes_and_survival(
        self,
        mpv_df: pd.DataFrame | None = None,
        save_path: str | None = None,
        plot: bool = True,
    ):
        """Two-panel: group sizes (top) and per-group median survival
        (bottom) across thresholds. Direct, corrected replacement for
        the old `_plot_with_counts` (same intent, fixed columns
        `group0_n`/`group1_n`, `group0_median_survival`/
        `group1_median_survival`).
        """
        if mpv_df is None:
            mpv_df = self.get_mpv_df()
        valid = self._valid_sorted(mpv_df)
        required = {"group0_n", "group1_n", "group0_median_survival", "group1_median_survival"}
        missing = required - set(valid.columns)
        if missing:
            raise KeyError(f"mpv_df missing columns: {missing}")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

        ax1.plot(valid["threshold"], valid["group0_n"], color="tab:orange", marker="o", alpha=0.6, label="<=threshold (group0) n")
        ax1.plot(valid["threshold"], valid["group1_n"], color="tab:blue", marker="o", alpha=0.6, label=">threshold (group1) n")
        ax1.set_ylabel("Group size")
        ax1.legend(loc="best", fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.plot(valid["threshold"], valid["group0_median_survival"], color="red", marker="_", label="group0 median survival")
        ax2.plot(valid["threshold"], valid["group1_median_survival"], color="green", marker="_", label="group1 median survival")
        ax2.set_xlabel("Threshold")
        ax2.set_ylabel("Median survival")
        ax2.legend(loc="best", fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f"Group sizes and median survival across thresholds ({self.target_col_name})")

        self._finalize_plot(fig, save_path, plot)
        return fig, (ax1, ax2)

    def plot_group_balance(
        self,
        mpv_df: pd.DataFrame | None = None,
        min_group_fraction: float = 0.1,
        save_path: str | None = None,
        plot: bool = True,
    ):
        """Threshold vs. group0 fraction of n_total, with the
        `min_group_fraction` window used by `lausen_schumacher_correction`
        shaded out. New plot -- makes visible exactly which thresholds
        the LS correction excludes for being too unbalanced.
        """
        if mpv_df is None:
            mpv_df = self.get_mpv_df()
        valid = self._valid_sorted(mpv_df)
        n_total = len(self.surv_df_mpv)
        fraction = valid["group0_n"] / n_total

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(valid["threshold"], fraction, color="tab:orange", lw=1.2, marker=".")
        ax.axhline(min_group_fraction, color="gray", ls="--", lw=1, label=f"min_group_fraction={min_group_fraction}")
        ax.axhline(1 - min_group_fraction, color="gray", ls="--", lw=1)
        ax.fill_between(valid["threshold"], 0, min_group_fraction, color="red", alpha=0.08)
        ax.fill_between(valid["threshold"], 1 - min_group_fraction, 1, color="red", alpha=0.08)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("group0 fraction of n_total")
        ax.set_title("Group balance (red = excluded from Lausen-Schumacher)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        self._finalize_plot(fig, save_path, plot)
        return fig, ax

    def plot_permutation_null(
        self,
        correction,
        save_path: str | None = None,
        plot: bool = True,
    ):
        """Histogram of the permutation null distribution of minimum
        p-values (from `mpv_corrections.permutation_correction`), with
        the observed value marked -- the clearest single visual for "why
        the raw minimum p-value overstates significance."
        """
        if correction.method != "permutation":
            raise ValueError(f"Expected a permutation CorrectionResult, got {correction.method!r}")

        null_p = correction.details["null_min_p_values"]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(null_p, bins=40, color="lightgray", edgecolor="gray", label="Permuted minimum p-values (null)")
        ax.axvline(correction.raw_p_value, color="tab:red", lw=2, label=f"Observed min p = {correction.raw_p_value:.3g}")
        ax.axvline(correction.details["null_5th_percentile"], color="tab:blue", ls="--", lw=1, label="Null 5th percentile")
        ax.set_xlabel("Minimum log-rank p-value")
        ax.set_ylabel("Permutation count")
        ax.set_title(f"Permutation null (n={correction.details['n_permutations']}) -- adjusted p = {correction.adjusted_p_value:.3g}")
        ax.legend(loc="best", fontsize=8)

        self._finalize_plot(fig, save_path, plot)
        return fig, ax

    def plot_correction_comparison(
        self,
        comparison_df: pd.DataFrame,
        save_path: str | None = None,
        plot: bool = True,
    ):
        """Bar chart: raw vs. adjusted p-value per correction method,
        from `mpv_corrections.compare_corrections(...)`'s output.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(comparison_df))
        width = 0.35
        ax.bar(x - width / 2, comparison_df["raw_p_value"], width, label="raw p-value", color="lightcoral")
        ax.bar(x + width / 2, comparison_df["adjusted_p_value"], width, label="adjusted p-value", color="firebrick")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df["method"], rotation=15)
        ax.set_ylabel("p-value (log scale)")
        ax.set_title("Raw vs. multiplicity-adjusted p-value by method")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

        self._finalize_plot(fig, save_path, plot)
        return fig, ax

    def plot_km_at_best_threshold(
        self,
        mpv_df: pd.DataFrame | None = None,
        save_path: str | None = None,
        plot: bool = True,
        **plot_km_kwargs,
    ):
        """Find the best (min raw p-value) threshold and plot its KM
        curves via `BinaryClassSurv.plot_km_curves()` -- reuses the
        existing, full-featured KM plot (at-risk table, HR annotation)
        rather than reimplementing it here.

        `save_path`/`plot` are translated to `BinaryClassSurv`'s own
        `savepath`/`plot` kwargs so the calling convention stays
        consistent with the rest of this class's plotting methods.
        Extra `plot_km_kwargs` (e.g. `title=`, `x_axis_range=`) are
        passed straight through.

        Note: this calls `get_binary_class_at_best_threshold`, which in
        turn calls `get_best_threshold` and therefore emits the usual
        multiple-testing `UserWarning`.
        """
        bcs = self.get_binary_class_at_best_threshold(mpv_df=mpv_df)
        plot_km_kwargs.setdefault(
            "title",
            f"Best threshold ({self.target_col_name}, raw p = {bcs.log_rank_dict['p_value']:.3g})",
        )
        bcs.plot_km_curves(savepath=save_path, plot=plot, **plot_km_kwargs)
        return bcs

    def plot_dashboard(
        self,
        mpv_df: pd.DataFrame | None = None,
        correction=None,
        min_group_fraction: float = 0.1,
        save_path: str | None = None,
        plot: bool = True,
    ):
        """Combined 2x2: p-value curve, HR curve, group balance, and
        (if a permutation `CorrectionResult` is passed) its null
        histogram. One call instead of generating/saving three separate
        figures.
        """
        if mpv_df is None:
            mpv_df = self.get_mpv_df()

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Reuse the single-plot methods on the *same* fig/ax by drawing
        # into existing axes -- temporarily monkeypatch via local calls
        # to plt.gca() isn't needed since each helper builds its own
        # figure; instead, draw directly here using the same logic to
        # keep everything on one figure.
        valid = self._valid_sorted(mpv_df)

        # Panel 1: p-value curve
        ax = axes[0, 0]
        p = np.clip(valid["p_value"].to_numpy(), 1e-300, None)
        ax.plot(valid["threshold"], p, color="tab:blue", lw=1.2, marker=".", label="log-rank p-value")
        ax.axhline(self.alpha, color="tab:red", ls="--", lw=1, label=f"alpha = {self.alpha}")
        best_idx = valid["p_value"].idxmin()
        best_threshold = valid.loc[best_idx, "threshold"]
        ax.axvline(best_threshold, color="gray", ls=":", lw=1, label=f"min p @ threshold={best_threshold:.3g}")
        if correction is not None:
            ax.axhline(
                correction.adjusted_p_value, color="darkred", ls="-.", lw=1.4,
                label=f"{correction.method} adjusted p = {correction.adjusted_p_value:.3g}",
            )
        ax.set_yscale("log")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("p-value (log scale)")
        ax.set_title("Log-rank p-value")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, which="both", alpha=0.3)

        # Panel 2: hazard ratio curve
        ax = axes[0, 1]
        if "hr" in valid.columns:
            lower, upper = self._unpack_hr_ci(valid)
            ax.plot(valid["threshold"], valid["hr"], color="tab:purple", lw=1.2, marker=".", label="Hazard ratio")
            if not np.all(np.isnan(lower)):
                ax.fill_between(valid["threshold"], lower, upper, color="tab:purple", alpha=0.15, label="95% CI")
            ax.axhline(1.0, color="black", lw=0.8, ls="--", label="HR = 1")
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Hazard ratio")
            ax.set_title("Cox hazard ratio")
            ax.legend(loc="best", fontsize=7)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "no 'hr' column", ha="center", va="center")
            ax.set_axis_off()

        # Panel 3: group balance
        ax = axes[1, 0]
        n_total = len(self.surv_df_mpv)
        fraction = valid["group0_n"] / n_total
        ax.plot(valid["threshold"], fraction, color="tab:orange", lw=1.2, marker=".")
        ax.axhline(min_group_fraction, color="gray", ls="--", lw=1)
        ax.axhline(1 - min_group_fraction, color="gray", ls="--", lw=1)
        ax.fill_between(valid["threshold"], 0, min_group_fraction, color="red", alpha=0.08)
        ax.fill_between(valid["threshold"], 1 - min_group_fraction, 1, color="red", alpha=0.08)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("group0 fraction")
        ax.set_title("Group balance")
        ax.grid(True, alpha=0.3)

        # Panel 4: permutation null (if available)
        ax = axes[1, 1]
        if correction is not None and correction.method == "permutation":
            null_p = correction.details["null_min_p_values"]
            ax.hist(null_p, bins=40, color="lightgray", edgecolor="gray")
            ax.axvline(correction.raw_p_value, color="tab:red", lw=2, label=f"observed = {correction.raw_p_value:.3g}")
            ax.set_xlabel("Minimum log-rank p-value")
            ax.set_ylabel("Permutation count")
            ax.set_title(f"Permutation null (adj. p={correction.adjusted_p_value:.3g})")
            ax.legend(loc="best", fontsize=7)
        else:
            ax.text(0.5, 0.5, "pass a permutation CorrectionResult\nto see the null distribution", ha="center", va="center")
            ax.set_axis_off()

        fig.suptitle(f"MinimumPValue sweep diagnostics ({self.target_col_name})", fontsize=14)
        fig.tight_layout()

        self._finalize_plot(fig, save_path, plot)
        return fig, axes