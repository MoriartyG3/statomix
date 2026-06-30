"""
Single-arm survival analysis utilities built on top of lifelines.

This module wraps lifelines' KaplanMeierFitter to compute the standard set of
descriptive survival statistics for one cohort/arm: median survival time,
median potential follow-up time (reverse Kaplan-Meier / Schemper-Smith
method), survival probability at a given time point with confidence
intervals, and restricted mean survival time (RMST), plus a KM plot with an
at-risk table.

Expected input
---------------
`surv_df` must be a pandas DataFrame with exactly two relevant columns:
    - "time":  non-negative numeric follow-up duration for each subject.
    - "event": event indicator for each subject. Accepted as boolean
               (True = event occurred, False = censored) or as 0/1 integers
               (1 = event occurred, 0 = censored). Any other encoding raises
               a ValueError at construction time -- this is deliberate,
               because a silently misread event column produces survival
               numbers that look plausible but are wrong (see "Why
               validation matters" below).

Why validation matters
-----------------------
Two correctness bugs motivated the validation added here:

1. `~self.surv_df["event"]` (bitwise NOT) only behaves like logical negation
   when the column's dtype is actually `bool`. If "event" is stored as
   int64 (0/1), `~0 == -1` and `~1 == -2` -- both are truthy/non-zero, so
   lifelines would treat *every* row as an event when fitting the
   reverse-KM curve used for median follow-up. The result is a confidently
   wrong number with no error raised. Coercing "event" to a real boolean
   Series once, in `__init__`, removes this failure mode everywhere it's
   used downstream.

2. Computing a point survival estimate with `KaplanMeierFitter.predict()`
   (a step-function / right-continuous lookup, `interpolate=False` by
   default) while computing its confidence interval with `scipy.interp1d`
   (straight-line interpolation between knots) mixes two different
   estimators. The point estimate and its CI can therefore disagree with
   each other, particularly between two consecutive event times. Both are
   now read off the same step function via `pandas.Series.asof`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times, restricted_mean_survival_time
from lifelines.plotting import add_at_risk_counts


class SingleClassSurv:
    """Descriptive Kaplan-Meier survival analysis for a single cohort.

    Parameters
    ----------
    surv_label : str
        Human-readable label for this cohort (used as the lifelines fit
        label and in plot legends).
    surv_df : pandas.DataFrame
        Must contain a "time" column (non-negative numeric durations) and
        an "event" column (boolean, or 0/1 integers). Rows with missing
        values in either column are dropped, with a warning, before
        fitting.

    Attributes
    ----------
    kmf : lifelines.KaplanMeierFitter
        The fitted KM estimator. Populated by `_fit()`, which runs
        automatically at the end of `__init__`.
    surv_df : pandas.DataFrame
        Cleaned copy of the input data (NaNs dropped, "event" coerced to
        bool).
    max_followup_time : float
        Maximum observed "time" value in the cleaned data. Used as the
        default truncation point for RMST.
    descriptives : dict
        Populated by `_fit()`. Holds both formatted strings and raw
        (unformatted) numeric values for median survival, median
        follow-up, survival probabilities at requested time points, and
        (if requested) RMST.

        "median_survival" and "median_follow_up" are each a dict:
            {"label": "<formatted string>",
             "raw": {"median": float, "ci_lower": float, "ci_upper": float}}
        where "raw" values use `np.inf` (not the string "not reached")
        for any bound the KM curve never crossed -- check with
        `np.isfinite(...)` downstream rather than string-matching.

        "Surv Probability" is a dict keyed by the requested time point,
        each value shaped like:
            {"label": "<formatted string>",
             "raw": {"survival_prob": float, "ci_lower": float, "ci_upper": float}}
        using `np.nan` (not `np.inf`) for time points outside the
        observed range, since that's a genuinely unknown value rather
        than "later than anything observed".

        "rmst" is a dict keyed by `restricted_time`, with the shape
        returned by `get_rmst()` (see that method's docstring) -- this
        one keeps its original flatter shape ("rmst", "95% ci", etc. as
        top-level keys) rather than the {"label", "raw"} nesting used
        above, since RMST already separates its raw numeric values from
        the optional formatted "label"/"bootstrap_label" strings.

    Raises
    ------
    ValueError
        If required columns are missing, "time" contains negative or
        non-numeric values, or "event" contains values other than
        {0, 1, True, False}.

    Notes
    -----
    `__init__` calls `_fit()` automatically, so the KM curve and
    `self.descriptives` are always populated by the time the constructor
    returns -- there is no separate `run()` step. `_require_fitted()` is
    kept as a defensive guard for `get_rmst()`, `get_survival_probability()`,
    and `plot_km_curve()` in case a subclass ever overrides `__init__` and
    skips `_fit()`; it raises a clear `RuntimeError` rather than letting
    those methods fail later with an unrelated `AttributeError`.
    """

    def __init__(self, surv_label: str, surv_df: pd.DataFrame):
        required_cols = {"time", "event"}
        missing = required_cols - set(surv_df.columns)
        if missing:
            raise ValueError(
                f"surv_df is missing required column(s): {sorted(missing)}"
            )

        df = surv_df[["time", "event"]].copy()

        n_before = len(df)
        df = df.dropna(subset=["time", "event"])
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(
                f"[SingleClassSurv] Dropped {n_dropped} row(s) with missing "
                f"'time' or 'event' values."
            )

        if not pd.api.types.is_numeric_dtype(df["time"]):
            raise ValueError("'time' column must be numeric.")
        if (df["time"] < 0).any():
            raise ValueError("'time' column contains negative durations.")

        df["event"] = self._coerce_event_to_bool(df["event"])

        self.kmf = KaplanMeierFitter()
        self.surv_label = surv_label
        self.surv_df = df.reset_index(drop=True)
        self.max_followup_time = self.surv_df["time"].max()
        self.descriptives = None  # populated by _fit()

        self._fit()

    @staticmethod
    def _coerce_event_to_bool(event_series: pd.Series) -> pd.Series:
        """Safely coerce an event indicator column to true boolean dtype.

        Accepts boolean dtype as-is. Accepts integer/float columns only if
        every value is exactly 0 or 1 (1 = event, 0 = censored). Rejects
        anything else loudly, rather than silently misinterpreting it --
        this is the single check that prevents the `~` bitwise-vs-logical
        bug described in the module docstring.
        """
        if pd.api.types.is_bool_dtype(event_series):
            return event_series.astype(bool)

        unique_vals = set(pd.unique(event_series))
        if unique_vals <= {0, 1}:
            return event_series.astype(bool)

        raise ValueError(
            "'event' column must be boolean, or numeric containing only "
            f"0/1. Found unexpected values: {sorted(unique_vals)}"
        )

    def _fit(self) -> None:
        """Fit the Kaplan-Meier curve and compute the standard descriptives.

        Populates `self.kmf` and `self.descriptives` with:
            - "median_survival": dict with "label" (formatted string) and
              "raw" (dict of unformatted floats: "median", "ci_lower",
              "ci_upper"; "not reached" is represented as `np.inf` in
              "raw", never as a string).
            - "median_follow_up": same shape as "median_survival", for the
              median potential follow-up time (reverse Kaplan-Meier /
              Schemper-Smith method).
            - "Surv Probability": empty dict, filled in by calls to
              `get_survival_probability(time_point)`.

        Called automatically from `__init__`. Must have run (it always
        will have, by construction) before `get_rmst()`,
        `get_survival_probability()`, or `plot_km_curve()` are called --
        `_require_fitted()` enforces this defensively.
        """
        self.kmf.fit(
            durations=self.surv_df["time"],
            event_observed=self.surv_df["event"],
            label=self.surv_label,
        )
        self.descriptives = {}
        self.descriptives["median_survival"] = self._get_median_survival()
        self.descriptives["median_follow_up"] = self._get_median_follow_up()
        self.descriptives["Surv Probability"] = {}
        self.descriptives["rmst"] = {}

    def _require_fitted(self) -> None:
        """Defensive guard: raise a clear RuntimeError if `_fit()` has not
        run yet, instead of letting callers hit an unrelated AttributeError
        or KeyError deeper in lifelines.

        Under normal use this can never actually trigger, since `__init__`
        always calls `_fit()` before returning. It exists for robustness
        against subclasses that might override `__init__` and forget to
        call `_fit()`.
        """
        if self.descriptives is None:
            raise RuntimeError(
                "_fit() must be called before this method (no fitted "
                "KaplanMeierFitter / descriptives available yet)."
            )

    @staticmethod
    def _format_median(median_value: float, ci_lower: float, ci_upper: float) -> dict:
        """Format a median + 95% CI, handling the case where the median
        survival was never reached (median_value is +inf because the KM
        curve never drops to 50%).

        Returns
        -------
        dict
            "label": the formatted display string, e.g.
                "86.73 (95% CI, 82.80 - not reached)".
            "raw": dict with the unformatted float values --
                {"median": ..., "ci_lower": ..., "ci_upper": ...} -- each
                either a plain float or `np.inf` (never the string "not
                reached"). `np.inf` is used rather than `np.nan` because
                an unreached median is a real, well-defined concept
                ("later than anything observed"), distinct from a missing
                or invalid value. Use `np.isfinite(...)` downstream to
                detect "not reached" in the raw values, exactly as this
                method does internally.
        """
        median_str = "not reached" if not np.isfinite(median_value) else f"{median_value:.2f}"
        lower_str = "not reached" if not np.isfinite(ci_lower) else f"{ci_lower:.2f}"
        upper_str = "not reached" if not np.isfinite(ci_upper) else f"{ci_upper:.2f}"
        label = f"{median_str} (95% CI, {lower_str} - {upper_str})"

        return {
            "label": label,
            "raw": {
                "median": float(median_value),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
            },
        }

    def _get_median_survival(self) -> dict:
        """Median survival time (time at which the KM curve crosses 50%
        survival probability), with its 95% confidence interval.

        Returns
        -------
        dict
            Same shape as `_format_median`'s return value: a "label"
            string and a "raw" dict of unformatted floats (using
            `np.inf` for "not reached").
        """
        median_survival = self.kmf.median_survival_time_
        median_ci = median_survival_times(self.kmf.confidence_interval_)

        ci_lower = median_ci[f"{self.kmf.label}_lower_0.95"][0.5]
        ci_upper = median_ci[f"{self.kmf.label}_upper_0.95"][0.5]

        return self._format_median(median_survival, ci_lower, ci_upper)

    def get_survival_probability(self, time_point: float) -> dict:
        """Estimate the survival probability at `time_point`, with its 95%
        confidence interval, and store it in
        `self.descriptives["Surv Probability"][time_point]`.

        Both the point estimate and the CI bounds are read off the same
        right-continuous step function (last observed value at or before
        `time_point`), so they are guaranteed to be mutually consistent.
        `time_point`s outside the observed time range are recorded as NaN
        rather than extrapolated, since KM provides no information beyond
        the last observation.

        Stores a dict with "label" (formatted string) and "raw" (dict of
        unformatted floats: "survival_prob", "ci_lower", "ci_upper";
        `np.nan` for out-of-range `time_point`s -- distinct from the
        `np.inf`/"not reached" convention used by the median getters,
        since an out-of-range probability isn't "later than observed", it
        is genuinely unknown).
        """
        self._require_fitted()

        ci_df = self.kmf.confidence_interval_
        time_points = ci_df.index

        if time_point < time_points[0] or time_point > time_points[-1]:
            survival_prob, ci_lower, ci_upper = np.nan, np.nan, np.nan
        else:
            survival_prob = self.kmf.predict(time_point)  # step-function lookup
            ci_lower = ci_df[f"{self.kmf.label}_lower_0.95"].asof(time_point)
            ci_upper = ci_df[f"{self.kmf.label}_upper_0.95"].asof(time_point)

        label = f"{round(survival_prob, 2)} (95% CI, {ci_lower:.2f} - {ci_upper:.2f})"

        self.descriptives["Surv Probability"][time_point] = {
            "label": label,
            "raw": {
                "survival_prob": float(survival_prob),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
            },
        }

        return self.descriptives

    def _get_median_follow_up(self) -> dict:
        """Median potential follow-up time, via the reverse Kaplan-Meier
        (Schemper & Smith, 1996) method.

        Method: refit a KM curve on the same durations but with the event
        indicator flipped, so that censoring becomes the "event of
        interest" and actual events become "censoring". The median of
        *this* curve estimates how long subjects would have been followed
        had the event of interest never occurred -- i.e. the median
        potential follow-up time. This is the standard way to report
        follow-up duration for a survival cohort (as opposed to a naive
        median of the raw "time" column, which is biased downward because
        it ignores the fact that event times themselves are also a form of
        truncated observation).

        Returns
        -------
        dict
            Same shape as `_format_median`'s return value: a "label"
            string and a "raw" dict of unformatted floats (using
            `np.inf` for "not reached").
        """
        label = f"{self.surv_label} Follow-Up"
        followup_kmf = KaplanMeierFitter()

        # self.surv_df["event"] is guaranteed real bool dtype (coerced in
        # __init__), so logical negation via `~` is safe here.
        followup_kmf.fit(
            durations=self.surv_df["time"],
            event_observed=~self.surv_df["event"],
            label=label,
        )

        median_followup = followup_kmf.median_survival_time_
        median_ci = median_survival_times(followup_kmf.confidence_interval_)

        ci_lower = median_ci[f"{label}_lower_0.95"][0.5]
        ci_upper = median_ci[f"{label}_upper_0.95"][0.5]

        return self._format_median(median_followup, ci_lower, ci_upper)

    def get_rmst(
        self,
        restricted_time: float | None = None,
        bootstrap_ci: bool = False,
        n_bootstraps: int = 1000,
        show_progress: bool = True,
        random_seed: int | None = 42,
    ) -> dict:
        """Compute the restricted mean survival time (RMST) up to
        `restricted_time`.

        RMST is the area under the KM curve from 0 to `restricted_time`. It
        summarizes mean survival without requiring the curve to reach 50%
        (unlike median survival, which is undefined/"not reached" when it
        doesn't), and is the recommended summary statistic when comparing
        treatment arms whose KM curves cross.

        Parameters
        ----------
        restricted_time : float, optional
            Truncation time for the RMST integral. Defaults to
            `self.max_followup_time` (the latest observed time in the
            data) -- the standard, conservative default, since lifelines
            cannot extrapolate the curve beyond the last observation.
        bootstrap_ci : bool, default False
            If True, estimate a 95% CI for the RMST via case resampling
            with replacement (`n_bootstraps` resamples of the full cohort,
            each refit with its own KaplanMeierFitter).
        n_bootstraps : int, default 1000
        show_progress : bool, default True
        random_seed : int or None, default 42
            Base seed for reproducible bootstrap resampling. Each bootstrap
            iteration uses `random_seed + iteration_index` (when not None),
            so results are reproducible across runs without every resample
            being identical.

        Returns
        -------
        dict
            Always contains "restricted_time" and "rmst" (point estimate).
            When `bootstrap_ci=True`, also contains "95% ci"
            (`[lower, upper]` as a numpy array), "mean_rmst" (mean of the
            bootstrap distribution), and formatted "label"/"bootstrap_label"
            strings.

        Raises
        ------
        RuntimeError
            If called before `_fit()` has run.
        """
        self._require_fitted()

        if restricted_time is None:
            restricted_time = self.max_followup_time

        rmst_dict = {"restricted_time": restricted_time}
        rmst_dict["rmst"] = restricted_mean_survival_time(self.kmf, t=restricted_time)

        if bootstrap_ci:
            rmst_samples = []
            iterator = tqdm(range(n_bootstraps)) if show_progress else range(n_bootstraps)

            for idx in iterator:
                random_state = random_seed + idx if random_seed is not None else None

                bootstrap_sample = self.surv_df.sample(
                    n=len(self.surv_df), replace=True, random_state=random_state
                )
                kmf_bootstrap = KaplanMeierFitter()
                kmf_bootstrap.fit(
                    bootstrap_sample["time"], event_observed=bootstrap_sample["event"]
                )
                rmst_bootstrap = restricted_mean_survival_time(
                    kmf_bootstrap, t=restricted_time
                )
                rmst_samples.append(rmst_bootstrap)

            ci = np.percentile(rmst_samples, [2.5, 97.5])
            mean_rmst = np.mean(rmst_samples)

            rmst_dict["95% ci"] = ci
            rmst_dict["mean_rmst"] = mean_rmst
            rmst_dict["bootstrap_label"] = f"{mean_rmst:.2f} (95% CI, {ci[0]:.2f} - {ci[1]:.2f})"
            rmst_dict["label"] = f"{rmst_dict['rmst']:.2f} (95% CI, {ci[0]:.2f} - {ci[1]:.2f})"

        self.descriptives["rmst"][restricted_time] = rmst_dict
        return rmst_dict

    def plot_km_curve(
        self,
        table_title: str = "Risk Table",
        xlabel: str = "Time (Months)",
        ylabel: str = "Survival Probability",
        plot: bool = True,
        title: str | None = None,
        savepath: str | None = None,
        plot_grid: bool = True,
        x_axis_range=None,
        add_risk_table: bool = True,
        plot_whole_y_axis: bool = True,
        print_median_survival: bool = True,
    ) -> None:
        """Plot the Kaplan-Meier curve with a 50% reference line, optional
        at-risk table, and optional median-survival annotation.
 
        Parameters
        ----------
        xlabel : str, default "Time (Months)"
            NOTE: this default assumes time is recorded in months (it also
            drives the default tick spacing of 12 units, below). If your
            "time" column uses a different unit, pass a matching `xlabel`
            and `x_axis_range`.
        x_axis_range : iterable, optional
            Defaults to `range(0, int(max_observed_time) + 1, 12)`, i.e.
            ticks every 12 time units -- sensible for monthly data, not for
            other units. Override explicitly for non-monthly time scales.
        savepath : str, optional
            If given, save the figure to this path via `plt.savefig`.
 
        Raises
        ------
        RuntimeError
            If called before `_fit()` has run.
        """
        self._require_fitted()
 
        self.kmf.plot_survival_function(ci_show=False, legend=False)
 
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
 
        if title is not None:
            plt.title(title)
 
        if plot_whole_y_axis:
            plt.ylim(0, 1)
 
        if plot_grid:
            plt.grid(True)
 
        if x_axis_range is None:
            max_time = self.kmf.event_table.index[-1]
            x_axis_range = range(0, int(max_time) + 1, 12)
 
        # Make x_axis_range concrete (it may be a `range` object) so the
        # same list of tick values can be reused below for both the
        # visible axis ticks and the at-risk table's column positions.
        x_axis_range = list(x_axis_range)
        plt.xticks(x_axis_range)
 
        plt.axhline(y=0.5, color="red", linestyle="--")
 
        if print_median_survival:
            plt.text(
                0.05,
                0.15,
                f"Median survival, {self.descriptives['median_survival']['label']}",
                fontsize=9,
                verticalalignment="top",
            )
 
        if add_risk_table:
            # IMPORTANT: pass `xticks=x_axis_range` explicitly. Without
            # it, add_at_risk_counts reads whatever `ax.get_xticks()`
            # happens to be at call time (lifelines source, plotting.py:
            # `if xticks is None: xticks = ax.get_xticks()`), which can
            # silently diverge from the `plt.xticks(x_axis_range)` call
            # above once matplotlib's layout/autolocator re-runs (e.g.
            # during tight_layout()). That mismatch is what produces a
            # risk-table row with one column per *event time* instead of
            # one column per intended tick -- the dense, overlapping
            # table seen when this isn't pinned down explicitly.
            add_at_risk_counts(self.kmf, labels=[table_title], xticks=x_axis_range)
 
        plt.tight_layout()
 
        if savepath is not None:
            plt.savefig(savepath)
 
        if plot:
            plt.show()
        else:
            plt.close()