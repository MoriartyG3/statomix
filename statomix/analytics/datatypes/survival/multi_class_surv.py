"""
Multi-arm (3+ groups) survival comparison built on top of SingleClassSurv
and lifelines, following the same patterns as BinaryClassSurv.

This module compares K >= 2 survival cohorts defined by one categorical
grouping column. It generalizes BinaryClassSurv in three ways that aren't
just "loop the binary class K times":

1. Omnibus significance test. With K groups there is no single log-rank
   p-value comparing "group A vs group B" -- the standard omnibus test is
   the multivariate log-rank test (lifelines.statistics.
   multivariate_logrank_test), which tests the null hypothesis that all K
   survival curves are identical. That's what `get_log_rank_dict()`
   reports as "p_value". Pairwise log-rank p-values (each group vs
   baseline) are also computed and stored separately under
   "pairwise_vs_baseline", since the omnibus test alone doesn't tell you
   *which* group differs.

2. Hazard ratios are all read off ONE Cox model, not K-1 separate ones.
   All non-baseline groups are one-hot encoded as covariates in a single
   CoxPHFitter fit (baseline group's dummy column dropped, exactly as
   BinaryClassSurv does for its one comparison group). Each non-baseline
   group therefore gets its own hazard ratio vs baseline, all estimated
   jointly and sharing one baseline hazard -- this is the standard way to
   report "K-1 hazard ratios vs a reference arm" and avoids the subtly
   different (and not directly comparable) baseline hazard each pairwise
   Cox model would otherwise have.

3. Baseline resolution generalizes BinaryClassSurv's keyword scheme
   ("largest"/"smallest"/"first"/"second"/explicit label) to K categories:
   "largest"/"smallest" compare counts across all K groups, "first"/
   "second" use first-seen `.unique()` order (so "second" only makes
   sense as a keyword when there are >= 2 groups, exactly as it did
   for K=2). Group identity (group0 = baseline, group1..group{K-1} = the
   rest, in first-seen order with the baseline removed) is, as in
   BinaryClassSurv, a deterministic function of resolved baseline_group
   and the category labels/counts -- never of row order in the input
   DataFrame.

Parameters
----------
surv_df_multi : pandas.DataFrame
    Must contain exactly three relevant columns:
        - "time": non-negative numeric follow-up duration.
        - "event": event indicator (boolean, or 0/1 integers).
        - exactly one additional grouping column with >= 2 unique values
          (K == 2 is allowed and falls back to the same behavior as
          BinaryClassSurv, just expressed through this class's dict
          shapes; K == 1 raises).
surv_label : str
    Human-readable label, passed through to each group's SingleClassSurv
    fit (legend label / fit name -- shared across groups, exactly as in
    BinaryClassSurv, since SingleClassSurv distinguishes curves by which
    `surv_df` it was given, not by a per-group label).
alpha : float, default 0.05
    Significance level used for the Cox model's confidence intervals.
baseline_group : {"largest", "smallest", "first", "second"} or any actual
    category value found in the grouping column, default "largest".
    Same semantics as BinaryClassSurv, generalized to K categories:
        - "largest"  -> the group with the most observations (ties go to
                         whichever category appears first in `.unique()`).
        - "smallest" -> the group with the fewest observations.
        - "first"    -> the first-seen category in the data
                         (`.unique()` order).
        - "second"   -> the second-seen category in the data.
        - any other value is matched against the actual category labels
          found in the grouping column, so you can pin the baseline
          explicitly, e.g. baseline_group="control".

Group identity
--------------
`group_labels` maps {0: baseline_label, 1: <2nd category>, ..., K-1: <Kth
category>}, where index 0 is always the resolved baseline (see
`baseline_group` above) and indices 1..K-1 are the remaining categories in
first-seen `.unique()` order with the baseline removed. As in
BinaryClassSurv, baseline_group is resolved once, against the category
labels and counts found in the data, BEFORE any group is split out of
surv_df_multi -- so group identity never depends on row order in the
input DataFrame.

Attributes
----------
target_col_name : str
    Name of the grouping column.
n_groups : int
    K, the number of distinct categories (>= 2).
group_labels : dict
    {0: baseline_label, 1: ..., K-1: ...} -- see "Group identity" above.
baseline_label : Any
    The actual category value resolved as both the Cox-model reference
    level and group_labels[0]. Single source of truth for "which group
    is baseline".
baseline_idx : int
    Always 0, kept as an attribute for symmetry with BinaryClassSurv.
surv_dfs : dict
    {0: <time/event DataFrame for baseline>, 1: ..., K-1: ...}.
kms : dict
    {0: <fitted SingleClassSurv>, 1: ..., K-1: ...}, populated by
    `get_log_rank_dict()` (mirrors BinaryClassSurv.km0/km1, generalized
    to a dict since K is not fixed at 2).
split_valid : bool
    True only if every group individually passes the same per-group
    checks BinaryClassSurv applies pairwise (group size > 1, at least one
    event, at least one censored observation) -- see
    `_checks_group_split_validity`.
split_invalid_reason : str or None
    First failing reason found, in the same vocabulary as
    BinaryClassSurv ("group_size_1", "no_events", "no_censoring"), plus
    which group triggered it.

Raises
------
ValueError
    If required columns are missing, there isn't exactly one grouping
    column, that column has fewer than 2 unique values, or
    `baseline_group` doesn't resolve to a known option or category.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts

from fileverse.logger import Logger

from .single_class_surv import SingleClassSurv
from .formatting import get_p_value_label, interpret_hazard_ratio

# from statomix.analytics.datatypes.time_to_event.single_class_surv import SingleClassSurv
# from statomix.analytics.datatypes.time_to_event.formatting import get_p_value_label, interpret_hazard_ratio

logger = Logger(name="MultiClassSurv").get_logger()


class MultiClassSurv:
    """Compares K >= 2 survival cohorts defined by a single categorical
    column. See module docstring for full parameter/attribute reference.
    """

    _BASELINE_KEYWORDS = ("largest", "smallest", "first", "second")

    def __init__(
        self,
        surv_df_multi: pd.DataFrame,
        surv_label: str,
        alpha: float = 0.05,
        baseline_group: str = "largest",
        censoring: str = "right",
    ):
        required_cols = {"time", "event"}
        missing = required_cols - set(surv_df_multi.columns)
        if missing:
            raise ValueError(
                f"surv_df_multi is missing required column(s): {sorted(missing)}"
            )
        self.alpha = alpha
        self.censoring = censoring
        self.surv_label = surv_label

        grouping_cols = [
            c for c in surv_df_multi.columns if c not in ("time", "event")
        ]
        if len(grouping_cols) != 1:
            error_msg = (
                "Expected exactly one grouping column besides 'time'/'event', "
                + f"\nfound {len(grouping_cols)}: {grouping_cols}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        target_col_name = grouping_cols[0]

        # first-seen order; used only to resolve "first"/"second" keywords
        # and to fix the order of non-baseline groups -- NOT to decide
        # which category becomes baseline. See _resolve_baseline_group /
        # "Group identity" in the module docstring.
        categories = list(pd.unique(surv_df_multi[target_col_name]))
        if len(categories) < 2:
            raise ValueError(
                f"Expected at least 2 groups in '{target_col_name}', got "
                f"{len(categories)}: {sorted(categories)}"
            )

        self.target_col_name = target_col_name
        self.surv_df_multi = surv_df_multi
        self.n_groups = len(categories)

        baseline_label, ordered_labels = self._resolve_baseline_group(
            baseline_group=baseline_group,
            categories=categories,
            surv_df_multi=surv_df_multi,
        )

        # group_labels[0] is always the resolved baseline; the rest keep
        # first-seen order with the baseline removed.
        self.group_labels = {i: label for i, label in enumerate(ordered_labels)}
        self.baseline_idx = 0
        self.baseline_label = baseline_label

        self.surv_dfs = {}
        for idx, label in self.group_labels.items():
            mask = surv_df_multi[target_col_name] == label
            self.surv_dfs[idx] = surv_df_multi[mask][["time", "event"]].copy()

        self._checks_group_split_validity()

    def _checks_group_split_validity(self):
        """Per-group validity check, generalizing BinaryClassSurv's
        pairwise check to K groups: every group individually must have
        >1 row, >=1 event, and >=1 censored observation. The first group
        (in group_labels order) that fails sets split_invalid_reason,
        using the same reason vocabulary BinaryClassSurv uses
        ("group_size_1", "no_events", "no_censoring"), prefixed with
        which group triggered it so a K-group failure is still
        diagnosable.
        """
        self.split_valid = True
        self.split_invalid_reason = None

        for idx, df in self.surv_dfs.items():
            n = len(df)
            n_events = df["event"].sum()
            n_censored = n - n_events
            label = self.group_labels[idx]

            if n <= 1:
                reason = "group_size_1"
            elif n_events == 0:
                reason = "no_events"
            elif n_censored == 0:
                reason = "no_censoring"
            else:
                continue

            self.split_valid = False
            self.split_invalid_reason = reason
            logger.warning(
                f"[INVALID SPLIT] {reason}: group{idx} ({label!r}) "
                f"n={n}, events={n_events}, censored={n_censored}"
            )
            break

    def _resolve_baseline_group(self, baseline_group, categories, surv_df_multi) -> tuple:
        """Resolve `baseline_group` to (baseline_label, ordered_labels),
        where ordered_labels[0] == baseline_label and ordered_labels[1:]
        are the remaining categories in first-seen order.

        Resolved against `categories` (first-seen order) and group counts
        in `surv_df_multi` -- NOT against any per-group split, since that
        split doesn't exist yet when this runs. Generalizes
        BinaryClassSurv._resolve_baseline_group from 2 to K categories.
        """
        counts = {
            cat: int((surv_df_multi[self.target_col_name] == cat).sum())
            for cat in categories
        }

        if baseline_group == "largest":
            baseline_label = max(categories, key=lambda c: counts[c])
        elif baseline_group == "smallest":
            baseline_label = min(categories, key=lambda c: counts[c])
        elif baseline_group == "first":
            baseline_label = categories[0]
        elif baseline_group == "second":
            if len(categories) < 2:
                raise ValueError(
                    "baseline_group='second' requires at least 2 categories, "
                    f"found {len(categories)}."
                )
            baseline_label = categories[1]
        elif baseline_group in categories:
            baseline_label = baseline_group
        else:
            error_msg = (
                f"baseline_group={baseline_group!r} is not a recognized keyword "
                + f"\n({self._BASELINE_KEYWORDS}) and does not match any category"
                + f"\nfound in '{self.target_col_name}': "
                + f"\n{sorted(categories, key=str)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        ordered_labels = [baseline_label] + [c for c in categories if c != baseline_label]
        return baseline_label, ordered_labels

    def get_tests_dict(self):
        tests_dict = getattr(self, "tests_dict", None)
        if tests_dict is not None:
            return tests_dict

        tests_dict = {}
        tests_dict["group_sizes"] = {
            self.group_labels[idx]: len(df) for idx, df in self.surv_dfs.items()
        }
        tests_dict["cox_ph"] = self.get_cox_ph_dict()
        tests_dict["log_rank"] = self.get_log_rank_dict()

        self.tests_dict = tests_dict
        return tests_dict

    def get_log_rank_dict(self):
        """Fit a SingleClassSurv per group, run the omnibus multivariate
        log-rank test across all K groups, and additionally run a
        pairwise log-rank test of each non-baseline group against
        baseline (since the omnibus test rejects "all curves equal" but
        does not say which group differs).

        Returns
        -------
        dict
            "p_value": omnibus multivariate log-rank p-value (null:
                all K survival curves are identical).
            "p_value_label": formatted string for the omnibus p-value.
            "group_n": {label: n, ...} for all K groups.
            "median_survival": {label: <SingleClassSurv median_survival
                dict>, ...} for all K groups.
            "median_follow_up": {label: <SingleClassSurv
                median_follow_up dict>, ...} for all K groups.
            "pairwise_vs_baseline": {other_label: {"p_value": ...,
                "p_value_label": ...}, ...} -- one entry per non-baseline
                group, from a 2-sample log-rank test against
                self.baseline_label.
        """
        self.kms = {}
        for idx, df in self.surv_dfs.items():
            self.kms[idx] = SingleClassSurv(surv_label=self.surv_label, surv_df=df)

        all_durations = self.surv_df_multi["time"]
        all_events = SingleClassSurv._coerce_event_to_bool(self.surv_df_multi["event"])
        all_groups = self.surv_df_multi[self.target_col_name]

        omnibus_result = multivariate_logrank_test(
            event_durations=all_durations,
            groups=all_groups,
            event_observed=all_events,
        )

        log_rank_dict = {
            "p_value": omnibus_result.p_value,
            "p_value_label": get_p_value_label(omnibus_result.p_value),
            "group_n": {
                self.group_labels[idx]: len(df) for idx, df in self.surv_dfs.items()
            },
            "median_survival": {
                self.group_labels[idx]: km.descriptives["median_survival"]
                for idx, km in self.kms.items()
            },
            "median_follow_up": {
                self.group_labels[idx]: km.descriptives["median_follow_up"]
                for idx, km in self.kms.items()
            },
        }

        pairwise = {}
        baseline_df = self.surv_dfs[self.baseline_idx]
        for idx, df in self.surv_dfs.items():
            if idx == self.baseline_idx:
                continue
            pairwise_result = logrank_test(
                baseline_df["time"],
                df["time"],
                event_observed_A=baseline_df["event"],
                event_observed_B=df["event"],
            )
            pairwise[self.group_labels[idx]] = {
                "p_value": pairwise_result.p_value,
                "p_value_label": get_p_value_label(pairwise_result.p_value),
            }
        log_rank_dict["pairwise_vs_baseline"] = pairwise

        return log_rank_dict

    def get_cox_ph_dict(self):
        """Fit ONE Cox model with all K-1 non-baseline groups one-hot
        encoded as covariates (baseline's dummy column dropped), so every
        non-baseline group's hazard ratio is estimated jointly against a
        shared baseline hazard -- see point 2 in the module docstring for
        why this isn't done as K-1 separate two-group Cox fits.

        Returns
        -------
        dict
            If `self.split_valid` is False: {"split_valid": False,
            "split_invalid_reason": <reason>} -- no Cox model is fit.

            Otherwise:
            "baseline_group": the resolved baseline label (== self.baseline_label).
            "hazard_ratios": {other_label: {"raw": {"hr", "ci_lower",
                "ci_upper"}, "label": <formatted str>, "p_value": ...,
                "p_value_label": ..., "interpretation": ...}, ...} --
                one entry per non-baseline group, all from the same
                jointly-fit Cox model.
        """
        tests_dict = getattr(self, "tests_dict", None)
        if tests_dict is not None:
            return tests_dict["cox_ph"]

        if not self.split_valid:
            logger.warning(
                f"[INVALID SPLIT] Group split not valid due to {self.split_invalid_reason}"
            )
            return {
                "split_valid": self.split_valid,
                "split_invalid_reason": self.split_invalid_reason,
            }

        baseline_label = self.baseline_label
        prefix = "group"

        cox_ph_df = self.surv_df_multi.copy()
        cox_ph_df[self.target_col_name] = cox_ph_df[self.target_col_name].astype("category")
        cox_ph_df = pd.get_dummies(
            cox_ph_df, columns=[self.target_col_name], prefix_sep="_", prefix=prefix
        )

        baseline_col_name = f"{prefix}_{baseline_label}"
        if baseline_col_name not in cox_ph_df.columns:
            raise ValueError(
                f"Could not find dummy column for baseline category "
                f"{baseline_col_name!r} after pd.get_dummies. This usually "
                f"means the category value isn't a clean string. Got "
                f"columns: {list(cox_ph_df.columns)}"
            )
        cox_ph_df = cox_ph_df.drop(columns=[baseline_col_name])

        self.cph = CoxPHFitter(alpha=self.alpha)

        if self.censoring == "right":
            self.cph.fit_right_censoring(cox_ph_df, duration_col="time", event_col="event")
        else:
            raise NotImplementedError(f"Censoring type {self.censoring!r} not implemented.")

        summary = self.cph.summary

        hazard_ratios = {}
        for idx in range(1, self.n_groups):
            other_label = self.group_labels[idx]
            row_name = f"{prefix}_{other_label}"
            if row_name not in summary.index:
                raise RuntimeError(
                    f"Expected Cox covariate row {row_name!r} not found in "
                    f"model summary. Found rows: {summary.index.tolist()}"
                )
            row = summary.loc[row_name]
            hr = row["exp(coef)"]
            hr_ci = [row["exp(coef) lower 95%"], row["exp(coef) upper 95%"]]
            p_value = row["p"]

            hazard_ratios[other_label] = {
                "raw": {"hr": hr, "ci_lower": hr_ci[0], "ci_upper": hr_ci[1]},
                "label": f"Hazard ratio, {hr:.2f} (95% CI, {hr_ci[0]:.2f} - {hr_ci[1]:.2f})",
                "p_value": p_value,
                "p_value_label": get_p_value_label(p_value),
                "interpretation": interpret_hazard_ratio(
                    hazard_ratio=hr,
                    baseline_group_name=baseline_label,
                    other_group_name=other_label,
                ),
            }

        cox_ph_dict = {
            "baseline_group": baseline_label,
            "hazard_ratios": hazard_ratios,
        }
        return cox_ph_dict

    def plot_km_curves(
        self,
        cox_ph_dict=None,
        log_rank_dict=None,
        print_hazard_stats=True,
        plot=True,
        title=None,
        save_path=None,
        plot_grid=True,
        x_axis_range=None,
        add_risk_table=True,
    ):
        """Plot all K groups' KM curves on one axis.

        Parameters
        ----------
        cox_ph_dict, log_rank_dict : dict, optional
            Default to `self.get_cox_ph_dict()` / `self.get_log_rank_dict()`.
            Lets a caller display stats from a different fit without
            recomputing or overwriting cached results.
        x_axis_range : iterable, optional
            Defaults to 12-unit ticks from 0 to the latest observed time
            across all K groups -- assumes monthly time units, as in
            SingleClassSurv/BinaryClassSurv.
        """
        if not hasattr(self, "kms"):
            self.get_log_rank_dict()  # populates self.kms

        fig, ax = plt.subplots(figsize=(12, 8))

        cmap = get_cmap("tab10")
        kmfs_for_table = []
        labels_for_table = []
        for idx in range(self.n_groups):
            label = self.group_labels[idx]
            kmf = self.kms[idx].kmf
            kmf.plot_survival_function(ax=ax, label=f"{label}", ci_show=False, color=cmap(idx))
            kmfs_for_table.append(kmf)
            labels_for_table.append(label)

        legend = ax.legend(loc="lower right")
        legend.set_title("Survival Groups")
        plt.xlabel("Time (months)")
        plt.ylabel("Survival Probability")
        
        if title is not None:
            plt.title(title)
        else:
            plt.title(f"{self.surv_label} {self.target_col_name}")
            
        ax.set_ylim(0, 1)

        if x_axis_range is None:
            max_time = max(kmf.event_table.index[-1] for kmf in kmfs_for_table)
            x_axis_range = range(0, int(max_time) + 1, 12)

        x_axis_range = list(x_axis_range)
        plt.xticks(x_axis_range)
        plt.axhline(y=0.5, color="red", linestyle="--")

        if add_risk_table:
            add_at_risk_counts(*kmfs_for_table, ax=ax, labels=labels_for_table, xticks=x_axis_range)
            fig.subplots_adjust(left=0.2, bottom=0.3)

        #Print P-Value
        if log_rank_dict is None:
            log_rank_dict = self.get_log_rank_dict()
        y = 0.20
        plt.text(
            x=0.05, y=y, s=f"Overall {log_rank_dict['p_value_label']}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
        )
            
        if print_hazard_stats:
            if cox_ph_dict is None:
                cox_ph_dict = self.get_cox_ph_dict()

            y -= 0.05
            if "hazard_ratios" in cox_ph_dict:
                for other_label, hr_dict in cox_ph_dict["hazard_ratios"].items():
                    plt.text(
                        x=0.05, y=y,
                        s=f"{other_label} vs {cox_ph_dict['baseline_group']}: "
                          f"{hr_dict['label']}, {hr_dict['p_value_label']}",
                        transform=ax.transAxes, fontsize=8, verticalalignment="top",
                    )
                    y -= 0.04

        if plot_grid:
            plt.grid(True)

        if save_path is not None:
            plt.savefig(save_path)

        if plot:
            plt.show()
        else:
            plt.close()