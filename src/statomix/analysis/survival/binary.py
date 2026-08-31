"""Binary-group log-rank, Kaplan-Meier, and Cox-PH analysis."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test

from statomix.analysis.survival.data import prepare_survival_data
from statomix.analysis.survival.single import SingleClassSurv
from statomix.logging import get_logger

from .formatting import get_p_value_label, interpret_hazard_ratio

logger = get_logger(name="BinaryClassSurv")


class BinaryClassSurv:
    """Compares two survival cohorts defined by a single binary column.

    Runs a log-rank test, fits per-group Kaplan-Meier curves, fits a Cox
    proportional-hazards model to estimate the hazard ratio between the
    two groups, and can plot both KM curves on one axis with an at-risk
    table and hazard-ratio annotation.

    Parameters
    ----------
    surv_df_binary : pandas.DataFrame
        Must contain exactly three columns:
            - "time": non-negative numeric follow-up duration.
            - "event": event indicator (boolean, or 0/1 integers).
            - exactly one additional grouping column with exactly two
              unique values.
    surv_label : str
        Human-readable label, passed through to the underlying
        SingleClassSurv fits (used as legend label / fit name).
    alpha : float, default 0.05
        Significance level used for the Cox model's confidence intervals.
    baseline_group : {"largest", "smallest", "first", "second"} or any
        actual category value found in the grouping column, default
        "largest".
        Controls which group is used as the Cox-model reference level,
        AND which group is anchored as "group0" everywhere else in this
        class (group_labels[0], surv_df0, log_rank_dict["group0_*"],
        etc.) -- see "Group identity" below.
            - "largest"  -> the group with more observations (ties go to
                             whichever category appears first in
                             `.unique()`).
            - "smallest" -> the group with fewer observations.
            - "first"    -> the first-seen category in the data
                             (`.unique()` order).
            - "second"   -> the second-seen category in the data.
            - any other value is matched against the actual category
              labels found in the grouping column, so you can pin the
              baseline explicitly, e.g. baseline_group="control".

    Group identity
    ----------
    "group0" and "group1" are NOT determined by row order in
    `surv_df_binary`. `baseline_group` is resolved first, against the
    two category labels found in the data, and group0 is then defined
    to *be* that resolved baseline label; group1 is whichever category
    is left over. This means group0/group1 identity is a deterministic
    function of the category labels (and counts, for the "largest"/
    "smallest" keywords) -- never of which row happens to appear first
    in the input DataFrame. (Earlier versions of this class fixed
    group0/group1 from first-seen `.unique()` order *before* resolving
    baseline_group, which made group0/group1 identity depend on row
    order whenever the grouping column's values were themselves a
    function of an external parameter, e.g. a scanned threshold -- the
    first-seen category could flip from one call to the next even
    though baseline_group was passed identically each time. That
    dependency has been removed: group0 is always the resolved
    baseline now.)

    Attributes
    ----------
    target_col_name : str
        Name of the grouping column.
    group_labels : dict
        Maps {0: category0, 1: category1}. category0 is always the
        resolved baseline label (see "Group identity" above); category1
        is the other category.
    baseline_label : Any
        The actual category value (one of `group_labels.values()`) that
        was resolved as the Cox-model reference level. Always equal to
        `group_labels[0]`. This is the single source of truth for "which
        group is baseline" -- callers should read this attribute rather
        than re-deriving baseline group identity from the
        `baseline_group` argument they originally passed in, since
        keyword forms like "largest"/"smallest" only resolve to an
        actual label here.
    baseline_idx : int
        Always 0, kept as an attribute (rather than hardcoding `0`
        everywhere it's read) so downstream code that reads
        `self.baseline_idx` doesn't need to change.
    surv_df0, surv_df1 : pandas.DataFrame
        "time"/"event" subsets for group 0 (baseline) and group 1
        (other) respectively.
    km0, km1 : SingleClassSurv
        Fitted KM models for each group.
    log_rank_dict : dict
        Family-specific validity, log-rank test statistic and p-value, group
        sizes, and per-group median survival/follow-up. A Cox-fit failure does
        not remove this result.
    cph : lifelines.CoxPHFitter
        Fitted Cox model with the grouping variable as the single
        covariate.
    cox_ph_dict : dict
        Hazard ratio, its 95% CI, p-value, formatted label, and a
        plain-English interpretation. `cox_ph_dict["baseline_group"]` is
        always equal to `self.baseline_label` (see above) -- it reflects
        whichever group was actually resolved as baseline, regardless of
        whether `baseline_group` was passed as a keyword or an explicit
        label.

    Notes
    -----
    Cox and log-rank eligibility are tracked separately. Cox failures are
    returned as structured, non-finite results so a valid log-rank comparison
    remains available to threshold-scan callers.

    Raises
    ------
    ValueError
        If required columns are missing, there isn't exactly one grouping
        column, that column doesn't have exactly two unique values, or
        `baseline_group` doesn't resolve to a known option or category.
    """

    MODULE_NAME = "Survival - Binary"
    _BASELINE_KEYWORDS = ("largest", "smallest", "first", "second")

    def __init__(
        self,
        surv_df_binary: pd.DataFrame,
        surv_label: str,
        alpha: float = 0.05,
        baseline_group: str = "largest",
        censoring: str = "right",
        verbose: bool = True,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        if censoring != "right":
            raise NotImplementedError(
                "Only right-censored survival data are currently supported."
            )
        required_cols = {"time", "event"}
        missing = required_cols - set(surv_df_binary.columns)
        if missing:
            raise ValueError(
                f"surv_df_binary is missing required column(s): {sorted(missing)}"
            )
        self.verbose = verbose
        self.alpha = alpha
        # self.cox_ph_dict = {"not_created": "method self.create_cox_ph_dict() not called."}
        # self.log_rank_dict = {"not_created": "method self.create_log_rank_dcit() not called."}
        self.censoring = censoring
        self.surv_label = surv_label

        grouping_cols = [
            c for c in surv_df_binary.columns if c not in ("time", "event")
        ]
        if len(grouping_cols) != 1:
            error_msg = (
                "Expected exactly one grouping column besides 'time'/'event', "
                + f"\nfound {len(grouping_cols)}: {grouping_cols}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        target_col_name = grouping_cols[0]
        prepared = prepare_survival_data(
            frame=surv_df_binary,
            grouping_column=target_col_name,
        )
        if prepared.dropped_rows:
            logger.warning(
                "[BinaryClassSurv] Dropped %s row(s) missing time, event, " "or group.",
                prepared.dropped_rows,
            )
        analysis_df = prepared.frame

        # first-seen order; deterministic given the input, but NOTE this
        # is only used to resolve the "first"/"second" baseline_group
        # keywords below -- it is NOT used to decide which category
        # becomes group0. See _resolve_baseline_group / "Group identity"
        # in the class docstring.
        categories = analysis_df[target_col_name].unique()
        if categories.shape[0] != 2:
            raise ValueError(
                f"Expected exactly 2 groups in '{target_col_name}', got "
                f"{categories.shape[0]}: {sorted(categories, key=str)}"
            )

        self.target_col_name = target_col_name
        self.surv_df_binary = analysis_df

        # Single source of truth for "which group is baseline" -- resolved
        # once, here, from whatever form `baseline_group` was passed in
        # (keyword or explicit label), and BEFORE group0/group1 are split
        # out. group0 is then defined to be whatever this resolves to, so
        # group0/group1 identity is always a function of baseline_group
        # and the category labels/counts -- never of row order in
        # surv_df_binary. Every downstream consumer (cox_ph_dict,
        # plotting, callers like MinimumPValue) should read
        # `self.baseline_label` / `self.group_labels[0]` rather than
        # re-deriving it, so there is never a second, possibly
        # inconsistent, notion of "baseline" or "group0" floating around.
        baseline_label, other_label = self._resolve_baseline_group(
            baseline_group=baseline_group,
            categories=categories,
            surv_df_binary=analysis_df,
        )

        self.group_labels = {0: baseline_label, 1: other_label}
        self.baseline_idx = 0
        self.baseline_label = baseline_label

        mask = analysis_df[target_col_name] == baseline_label
        self.surv_df0 = analysis_df[mask][["time", "event"]].copy()
        self.surv_df1 = analysis_df[~mask][["time", "event"]].copy()

        self._checks_group_split_validity()

        # self._create_log_rank_dict()
        # self._create_cox_ph_dict()

    @staticmethod
    def get_config_df():
        return pd.DataFrame(columns=["Categorical", "Survival Labels"])

    @staticmethod
    def add_validation_to_analysis_config_file(path, max_row=500):
        from statomix.reporting.excel.validation import (
            add_datatype_list_validations,
        )

        add_datatype_list_validations(
            path=path,
            sheet_name=BinaryClassSurv.MODULE_NAME,
            max_row=max_row,
        )

    def _checks_group_split_validity(self):
        group0_n = len(self.surv_df0)
        group1_n = len(self.surv_df1)

        group0_events = self.surv_df0["event"].sum()
        group1_events = self.surv_df1["event"].sum()

        group0_censored = group0_n - group0_events
        group1_censored = group1_n - group1_events

        self.log_rank_valid = group0_n > 0 and group1_n > 0
        self.log_rank_invalid_reason = None
        if not self.log_rank_valid:
            self.log_rank_invalid_reason = "empty_group"
        elif group0_events + group1_events == 0:
            self.log_rank_valid = False
            self.log_rank_invalid_reason = "no_events_in_sample"

        self.split_valid = True
        self.split_invalid_reason = None

        if group0_n <= 1 or group1_n <= 1:
            self.split_valid = False
            self.split_invalid_reason = "group_size_1"
            if self.verbose:
                logger.warning(
                    f"[INVALID SPLIT] {self.split_invalid_reason}: "
                    f"group0_n={group0_n}, group1_n={group1_n}"
                )

        elif group0_events == 0 or group1_events == 0:
            self.split_valid = False
            self.split_invalid_reason = "no_events"
            if self.verbose:
                logger.warning(
                    f"[INVALID SPLIT] {self.split_invalid_reason}: "
                    f"group0_events={group0_events}, group1_events={group1_events}"
                )

        elif (group0_censored == 0 or group1_censored == 0) and self.verbose:
            logger.warning(
                "[COX SPLIT NOTICE] A group has no censored observations: "
                "group0_censored=%s, group1_censored=%s. The fit will be "
                "attempted and its numerical result validated.",
                group0_censored,
                group1_censored,
            )

        # ``split_valid`` is retained as the established Cox-fit eligibility
        # contract.  A log-rank comparison can remain defined when a group has
        # no events or no censoring, so its eligibility is tracked separately.
        self.cox_ph_valid = self.split_valid
        self.cox_ph_invalid_reason = self.split_invalid_reason

        # elif group0_n < 10 or group1_n < 10:
        #     self.split_valid = False
        #     self.split_invalid_reason = "small_group"
        #     print(
        #         f"[INVALID SPLIT] {self.split_invalid_reason}: "
        #         f"group0_n={group0_n}, group1_n={group1_n}"
        #     )

        # elif group0_events < 5 or group1_events < 5:
        #     self.split_valid = False
        #     self.split_invalid_reason = "too_few_events"
        #     print(
        #         f"[INVALID SPLIT] {self.split_invalid_reason}: "
        #         f"group0_events={group0_events}, group1_events={group1_events}"
        #     )

    def _resolve_baseline_group(
        self, baseline_group, categories, surv_df_binary
    ) -> tuple:
        """Resolve the `baseline_group` argument to (baseline_label, other_label).

        Accepts the keywords "largest"/"smallest"/"first"/"second", or an
        actual category value present in the grouping column. Raises
        ValueError if it matches none of those.

        Resolved against `categories` (first-seen `.unique()` order, used
        only to break ties / define "first"/"second") and group counts in
        `surv_df_binary` -- NOT against any group0/group1 split, since
        that split doesn't exist yet when this runs. This ordering is
        what guarantees group0/group1 identity never depends on row
        order in the input (see "Group identity" in the class
        docstring).
        """
        cat_first, cat_second = categories[0], categories[1]
        n_first = int((surv_df_binary[self.target_col_name] == cat_first).sum())
        n_second = len(surv_df_binary) - n_first

        if baseline_group == "largest":
            baseline_label = cat_first if n_first >= n_second else cat_second
        elif baseline_group == "smallest":
            baseline_label = cat_first if n_first < n_second else cat_second
        elif baseline_group == "first":
            baseline_label = cat_first
        elif baseline_group == "second":
            baseline_label = cat_second
        elif baseline_group == cat_first:
            baseline_label = cat_first
        elif baseline_group == cat_second:
            baseline_label = cat_second
        else:
            error_msg = (
                f"baseline_group={baseline_group!r} is not a recognized keyword "
                + f"\n({self._BASELINE_KEYWORDS}) and does not match either category"
                + f"\nfound in '{self.target_col_name}': "
                + f"\n{[cat_first, cat_second]}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        other_label = cat_second if baseline_label == cat_first else cat_first
        return baseline_label, other_label

    def get_tests_dict(self):
        tests_dict = getattr(self, "tests_dict", None)
        if tests_dict is not None:
            return tests_dict

        split_ratio = self.surv_df0.shape[0] / self.surv_df1.shape[0]
        tests_dict = {"split_ratio": split_ratio}
        try:
            tests_dict["log_rank"] = self.get_log_rank_dict()
        except Exception as exc:
            logger.warning(
                "Log-rank analysis failed with %s: %s",
                type(exc).__name__,
                exc,
            )
            tests_dict["log_rank"] = {
                "valid": False,
                "invalid_reason": "analysis_error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "p_value": float("nan"),
                "test_statistic": float("nan"),
                "group0_n": self.surv_df0.shape[0],
                "group1_n": self.surv_df1.shape[0],
            }

        try:
            tests_dict["cox_ph"] = self.get_cox_ph_dict()
        except Exception as exc:
            logger.warning(
                "Cox-PH analysis failed with %s: %s",
                type(exc).__name__,
                exc,
            )
            tests_dict["cox_ph"] = {
                "split_valid": False,
                "split_invalid_reason": "analysis_error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "p_value": float("nan"),
            }

        self.tests_dict = tests_dict
        return tests_dict

    def get_log_rank_dict(self):
        self.km0 = SingleClassSurv(surv_label=self.surv_label, surv_df=self.surv_df0)
        self.km1 = SingleClassSurv(surv_label=self.surv_label, surv_df=self.surv_df1)

        if self.log_rank_valid:
            log_rank_results = logrank_test(
                self.surv_df0["time"],
                self.surv_df1["time"],
                event_observed_A=self.surv_df0["event"],
                event_observed_B=self.surv_df1["event"],
            )
            p_value = float(log_rank_results.p_value)
            test_statistic = float(log_rank_results.test_statistic)
            if not np.isfinite(p_value) or not np.isfinite(test_statistic):
                self.log_rank_valid = False
                self.log_rank_invalid_reason = "non_finite_test_result"
                p_value = float("nan")
                test_statistic = float("nan")
        else:
            p_value = float("nan")
            test_statistic = float("nan")

        log_rank_dict = {
            "valid": self.log_rank_valid,
            "invalid_reason": self.log_rank_invalid_reason,
            "p_value": p_value,
            "p_value_label": (
                get_p_value_label(p_value) if self.log_rank_valid else "Not estimable"
            ),
            "test_statistic": test_statistic,
            "group0_n": self.surv_df0.shape[0],
            "group1_n": self.surv_df1.shape[0],
            "group0_median_survival": self.km0.descriptives["median_survival"],
            "group1_median_survival": self.km1.descriptives["median_survival"],
            # "group0_median_survival_raw": self.km0.descriptives['median_survival']['raw'],
            # "group1_median_survival_raw": self.km1.descriptives['median_survival']['raw'],
            "group0_median_follow_up": self.km0.descriptives["median_follow_up"],
            "group1_median_follow_up": self.km1.descriptives["median_follow_up"],
            # "group0_median_follow_up_raw": self.km0.descriptives['median_follow_up']['raw'],
            # "group1_median_follow_up_raw": self.km1.descriptives['median_follow_up']['raw'],
        }

        # self.log_rank_dict = log_rank_dict
        return log_rank_dict

    def get_cox_ph_dict(self):
        # Read from the single resolved source of truth set in __init__,
        # rather than re-resolving baseline_group here.

        tests_dict = getattr(self, "tests_dict", None)
        if tests_dict is not None:
            return tests_dict["cox_ph"]

        if not self.split_valid:
            if self.verbose:
                logger.warning(
                    "[INVALID SPLIT] Group split not valid due to %s",
                    self.split_invalid_reason,
                )
            cox_ph_dict = {
                "split_valid": self.split_valid,
                "split_invalid_reason": self.split_invalid_reason,
            }
            return cox_ph_dict

        baseline_idx = self.baseline_idx
        baseline_label = self.baseline_label
        other_label = self.group_labels[1 - baseline_idx]

        prefix = "group"

        cox_ph_df = self.surv_df_binary.copy()
        cox_ph_df[self.target_col_name] = cox_ph_df[self.target_col_name].astype(
            "category"
        )
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
            self.cph.fit_right_censoring(
                cox_ph_df,
                duration_col="time",
                event_col="event",
            )
        else:
            raise NotImplementedError(
                f"Censoring type {self.censoring!r} not implemented."
            )

        summary = self.cph.summary
        row_names = summary.index.tolist()
        if len(row_names) != 1:
            raise RuntimeError(
                "Expected exactly one covariate row in the Cox model "
                f"summary, found {len(row_names)}: {row_names}"
            )
        row = summary.loc[row_names[0]]
        hr = row["exp(coef)"]
        hr_ci = [row["exp(coef) lower 95%"], row["exp(coef) upper 95%"]]
        p_value = row["p"]
        if not np.isfinite([hr, *hr_ci, p_value]).all():
            self.cox_ph_valid = False
            self.cox_ph_invalid_reason = "non_finite_fit_result"
            self.split_valid = False
            self.split_invalid_reason = self.cox_ph_invalid_reason
            return {
                "split_valid": False,
                "split_invalid_reason": "non_finite_fit_result",
                "p_value": float("nan"),
            }

        cox_ph_dict = {
            # Guaranteed identical to self.baseline_label -- this is the
            # actual resolved category, never the raw "largest"/"smallest"
            # keyword the caller may have passed in.
            "split_valid": self.split_valid,
            "baseline_group": baseline_label,
            "hr": {"raw": {"hr": hr, "ci_lower": hr_ci[0], "ci_upper": hr_ci[1]}},
            # "hr": hr,
            # "hr_ci": hr_ci,
            "p_value": p_value,
        }
        cox_ph_dict["hr"]["label"] = (
            f"Hazard ratio, {hr:.2f} " f"(95% CI, {hr_ci[0]:.2f} - {hr_ci[1]:.2f})"
        )
        cox_ph_dict["p_value_label"] = get_p_value_label(cox_ph_dict["p_value"])
        cox_ph_dict["interpretation"] = interpret_hazard_ratio(
            hazard_ratio=cox_ph_dict["hr"]["raw"]["hr"],
            baseline_group_name=baseline_label,
            other_group_name=other_label,
        )

        return cox_ph_dict
        # self.cox_ph_dict = cox_ph_dict

    def plot_km_curves(
        self,
        cox_ph_dict=None,
        print_hazard_stats=True,
        plot=True,
        title=None,
        save_path=None,
        plot_grid=False,
        x_axis_range=None,
        add_risk_table=True,
    ):
        """Plot both groups' KM curves on one axis.

        Parameters
        ----------
        cox_ph_dict : dict, optional
            Defaults to `self.cox_ph_dict`. Lets a caller display a
            different HR (e.g. from a re-fit Cox model) without
            overwriting `self.cox_ph_dict`.
        x_axis_range : iterable, optional
            Defaults to 12-unit ticks from 0 to the later of the two
            groups' max observed times -- assumes monthly time units.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        self.km0.kmf.plot_survival_function(
            ax=ax, label=f"{self.group_labels[0]}", ci_show=False
        )
        self.km1.kmf.plot_survival_function(
            ax=ax, label=f"{self.group_labels[1]}", ci_show=False
        )

        legend = ax.legend(loc="lower right")
        legend.set_title("Survival Groups")
        plt.xlabel("Time (months)")
        plt.ylabel("Survival Probability")
        if title is not None:
            plt.title(title)
        ax.set_ylim(0, 1)

        if x_axis_range is None:
            max_time = max(
                self.km0.kmf.event_table.index[-1], self.km1.kmf.event_table.index[-1]
            )
            x_axis_range = range(0, int(max_time) + 1, 12)

        plt.xticks(x_axis_range)
        plt.axhline(y=0.5, color="red", linestyle="--")

        if add_risk_table:
            add_at_risk_counts(
                self.km0.kmf,
                self.km1.kmf,
                ax=ax,
                labels=[self.group_labels[0], self.group_labels[1]],
            )
            fig.subplots_adjust(left=0.2, bottom=0.3)

        if print_hazard_stats:
            if cox_ph_dict is None:
                # print(
                #     f"Print hazard stats is {print_hazard_stats} but Cox Ph dict is not passed calculating."
                # )
                cox_ph_dict = self.get_cox_ph_dict()

            if cox_ph_dict["split_valid"]:
                plt.text(
                    x=0.05,
                    y=0.15,
                    s=cox_ph_dict["hr"]["label"],
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                )

                plt.text(
                    x=0.05,
                    y=0.10,
                    s=cox_ph_dict["p_value_label"],
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                )
            else:
                plt.text(
                    x=0.05,
                    y=0.15,
                    s="Invalid Split: " + cox_ph_dict["split_invalid_reason"],
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                )

                log_rank_dict = self.get_log_rank_dict()
                plt.text(
                    x=0.05,
                    y=0.10,
                    s=log_rank_dict["p_value_label"],
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                )

        if plot_grid:
            plt.grid(True)

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")

        if plot:
            plt.show()
        else:
            plt.close()
