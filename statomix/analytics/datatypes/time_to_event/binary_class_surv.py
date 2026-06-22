import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts

from .single_class_surv import SingleClassSurv
from .formatting import get_p_value_label, interpret_hazard_ratio


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
        Controls which group is used as the Cox-model reference level:
            - "largest"  -> the group with more observations (ties go to
                             the first-seen category).
            - "smallest" -> the group with fewer observations.
            - "first"    -> the first-seen category in the data.
            - "second"   -> the second-seen category in the data.
            - any other value is matched against the actual category
              labels found in the grouping column, so you can pin the
              baseline explicitly, e.g. baseline_group="control".

    Attributes
    ----------
    target_col_name : str
        Name of the grouping column.
    group_labels : dict
        Maps {0: category0, 1: category1} -- the two unique values found
        in the grouping column, in first-seen order from `.unique()`.
    surv_df0, surv_df1 : pandas.DataFrame
        "time"/"event" subsets for group 0 and group 1 respectively.
    km0, km1 : SingleClassSurv
        Fitted KM models for each group.
    log_rank_dict : dict
        Log-rank test p-value, group sizes, and per-group median
        survival/follow-up.
    cph : lifelines.CoxPHFitter
        Fitted Cox model with the grouping variable as the single
        covariate.
    hazard_dict : dict
        Hazard ratio, its 95% CI, p-value, formatted label, and a
        plain-English interpretation.

    Raises
    ------
    ValueError
        If required columns are missing, there isn't exactly one grouping
        column, that column doesn't have exactly two unique values, or
        `baseline_group` doesn't resolve to a known option or category.
    """

    _BASELINE_KEYWORDS = ("largest", "smallest", "first", "second")

    def __init__(
        self,
        surv_df_binary: pd.DataFrame,
        surv_label: str,
        alpha: float = 0.05,
        baseline_group: str = "largest",
        censoring: str = "right",
    ):
        required_cols = {"time", "event"}
        missing = required_cols - set(surv_df_binary.columns)
        if missing:
            raise ValueError(
                f"surv_df_binary is missing required column(s): {sorted(missing)}"
            )

        self.alpha = alpha
        self.surv_label = surv_label
        self.censoring = censoring

        grouping_cols = [c for c in surv_df_binary.columns if c not in ("time", "event")]
        if len(grouping_cols) != 1:
            raise ValueError(
                "Expected exactly one grouping column besides 'time'/'event', "
                f"found {len(grouping_cols)}: {grouping_cols}"
            )
        target_col_name = grouping_cols[0]

        # first-seen order; deterministic given the input, but NOTE this
        # depends on row order in surv_df_binary, not on sorting.
        categories = surv_df_binary[target_col_name].unique()
        if categories.shape[0] != 2:
            raise ValueError(
                f"Expected exactly 2 groups in '{target_col_name}', got "
                f"{categories.shape[0]}: {sorted(categories)}"
            )

        category0, category1 = categories[0], categories[1]
        mask = surv_df_binary[target_col_name] == category0

        self.target_col_name = target_col_name
        self.surv_df_binary = surv_df_binary
        self.group_labels = {0: category0, 1: category1}
        self.surv_df0 = surv_df_binary[mask][["time", "event"]].copy()
        self.surv_df1 = surv_df_binary[~mask][["time", "event"]].copy()

        self._baseline_idx = self._resolve_baseline_group(baseline_group)

        self._create_log_rank_dict()
        self._create_hazard_dict(censoring=censoring)

    def _resolve_baseline_group(self, baseline_group) -> int:
        """Resolve the `baseline_group` argument to 0 or 1 (an index into
        `self.group_labels`).

        Accepts the keywords "largest"/"smallest"/"first"/"second", or an
        actual category value present in the grouping column. Raises
        ValueError if it matches none of those.
        """
        n0, n1 = self.surv_df0.shape[0], self.surv_df1.shape[0]

        if baseline_group == "largest":
            return 0 if n0 >= n1 else 1
        if baseline_group == "smallest":
            return 0 if n0 < n1 else 1
        if baseline_group == "first":
            return 0
        if baseline_group == "second":
            return 1

        for idx, label in self.group_labels.items():
            if label == baseline_group:
                return idx

        raise ValueError(
            f"baseline_group={baseline_group!r} is not a recognized keyword "
            f"({self._BASELINE_KEYWORDS}) and does not match either category "
            f"found in '{self.target_col_name}': "
            f"{[self.group_labels[0], self.group_labels[1]]}"
        )

    def _create_log_rank_dict(self):
        self.km0 = SingleClassSurv(surv_label=self.surv_label, surv_df=self.surv_df0)
        self.km1 = SingleClassSurv(surv_label=self.surv_label, surv_df=self.surv_df1)

        log_rank_results = logrank_test(
            self.surv_df0["time"],
            self.surv_df1["time"],
            event_observed_A=self.surv_df0["event"],
            event_observed_B=self.surv_df1["event"],
        )

        log_rank_dict = {
            "p_value": log_rank_results.p_value,
            f"{self.group_labels[0]}_n": self.surv_df0.shape[0],
            f"{self.group_labels[1]}_n": self.surv_df1.shape[0],
            f"{self.group_labels[0]}_median_survival": self.km0.descriptives["median_survival"],
            f"{self.group_labels[1]}_median_survival": self.km1.descriptives["median_survival"],
            f"{self.group_labels[0]}_median_follow_up": self.km0.descriptives["median_follow_up"],
            f"{self.group_labels[1]}_median_follow_up": self.km1.descriptives["median_follow_up"],
        }

        self.log_rank_dict = log_rank_dict

    def _create_hazard_dict(self, censoring: str = "right"):
        baseline_group = self._baseline_idx
        baseline_label = self.group_labels[baseline_group]
        other_label = self.group_labels[1 - baseline_group]

        prefix = "group"
        
        cox_ph_df = self.surv_df_binary.copy()
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

        if censoring == "right":
            self.cph.fit_right_censoring(
                cox_ph_df,
                duration_col="time",
                event_col="event",
            )
        else:
            raise NotImplementedError(f"Censoring type {censoring!r} not implemented.")

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

        hazard_dict = {
            "baseline_group": baseline_label,
            "hr": hr,
            "hr_ci": hr_ci,
            "p_value": row["p"],
        }
        hazard_dict["label"] = (
            f"Hazard ratio, {hazard_dict['hr']:.2f} "
            f"(95% CI, {hazard_dict['hr_ci'][0]:.2f} - {hazard_dict['hr_ci'][1]:.2f})"
        )
        hazard_dict["p_value_label"] = get_p_value_label(hazard_dict["p_value"])
        hazard_dict["interpretation"] = interpret_hazard_ratio(
            hazard_ratio=hazard_dict["hr"],
            baseline_group_name=baseline_label,
            other_group_name=other_label,
        )

        self.hazard_dict = hazard_dict

    def plot_km_curves(
        self,
        hazard_dict=None,
        print_hazard_stats=True,
        plot=True,
        title=None,
        savepath=None,
        plot_grid=True,
        x_axis_range=None,
        add_risk_table=True,
    ):
        """Plot both groups' KM curves on one axis.

        Parameters
        ----------
        hazard_dict : dict, optional
            Defaults to `self.hazard_dict`. Lets a caller display a
            different HR (e.g. from a re-fit Cox model) without
            overwriting `self.hazard_dict`.
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
            #plt.subplots_adjust(left=0.2, bottom=0.3)

        if print_hazard_stats:
            if hazard_dict is None:
                hazard_dict = self.hazard_dict

            plt.text(
                x=0.05,
                y=0.15,
                s=hazard_dict["label"],
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
            )
            plt.text(
                x=0.05,
                y=0.10,
                s=hazard_dict["p_value_label"],
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
            )

        if plot_grid:
            plt.grid(True)

        if savepath is not None:
            plt.savefig(savepath)

        if plot:
            plt.show()
        else:
            plt.close()