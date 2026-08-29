"""Exploratory survival threshold scan with multiplicity control."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fileverse.clean_path_name import clean_path_name
from fileverse.formats.zarr import BaseZARR
from tqdm.auto import tqdm

from statomix.analysis.multiplicity import holm_adjust_with_missing
from statomix.analysis.survival.binary import BinaryClassSurv
from statomix.analysis.survival.data import prepare_survival_data
from statomix.logging import get_logger

logger = get_logger(name="MinimumPValue")


class MinimumPValue:
    """Scan numerical cutoffs while retaining invalid and failed splits.

    Raw p-values remain available for exploratory inspection. By default,
    Holm-adjusted p-values are added across all successfully tested cutoffs,
    and significance-dependent markers use those adjusted values.
    """

    MODULE_NAME = "Survival -Threshold MPV"

    def __init__(
        self,
        surv_label: str,
        surv_df_mpv: pd.DataFrame,
        root_group,
        trunc_pct: float | None = None,
        iqr_multiplier: float | None = 1.5,
        use_synthetic_cutoffs: bool = False,
        search_resolution: float = 0.5,
        show_progress: bool = True,
        alpha: float = 0.05,
        multiplicity_method: str = "holm",
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        if use_synthetic_cutoffs and search_resolution <= 0:
            raise ValueError(
                f"search_resolution must be positive, got {search_resolution!r}"
            )
        if trunc_pct is not None and not 0 <= trunc_pct < 50:
            raise ValueError(f"trunc_pct must be in [0, 50), got {trunc_pct!r}")
        if iqr_multiplier is not None and iqr_multiplier < 0:
            raise ValueError(
                "iqr_multiplier must be non-negative or None, got "
                f"{iqr_multiplier!r}"
            )
        if multiplicity_method not in {"holm", "none"}:
            raise ValueError(
                "multiplicity_method must be 'holm' or 'none', got "
                f"{multiplicity_method!r}"
            )

        self.alpha = alpha
        self.trunc_pct = trunc_pct
        self.surv_label = surv_label
        self.show_progress = show_progress
        self.iqr_multiplier = iqr_multiplier
        self.search_resolution = search_resolution
        self.use_synthetic_cutoffs = use_synthetic_cutoffs
        self.multiplicity_method = multiplicity_method
        self.surv_df_mpv = surv_df_mpv

        required_cols = {"time", "event"}
        missing = required_cols - set(self.surv_df_mpv.columns)
        if missing:
            raise ValueError(
                f"surv_df_mpv is missing required column(s): {sorted(missing)}"
            )

        grouping_cols = [
            c for c in self.surv_df_mpv.columns if c not in ("time", "event")
        ]
        if len(grouping_cols) != 1:
            raise ValueError(
                "Expected exactly one grouping column besides 'time'/'event', "
                f"found {len(grouping_cols)}: {grouping_cols}"
            )

        target_name = grouping_cols[0]
        prepared = prepare_survival_data(
            frame=surv_df_mpv,
            grouping_column=target_name,
        )
        if prepared.dropped_rows:
            logger.warning(
                "[MinimumPValue] Dropped %s row(s) missing time, event, "
                "or threshold variable.",
                prepared.dropped_rows,
            )
        self.surv_df_mpv = prepared.frame

        self.target_col_stats = {
            "name": target_name,
            "median": float(self.surv_df_mpv[target_name].median()),
        }

        target = self.surv_df_mpv[target_name]
        if not pd.api.types.is_numeric_dtype(target):
            raise ValueError(
                f"'{target_name}' must be numeric to search over "
                "thresholds with <=/> comparisons."
            )
        if not np.isfinite(target.to_numpy(dtype=float)).all():
            raise ValueError(f"'{target_name}' must contain finite values.")
        if target.nunique() < 2:
            raise ValueError(
                f"'{target_name}' must contain at least two distinct values."
            )

        self._create_groups(root_group=root_group)
        self._create_paths()

    @staticmethod
    def get_config_df():
        return pd.DataFrame(columns=["Numerical", "Survival Labels"])

    @staticmethod
    def add_validation_to_analysis_config_file(path, max_row=500):
        from statomix.reporting.excel.validation import (
            add_datatype_list_validations,
        )

        add_datatype_list_validations(
            path=path,
            sheet_name=MinimumPValue.MODULE_NAME,
            max_row=max_row,
        )

    def _create_groups(self, root_group):
        self.groups = {}
        self.groups["root"] = root_group
        clean_col_name = clean_path_name(path=self.target_col_stats["name"])
        self.groups["col_general"] = self.groups["root"].root_group.require_group(
            f"{str(clean_col_name)}_trunc_pct_{self.trunc_pct}_iqr_multiplier_{self.iqr_multiplier}"
        )
        self.groups["col"] = self.groups["col_general"].require_group(self.surv_label)

        col_group = self.groups["col"]
        col_meta = dict(col_group.attrs.get("meta", {}))
        col_meta.setdefault("mpv_data_exists", False)
        col_meta["multiplicity_method"] = self.multiplicity_method
        col_group.attrs["meta"] = col_meta

    def _create_paths(self):
        base_path = BaseZARR.get_abs_path(group=self.groups["col"])

        self.paths = {}
        self.paths["base"] = base_path

        self.paths["mpv_df"] = base_path / "mpv_df.parquet"
        self.paths["marked_thresholds_df"] = base_path / "marked_thresholds_df.parquet"

        self.paths["plot_dashboard"] = base_path / "plot_dashboard.png"
        self.paths["plot_median_follow_up"] = base_path / "plot_median_follow_up.png"
        self.paths["plot_hr_vs_p_value_scatter"] = (
            base_path / "plot_hr_vs_p_value_scatter.png"
        )

    def _get_thresholds(self) -> np.ndarray:
        target_col = self.surv_df_mpv[self.target_col_stats["name"]]
        if self.use_synthetic_cutoffs:
            # Use the true float bounds of the data (not int()-truncated)
            # so the grid always covers the full observed range, and add
            # one extra step so `target_col.max()` itself is included
            # even when it doesn't fall exactly on the grid.
            lo, hi = float(target_col.min()), float(target_col.max())
            thresholds = np.arange(
                lo, hi + self.search_resolution, self.search_resolution
            )
            thresholds = thresholds[thresholds <= hi + 1e-9]
        else:
            # Drop the maximum observed value: thresholding at the max
            # always puts every row in the "<=" group (a guaranteed
            # degenerate split), so it's never worth testing.
            sorted_vals = np.sort(target_col.unique())
            thresholds = sorted_vals[:-1] if len(sorted_vals) > 1 else sorted_vals

        thresholds = np.unique(thresholds)
        n_initial = len(thresholds)

        # 1. IQR outlier fence — drop candidates outside [Q1 - k*IQR, Q3 + k*IQR].
        iqr_multiplier = getattr(self, "iqr_multiplier", None)
        n_before_iqr = len(thresholds)
        if iqr_multiplier and len(thresholds) > 0:
            q1, q3 = np.percentile(thresholds, [25, 75])
            iqr = q3 - q1
            lower_fence = q1 - iqr_multiplier * iqr
            upper_fence = q3 + iqr_multiplier * iqr
            mask = (thresholds >= lower_fence) & (thresholds <= upper_fence)
            # Guard against an empty result (e.g. IQR == 0 collapsing the fence).
            if mask.any():
                thresholds = thresholds[mask]

        n_removed_iqr = n_before_iqr - len(thresholds)

        # 2. Percent-trim — strip the first/last `truncate_pct`% of what
        #    remains (sorted ascending), e.g. to further avoid extreme
        #    thresholds that create tiny, unstable groups.
        trunc_pct = getattr(self, "trunc_pct", 0)
        n_before_trim = len(thresholds)
        if trunc_pct:
            n = len(thresholds)
            n_trim = int(np.floor(n * (trunc_pct / 100.0)))
            if n_trim > 0:
                # Guard against trimming away everything when n_trim would
                # otherwise consume the whole array.
                n_trim = min(n_trim, (n - 1) // 2)
                if n_trim > 0:
                    thresholds = thresholds[n_trim : n - n_trim]
        n_removed_trim = n_before_trim - len(thresholds)

        logger.info(
            f"[Thresholds] Initial Candidates: {n_initial} | "
            f"Removed by IQR (k={iqr_multiplier}): {n_removed_iqr} | "
            f"Removed by percent-trim ({trunc_pct}%): {n_removed_trim} | "
            f"Final: {len(thresholds)}"
        )

        return thresholds

    def _get_mpv_data_at_threshold(self, threshold: float) -> dict:
        low_label = f"<={threshold:.2f}"
        high_label = f">{threshold:.2f}"

        mpv_dict = {
            "threshold": threshold,
            "group0_label": low_label,
            "group1_label": high_label,
            "valid_split": False,
            "invalid_reason": None,
            "error_type": None,
            "error_message": None,
        }

        surv_df_binary = self.surv_df_mpv.copy()
        is_low = surv_df_binary[self.target_col_stats["name"]] <= threshold
        surv_df_binary[self.target_col_stats["name"]] = np.where(
            is_low, low_label, high_label
        )

        if surv_df_binary[self.target_col_stats["name"]].nunique() != 2:
            mpv_dict["invalid_reason"] = "degenerate_split"
            return {
                "mpv_dict": mpv_dict,
                "binary_class_surv_object": None,
            }

        bcs = BinaryClassSurv(
            surv_df_binary=surv_df_binary,
            surv_label=self.surv_label,
            alpha=self.alpha,
            baseline_group=low_label,
            verbose=False,
        )

        tests_dict = bcs.get_tests_dict()
        mpv_dict["valid_split"] = bool(bcs.split_valid)
        mpv_dict["invalid_reason"] = bcs.split_invalid_reason
        mpv_dict["split_ratio"] = tests_dict["split_ratio"]
        mpv_dict["cox_ph"] = tests_dict["cox_ph"]
        mpv_dict["log_rank"] = tests_dict["log_rank"]

        return {"mpv_dict": mpv_dict, "binary_class_surv_object": bcs}

    def _evaluate_threshold(self, *, threshold: float) -> dict:
        """Evaluate one threshold and persist failures as auditable rows."""

        try:
            return self._get_mpv_data_at_threshold(threshold=threshold)
        except Exception as exc:
            logger.warning(
                "Threshold %s failed with %s: %s",
                threshold,
                type(exc).__name__,
                exc,
            )
            return {
                "mpv_dict": {
                    "threshold": float(threshold),
                    "group0_label": f"<={threshold:.2f}",
                    "group1_label": f">{threshold:.2f}",
                    "valid_split": False,
                    "invalid_reason": "analysis_error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                "binary_class_surv_object": None,
            }

    def _require_mpv_df(self) -> None:
        if getattr(self, "mpv_df", None) is None:
            raise RuntimeError(
                "create_mpv_data() must be called before this method (no mpv_df "
                "available yet)."
            )

    def create_mpv_data(self, replace: bool = False) -> pd.DataFrame:

        col_group = self.groups["col"]
        col_meta = dict(col_group.attrs.get("meta", {}))

        if col_meta.get("mpv_data_exists", False) and not replace:
            logger.info(
                "mpv data already exists for %s:%s. Set replace=True to "
                "create a new one.",
                self.surv_label,
                self.target_col_stats["name"],
            )
            self._create_mpv_df(replace=False)
            return self.mpv_df

        col_meta["mpv_data_exists"] = False
        col_meta["status"] = "pending"
        col_group.attrs["meta"] = col_meta

        self._create_mpv_df(replace=replace)

        if (
            "cox_ph.p_value" not in self.mpv_df.columns
            or self.mpv_df["cox_ph.p_value"].dropna().empty
        ):
            message = (
                f"No valid Cox-PH splits for {self.surv_label}:"
                f"{self.target_col_stats['name']}."
            )
            logger.warning(message)
            col_meta["status"] = "failed"
            col_meta["failure_reason"] = "no_valid_cox_splits"
            col_group.attrs["meta"] = col_meta
            return self.mpv_df

        self._save_marked_thresholds_data(replace=replace)

        _ = self.plot_dashboard(save_path=self.paths["plot_dashboard"])
        _ = self.plot_median_follow_up(save_path=self.paths["plot_median_follow_up"])
        _ = self.plot_hr_vs_pvalue_scatter(
            save_path=self.paths["plot_hr_vs_p_value_scatter"]
        )

        col_meta["mpv_data_exists"] = True
        col_meta["status"] = "completed"
        col_meta["failure_reason"] = None
        col_group.attrs["meta"] = col_meta
        return self.mpv_df

    def _create_mpv_df(self, replace: bool) -> pd.DataFrame:

        if self.paths["mpv_df"].exists() and not replace:
            logger.info(
                f"mpv_df already exists at {self.paths['mpv_df']}, set replace=True to create a new one"
            )
            self.mpv_df = pd.read_parquet(self.paths["mpv_df"])
            self.marked_threshold_dicts = self._build_marked_threshold_dicts()
            return self.mpv_df

        thresholds = self._get_thresholds()
        iterator = tqdm(thresholds) if self.show_progress else thresholds
        mpv_dicts: list[dict] = []
        for threshold in iterator:
            mpv_data = self._evaluate_threshold(threshold=float(threshold))
            mpv_dicts.append(mpv_data["mpv_dict"])

        target_median = float(self.target_col_stats["median"])
        if not np.isclose(thresholds, target_median).any():
            median_mpv_data = self._evaluate_threshold(threshold=target_median)
            mpv_dicts.append(median_mpv_data["mpv_dict"])

        mpv_df = pd.json_normalize(data=mpv_dicts)
        if mpv_df.empty:
            mpv_df = pd.DataFrame(
                columns=[
                    "threshold",
                    "group0_label",
                    "group1_label",
                    "valid_split",
                    "invalid_reason",
                    "error_type",
                    "error_message",
                ]
            )
        else:
            mpv_df = mpv_df.sort_values(by="threshold").reset_index(drop=True)

        self._add_multiplicity_columns(mpv_df=mpv_df)
        mpv_df.to_parquet(self.paths["mpv_df"], index=False)
        mpv_df.to_csv(self.paths["mpv_df"].with_suffix(suffix=".csv"), index=False)

        self.mpv_df = mpv_df
        self.marked_threshold_dicts = self._build_marked_threshold_dicts()

        return self.mpv_df

    def _add_multiplicity_columns(self, *, mpv_df: pd.DataFrame) -> None:
        """Add separate multiplicity results and family sizes to ``mpv_df``.

        Cox-PH and log-rank p-values are separate correction families. Their
        reported family sizes must therefore be computed independently using
        the same finite-value rule as ``holm_adjust_with_missing``.

        This method deliberately does not rewrite historical artifacts. New
        MPV artifacts contain the per-family fields; existing artifacts loaded
        with ``replace=False`` retain their original schema.
        """

        p_value_families = (
            ("cox_ph.p_value", "cox_ph.multiplicity.n_tests"),
            ("log_rank.p_value", "log_rank.multiplicity.n_tests"),
        )

        for p_value_column, count_column in p_value_families:
            if p_value_column not in mpv_df.columns:
                mpv_df[count_column] = 0
                continue

            values = pd.to_numeric(mpv_df[p_value_column], errors="coerce")
            raw_values = values.to_numpy(dtype=float, na_value=np.nan)
            finite = np.isfinite(raw_values)
            mpv_df[count_column] = int(finite.sum())

            if self.multiplicity_method == "holm":
                adjusted = holm_adjust_with_missing(raw_values)
            else:
                adjusted = raw_values
            mpv_df[f"{p_value_column}_{self.multiplicity_method}"] = adjusted

        mpv_df["multiplicity.method"] = self.multiplicity_method

    def _save_marked_thresholds_data(self, replace):

        self._require_mpv_df()

        if self.paths["marked_thresholds_df"].exists() and not replace:
            return

        tests_dicts = []
        for threshold_dict in self.marked_threshold_dicts:
            idx = threshold_dict["idx"]

            if idx is None:
                logger.info(f"No valid cut-off for {threshold_dict['label']}.")
                continue

            threshold = self.mpv_df.iloc[idx]["threshold"]
            mpv_data = self._evaluate_threshold(threshold=float(threshold))
            bcs = mpv_data["binary_class_surv_object"]
            if bcs is None:
                logger.warning(
                    "Marked threshold %s is not analyzable; skipping its KM "
                    "curve and detailed test artifact.",
                    threshold,
                )
                continue

            save_path = (
                self.paths["base"]
                / f"km_curve_{threshold_dict['label']}:{threshold}.png"
            )
            bcs.plot_km_curves(plot=False, save_path=save_path)

            tests_dict = bcs.get_tests_dict()
            tests_dict = pd.json_normalize(tests_dict).to_dict(orient="records")[0]
            tests_dict["threshold"] = threshold

            tests_dicts.append(tests_dict)

        if not tests_dicts:
            logger.warning("No marked thresholds produced detailed results.")
            return

        tests_df = pd.DataFrame(data=tests_dicts).set_index(["threshold"])

        tests_df.to_parquet(path=self.paths["marked_thresholds_df"])
        tests_df.to_csv(
            self.paths["marked_thresholds_df"].with_suffix(suffix=".csv"), index=True
        )

    """
    Plot methods.
    """

    def _build_marked_threshold_dicts(self) -> list[dict]:
        """
        Compute the three reference markers (median / min p-value / closest
        significant threshold to median) from self.mpv_df. Called once by
        get_mpv_df() right after self.mpv_df is (re)assigned, and cached as
        self.marked_threshold_dicts -- not recomputed elsewhere, so it always
        reflects whichever mpv_df was most recently loaded or generated.
        """
        mpv_df = self.mpv_df
        target_median = self.target_col_stats["median"]
        adjusted_column = f"cox_ph.p_value_{self.multiplicity_method}"
        p_value_col = (
            adjusted_column if adjusted_column in mpv_df.columns else "cox_ph.p_value"
        )

        if mpv_df.empty or "threshold" not in mpv_df.columns:
            return [
                {"label": "Median", "idx": None, "color": "blue", "ls": "--"},
                {
                    "label": "Min P-Val",
                    "idx": None,
                    "color": "gray",
                    "ls": "--",
                },
                {
                    "label": "Closest to Median",
                    "idx": None,
                    "color": "green",
                    "ls": "--",
                },
            ]

        median_matches = mpv_df.index[mpv_df["threshold"] == target_median]
        median_idx = (
            median_matches[0]
            if len(median_matches) > 0
            else (mpv_df["threshold"] - target_median).abs().idxmin()
        )

        valid_rows = mpv_df
        if "valid_split" in valid_rows.columns:
            valid_rows = valid_rows[valid_rows["valid_split"].fillna(False)]
        if p_value_col in valid_rows.columns:
            valid_rows = valid_rows[valid_rows[p_value_col].notna()]

        if not valid_rows.empty and p_value_col in valid_rows.columns:
            min_p_val_idx = valid_rows[p_value_col].idxmin()
            sig = valid_rows[valid_rows[p_value_col] < self.alpha]
            closest_idx = (
                None if sig.empty else (sig["threshold"] - target_median).abs().idxmin()
            )
        else:
            closest_idx = None
            min_p_val_idx = None
            logger.warning("No valid splits for Cox-PH, skipping min p-value marker.")

        return [
            {"label": "Median", "idx": median_idx, "color": "blue", "ls": "--"},
            {"label": "Min P-Val", "idx": min_p_val_idx, "color": "gray", "ls": "--"},
            {
                "label": "Closest to Median",
                "idx": closest_idx,
                "color": "green",
                "ls": "--",
            },
        ]

    def _add_threshold_markers(self, ax) -> None:
        """
        Draw the three cached reference markers (self.marked_threshold_dicts) as
        vertical lines on `ax`, skipping any whose idx is None. Shared by
        every plot method below.
        """
        for threshold_dict in self.marked_threshold_dicts:
            if threshold_dict["idx"] is None:
                continue
            threshold = float(self.mpv_df.loc[threshold_dict["idx"], "threshold"])
            ax.axvline(
                threshold,
                color=threshold_dict["color"],
                ls=threshold_dict["ls"],
                lw=1,
                label=threshold_dict["label"],
            )

    def plot_hr_with_ci(
        self,
        ax=None,
        title=None,
        grid=True,
        figsize=(11, 5),
        log_scale=True,
        save_path=None,
        dpi=300,
    ):
        """Cox hazard ratio (log scale) with shaded 95% CI, across thresholds."""
        self._require_mpv_df()

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=figsize)

        x = self.mpv_df["threshold"]
        ax.plot(
            x,
            self.mpv_df["cox_ph.hr.raw.hr"],
            color="tab:purple",
            lw=1.4,
            marker=".",
            label="Cox HR",
        )
        ax.fill_between(
            x,
            self.mpv_df["cox_ph.hr.raw.ci_lower"],
            self.mpv_df["cox_ph.hr.raw.ci_upper"],
            color="tab:purple",
            alpha=0.15,
            label="95% CI",
        )
        ax.axhline(1.0, color="black", lw=0.8, alpha=0.5, label="HR = 1 (no effect)")

        self._add_threshold_markers(ax)

        ax.set_xlabel("Threshold")
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("Hazard Ratio (log scale)")
        else:
            ax.set_ylabel("Hazard Ratio")

        if title is None:
            ax.set_title(
                f"Cox proportional-hazards ratio across scanned thresholds\n{self.surv_label}:{self.target_col_stats['name']}"
            )
        else:
            ax.set_title(title)

        ax.legend(loc="upper right", fontsize=8)

        if grid:
            ax.grid(True, which="both", alpha=0.2)

        if save_path is not None:
            ax.figure.savefig(save_path, dpi=dpi, bbox_inches="tight")

        if own_fig:
            fig.tight_layout()
            return fig

        return ax

    def plot_p_values(
        self,
        p_value="both",
        ax=None,
        title=None,
        grid=True,
        figsize=(11, 5),
        log_scale=True,
        save_path=None,
        dpi=300,
    ):
        """
        Cox PH and/or log-rank p-values across the same threshold axis.

        p_value: "cox_ph", "log_rank", or "both" (default).
        """
        self._require_mpv_df()

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=figsize)

        x = self.mpv_df["threshold"]
        if "cox_ph.p_value" in self.mpv_df.columns:
            cox_p = (
                np.clip(self.mpv_df["cox_ph.p_value"].to_numpy(), 1e-300, None)
                if log_scale
                else self.mpv_df["cox_ph.p_value"].to_numpy()
            )
        else:
            cox_p = np.full_like(x, np.nan)
            logger.warning("No valid splits for Cox-PH, skipping Cox PH p-value plot.")

        if "log_rank.p_value" in self.mpv_df.columns:
            lr_raw = self.mpv_df["log_rank.p_value"].to_numpy(dtype=float)
            lr_p = np.clip(lr_raw, 1e-300, None) if log_scale else lr_raw
        else:
            lr_p = np.full(len(self.mpv_df), np.nan)

        cox_adjusted_column = f"cox_ph.p_value_{self.multiplicity_method}"
        log_rank_adjusted_column = f"log_rank.p_value_{self.multiplicity_method}"

        def plot_adjusted(*, column: str, color: str, label: str) -> None:
            if self.multiplicity_method == "none":
                return
            if column not in self.mpv_df.columns:
                return
            adjusted = self.mpv_df[column].to_numpy(dtype=float)
            if log_scale:
                adjusted = np.clip(adjusted, 1e-300, None)
            ax.plot(
                x,
                adjusted,
                color=color,
                lw=1.6,
                ls="--",
                label=label,
            )

        if p_value == "cox_ph":
            ax.plot(
                x, cox_p, color="tab:blue", lw=1.2, marker=".", label="Cox PH p-value"
            )
            plot_adjusted(
                column=cox_adjusted_column,
                color="navy",
                label=f"Cox PH ({self.multiplicity_method}-adjusted)",
            )
            default_title = f"Cox PH p-value across scanned thresholds\n{self.surv_label}:{self.target_col_stats['name']}"
        elif p_value == "log_rank":
            ax.plot(
                x,
                lr_p,
                color="tab:orange",
                lw=1.2,
                marker=".",
                label="Log-rank p-value",
            )
            plot_adjusted(
                column=log_rank_adjusted_column,
                color="darkorange",
                label=f"Log-rank ({self.multiplicity_method}-adjusted)",
            )
            default_title = f"Log-rank p-value across scanned thresholds\n{self.surv_label}:{self.target_col_stats['name']}"
        elif p_value == "both":
            ax.plot(
                x, cox_p, color="tab:blue", lw=1.2, marker=".", label="Cox PH p-value"
            )
            ax.plot(
                x,
                lr_p,
                color="tab:orange",
                lw=1.2,
                marker=".",
                label="Log-rank p-value",
            )
            plot_adjusted(
                column=cox_adjusted_column,
                color="navy",
                label=f"Cox PH ({self.multiplicity_method}-adjusted)",
            )
            plot_adjusted(
                column=log_rank_adjusted_column,
                color="darkorange",
                label=f"Log-rank ({self.multiplicity_method}-adjusted)",
            )
            default_title = f"Cox PH vs. log-rank p-value across scanned thresholds\n{self.surv_label}:{self.target_col_stats['name']}"
        else:
            raise ValueError('p_value must be one of "cox_ph", "log_rank", or "both"')

        ax.axhline(
            self.alpha, color="tab:red", ls="--", lw=1, label=f"alpha = {self.alpha}"
        )

        self._add_threshold_markers(ax)

        ax.set_xlabel("Threshold")
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("p-value (log scale)")
        else:
            ax.set_ylabel("p-value")

        if title is None:
            ax.set_title(default_title)
        else:
            ax.set_title(title)

        ax.legend(loc="upper right", fontsize=8)

        if grid:
            ax.grid(True, which="both", alpha=0.2)

        if save_path is not None:
            ax.figure.savefig(save_path, dpi=dpi, bbox_inches="tight")

        if own_fig:
            fig.tight_layout()
            return fig
        return ax

    def plot_ci_width(
        self,
        ax=None,
        title=None,
        grid=True,
        figsize=(11, 4.5),
        log_scale=True,
        save_path=None,
        dpi=300,
    ):
        """
        Width of the Cox HR confidence interval across thresholds (upper /
        lower, log scale, since HR CIs are multiplicative). Narrower = more
        stable estimate at that threshold.
        """
        self._require_mpv_df()

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=figsize)

        x = self.mpv_df["threshold"]
        ci_ratio = (
            self.mpv_df["cox_ph.hr.raw.ci_upper"]
            / self.mpv_df["cox_ph.hr.raw.ci_lower"]
        )

        ax.plot(x, ci_ratio, color="tab:brown", lw=1.2, marker=".")

        self._add_threshold_markers(ax)

        ax.set_xlabel("Threshold")
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("CI Width\n(Upper / Lower, log scale)")
        else:
            ax.set_ylabel("CI Width\n(Upper / Lower)")

        if title is None:
            ax.set_title(
                f"Cox HR confidence-interval width across thresholds\n(narrower = more stable estimate)\n{self.surv_label}:{self.target_col_stats['name']}"
            )
        else:
            ax.set_title(title)

        ax.legend(loc="upper right", fontsize=8)

        if grid:
            ax.grid(True, which="both", alpha=0.2)

        if save_path is not None:
            ax.figure.savefig(save_path, dpi=dpi, bbox_inches="tight")

        if own_fig:
            fig.tight_layout()
            return fig
        return ax

    def plot_median_survival(
        self,
        ax=None,
        cap_not_reached=True,
        title=None,
        grid=True,
        figsize=(11, 5),
        save_path=None,
        dpi=300,
    ):
        """
        Median survival time per group across thresholds, with shaded CI
        bands. "Not reached" is np.inf in the raw columns; by default
        (cap_not_reached=True) it's capped at 1.15x the largest finite value
        so it plots near the top of the axis rather than vanishing or
        breaking y-limits. Set cap_not_reached=False to leave it as a NaN gap
        instead.
        """
        self._require_mpv_df()

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=figsize)

        x = self.mpv_df["threshold"]
        g0_est = self.mpv_df["log_rank.group0_median_survival.raw.median"]
        g0_lo = self.mpv_df["log_rank.group0_median_survival.raw.ci_lower"]
        g0_hi = self.mpv_df["log_rank.group0_median_survival.raw.ci_upper"]
        g1_est = self.mpv_df["log_rank.group1_median_survival.raw.median"]
        g1_lo = self.mpv_df["log_rank.group1_median_survival.raw.ci_lower"]
        g1_hi = self.mpv_df["log_rank.group1_median_survival.raw.ci_upper"]

        if cap_not_reached:
            finite_vals = pd.concat(
                [
                    s.replace([np.inf, -np.inf], np.nan).dropna()
                    for s in (g0_est, g0_hi, g1_est, g1_hi)
                ]
            )
            cap = float(finite_vals.max()) * 1.15 if len(finite_vals) else 1.0
            g0_est_plot, g0_hi_plot = g0_est.replace(np.inf, cap), g0_hi.replace(
                np.inf, cap
            )
            g1_est_plot, g1_hi_plot = g1_est.replace(np.inf, cap), g1_hi.replace(
                np.inf, cap
            )
            ax.axhline(
                cap,
                color="gray",
                ls="--",
                lw=0.8,
                alpha=0.5,
                label='Capped = "Not Reached"',
            )
        else:
            g0_est_plot, g0_hi_plot = g0_est.replace(np.inf, np.nan), g0_hi.replace(
                np.inf, np.nan
            )
            g1_est_plot, g1_hi_plot = g1_est.replace(np.inf, np.nan), g1_hi.replace(
                np.inf, np.nan
            )

        ax.plot(
            x,
            g0_est_plot,
            color="tab:blue",
            lw=1.4,
            marker=".",
            label="Group 0 median survival",
        )
        ax.fill_between(x, g0_lo, g0_hi_plot, color="tab:blue", alpha=0.12)

        ax.plot(
            x,
            g1_est_plot,
            color="tab:orange",
            lw=1.4,
            marker=".",
            label="Group 1 median survival",
        )
        ax.fill_between(x, g1_lo, g1_hi_plot, color="tab:orange", alpha=0.12)

        self._add_threshold_markers(ax)

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Median Survival Time")

        if title is None:
            subtitle = (
                '("not reached" capped near top)'
                if cap_not_reached
                else '(gaps = "not reached")'
            )
            ax.set_title(
                f"Median survival per group across scanned thresholds\n{subtitle}\n{self.surv_label}:{self.target_col_stats['name']}"
            )
        else:
            ax.set_title(title)

        ax.legend(loc="upper right", fontsize=8)

        if grid:
            ax.grid(True, alpha=0.2)

        if save_path is not None:
            ax.figure.savefig(save_path, dpi=dpi, bbox_inches="tight")

        if own_fig:
            fig.tight_layout()
            return fig
        return ax

    def plot_median_follow_up(
        self, ax=None, title=None, grid=True, figsize=(11, 4.5), save_path=None, dpi=300
    ):
        """Median follow-up time per group across thresholds (sanity check)."""
        self._require_mpv_df()

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=figsize)

        x = self.mpv_df["threshold"]

        ax.plot(
            x,
            self.mpv_df["log_rank.group0_median_follow_up.raw.median"],
            color="tab:blue",
            lw=1.2,
            marker=".",
            label="Group 0 median follow-up",
        )
        ax.fill_between(
            x,
            self.mpv_df["log_rank.group0_median_follow_up.raw.ci_lower"],
            self.mpv_df["log_rank.group0_median_follow_up.raw.ci_upper"],
            color="tab:blue",
            alpha=0.10,
        )

        ax.plot(
            x,
            self.mpv_df["log_rank.group1_median_follow_up.raw.median"],
            color="tab:orange",
            lw=1.2,
            marker=".",
            label="Group 1 median follow-up",
        )
        ax.fill_between(
            x,
            self.mpv_df["log_rank.group1_median_follow_up.raw.ci_lower"],
            self.mpv_df["log_rank.group1_median_follow_up.raw.ci_upper"],
            color="tab:orange",
            alpha=0.10,
        )

        self._add_threshold_markers(ax)

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Median Follow-up Time")

        if title is None:
            ax.set_title(
                f"Median follow-up per group across scanned thresholds\n{self.surv_label}:{self.target_col_stats['name']}"
            )
        else:
            ax.set_title(title)

        ax.legend(loc="upper right", fontsize=8)

        if grid:
            ax.grid(True, alpha=0.2)

        if save_path is not None:
            ax.figure.savefig(save_path, dpi=dpi, bbox_inches="tight")

        if own_fig:
            fig.tight_layout()
            return fig
        return ax

    def plot_group_sizes(
        self, ax=None, title=None, grid=True, figsize=(11, 4.5), save_path=None, dpi=300
    ):
        """Absolute group sizes (group0_n, group1_n) across thresholds."""
        self._require_mpv_df()

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=figsize)

        x = self.mpv_df["threshold"]
        ax.plot(
            x,
            self.mpv_df["log_rank.group0_n"],
            color="tab:blue",
            lw=1.4,
            marker=".",
            label="Group 0 n",
        )
        ax.plot(
            x,
            self.mpv_df["log_rank.group1_n"],
            color="tab:orange",
            lw=1.4,
            marker=".",
            label="Group 1 n",
        )

        total = self.mpv_df["log_rank.group0_n"] + self.mpv_df["log_rank.group1_n"]
        ax.plot(
            x,
            total,
            color="gray",
            lw=1.0,
            ls="--",
            alpha=0.6,
            label="Total n (sanity check)",
        )

        self._add_threshold_markers(ax)

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Group size (n)")

        if title is None:
            ax.set_title(
                f"Absolute group sizes across scanned thresholds\n{self.surv_label}:{self.target_col_stats['name']}"
            )
        else:
            ax.set_title(title)

        ax.legend(loc="upper right", fontsize=8)

        if grid:
            ax.grid(True, alpha=0.2)

        if save_path is not None:
            ax.figure.savefig(save_path, dpi=dpi, bbox_inches="tight")

        if own_fig:
            fig.tight_layout()
            return fig
        return ax

    def plot_split_ratio(
        self,
        imbalance_factor=10.0,
        ax=None,
        title=None,
        grid=True,
        figsize=(11, 4.5),
        log_scale=True,
        save_path=None,
        dpi=300,
    ):
        """
        Group0_n / group1_n split ratio across thresholds (log scale), with
        a shaded "imbalanced" zone beyond `imbalance_factor`:1 in either
        direction.
        """
        self._require_mpv_df()

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=figsize)

        x = self.mpv_df["threshold"]
        ratio = self.mpv_df["split_ratio"]

        danger_lo, danger_hi = 1.0 / imbalance_factor, imbalance_factor
        ax.axhspan(
            danger_hi,
            max(ratio.max() * 1.1, danger_hi * 1.1),
            color="tab:red",
            alpha=0.08,
        )
        ax.axhspan(
            min(ratio.min() * 0.9, danger_lo * 0.9),
            danger_lo,
            color="tab:red",
            alpha=0.08,
            label=f"Imbalanced (>{imbalance_factor:g}:1)",
        )

        ax.plot(x, ratio, color="tab:green", lw=1.2, marker=".")
        ax.axhline(1.0, color="black", lw=0.8, alpha=0.4, label="Balanced (1:1)")

        self._add_threshold_markers(ax)

        ax.set_xlabel("Threshold")
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("Split ratio\n(group0_n / group1_n, log)")
        else:
            ax.set_ylabel("Split ratio\n(group0_n / group1_n)")

        if title is None:
            ax.set_title(
                f"Group split ratio across scanned thresholds\n{self.surv_label}:{self.target_col_stats['name']}"
            )
        else:
            ax.set_title(title)

        ax.legend(loc="upper right", fontsize=8)

        if grid:
            ax.grid(True, which="both", alpha=0.2)

        if save_path is not None:
            ax.figure.savefig(save_path, dpi=dpi, bbox_inches="tight")

        if own_fig:
            fig.tight_layout()
            return fig
        return ax

    def plot_hr_vs_pvalue_scatter(
        self,
        color_by="threshold",
        ax=None,
        title=None,
        grid=True,
        save_path=None,
        dpi=300,
    ):
        """
        Scatter of Cox HR (x, log scale) vs. Cox p-value (y, log scale),
        colored by `color_by` (default: "threshold"; alternative:
        "split_ratio"). The three reference thresholds are drawn as open
        circles at their (HR, p-value) position rather than axvlines, since
        the x-axis here is HR, not threshold/index.
        """
        self._require_mpv_df()

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(7, 6))
        else:
            fig = ax.figure

        if color_by not in self.mpv_df.columns:
            raise ValueError(f"color_by={color_by!r} is not a result column.")
        color_vals = self.mpv_df[color_by]
        adjusted_column = f"cox_ph.p_value_{self.multiplicity_method}"
        p_value_column = (
            adjusted_column
            if adjusted_column in self.mpv_df.columns
            else "cox_ph.p_value"
        )
        sc = ax.scatter(
            self.mpv_df["cox_ph.hr.raw.hr"],
            np.clip(self.mpv_df[p_value_column], 1e-300, None),
            c=color_vals,
            cmap="viridis",
            s=28,
            edgecolor="none",
            zorder=2,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(color_by)

        for threshold_dict in self.marked_threshold_dicts:
            if threshold_dict["idx"] is None:
                continue
            row = self.mpv_df.loc[threshold_dict["idx"]]
            ax.scatter(
                row["cox_ph.hr.raw.hr"],
                max(row[p_value_column], 1e-300),
                s=140,
                facecolor="none",
                edgecolor=threshold_dict["color"],
                linewidth=1.8,
                zorder=3,
                label=threshold_dict["label"],
            )

        ax.axhline(
            self.alpha, color="tab:red", ls="--", lw=1, label=f"alpha = {self.alpha}"
        )
        ax.axvline(1.0, color="black", lw=0.8, alpha=0.4, label="HR = 1")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Hazard Ratio (log scale)")
        ax.set_ylabel(f"Cox p-value ({self.multiplicity_method}-adjusted, log scale)")

        if title is None:
            ax.set_title(
                f"HR vs. p-value across thresholds, colored by {color_by}\n{self.surv_label}:{self.target_col_stats['name']}"
            )
        else:
            ax.set_title(title)

        ax.legend(loc="best", fontsize=8)

        if grid:
            ax.grid(True, which="both", alpha=0.2)

        if save_path is not None:
            ax.figure.savefig(save_path, dpi=dpi, bbox_inches="tight")

        if own_fig:
            fig.tight_layout()
            return fig
        return ax

    def plot_dashboard(
        self, imbalance_factor=10.0, title=None, grid=True, save_path=None, dpi=300
    ):
        """
        Multi-panel dashboard sharing one threshold x-axis:
          1. Cox + log-rank p-values
          2. Cox HR with CI band
          3. Group sizes (n)
          4. Split ratio with imbalance danger zones
          5. Median survival per group with CI bands ("not reached" capped)
        """
        self._require_mpv_df()

        fig, axes = plt.subplots(
            5,
            1,
            figsize=(12, 16),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 2, 1.3, 1.3, 2]},
        )
        ax_p, ax_hr, ax_n, ax_ratio, ax_surv = axes

        self.plot_p_values(p_value="both", ax=ax_p, grid=grid)
        ax_p.set_xlabel("")

        self.plot_hr_with_ci(ax=ax_hr, grid=grid)
        ax_hr.set_xlabel("")
        ax_hr.set_title("Cox Hazard Ratio (log scale) with 95% CI")

        self.plot_group_sizes(ax=ax_n, grid=grid)
        ax_n.set_title("")
        ax_n.set_xlabel("")

        self.plot_split_ratio(imbalance_factor=imbalance_factor, ax=ax_ratio, grid=grid)
        ax_ratio.set_title("")
        ax_ratio.set_xlabel("")

        self.plot_median_survival(ax=ax_surv, cap_not_reached=True, grid=grid)
        ax_surv.set_title("")

        axes[-1].set_xlabel("Threshold")
        fig.suptitle(
            (
                f"Threshold scan dashboard\n{self.surv_label}:{self.target_col_stats['name']}"
                if title is None
                else title
            ),
            fontsize=13,
            y=1.01,
        )
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return fig
