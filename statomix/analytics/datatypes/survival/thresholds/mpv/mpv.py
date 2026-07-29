import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from openpyxl import load_workbook
from openpyxl.worksheet.datavalidation import DataValidation

from fileverse.logger import Logger
from fileverse.formats.zarr import BaseZARR
from fileverse.formats.excel import BaseExcel
from fileverse.clean_path_name import clean_path_name

from statomix.analytics.datatypes.survival.binary_class_surv import BinaryClassSurv

logger = Logger(name="MinimumPValue").get_logger()


class MinimumPValue:
    MODULE_NAME = "Survival -Threshold MPV"
    def __init__(
        self, 
        surv_label:str,
        surv_df_mpv:pd.DataFrame,
        root_group,
        trunc_pct = None,
        iqr_multiplier = 1.5,
        use_synthetic_cutoffs:bool = False,
        search_resolution: float = 0.5,
        show_progress: bool =True,
        alpha: float = 0.05,
    ):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        if use_synthetic_cutoffs and search_resolution <= 0:
            raise ValueError(
                f"search_resolution must be positive, got {search_resolution!r}"
            )

        self.alpha = alpha
        self.trunc_pct = trunc_pct
        self.surv_label = surv_label
        self.surv_df_mpv = surv_df_mpv
        self.show_progress = show_progress
        self.iqr_multiplier = iqr_multiplier
        self.search_resolution = search_resolution
        self.use_synthetic_cutoffs = use_synthetic_cutoffs

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
        #self.target_col_stats['name'] = grouping_cols[0]
        #self.self.target_col_stats['name'] = self.target_col_stats['name']
        self.target_col_stats = {
            "name": grouping_cols[0],
            "median": self.surv_df_mpv[grouping_cols[0]].median()
        }

        if not pd.api.types.is_numeric_dtype(self.surv_df_mpv[self.target_col_stats['name']]):
            raise ValueError(
                f"'{self.target_col_stats['name']}' must be numeric to search over "
                "thresholds with <=/> comparisons."
            )

        self._create_groups(root_group=root_group)
        self._create_paths()

    @staticmethod
    def get_config_df():
        return pd.DataFrame(columns=["Numerical", "Survival Labels"])

    @staticmethod
    def add_validation_to_analysis_config_file(path, max_row=500):
        workbook = load_workbook(filename=path)
        datatype_col_map = BaseExcel.get_worksheet_col_map(workbook['Datatype Map'])
    
        worksheet = workbook[MinimumPValue.MODULE_NAME]
        module_col_map = BaseExcel.get_worksheet_col_map(worksheet)
    
        for key, value in module_col_map.items():
            cell_coordinate = datatype_col_map[key]
        
            n_options = len(workbook['Datatype Map'][cell_coordinate])
        
            validation = DataValidation(
                type="list",
                formula1=f"'Datatype Map'!${cell_coordinate}$2:${cell_coordinate}${n_options}",
                #allow_blank=True,
                #showErrorMessage=True,
                #errorStyle="stop",
                #errorTitle="Invalid Datatype",
                #error="You must select a valid datatype from the provided drop-down menu.",
            )
            
            worksheet.add_data_validation(validation)
            validation.add(f"{value}2:{value}{max_row}")
        
        workbook.save(filename=path)
        
    def _create_groups(self, root_group):
        self.groups = {}
        self.groups['root'] = root_group
        clean_col_name = clean_path_name(path=self.target_col_stats['name'])
        self.groups['col'] = self.groups['root'].root_group.require_group(f"{str(clean_col_name)}_trunc_pct_{self.trunc_pct}_iqr_multiplier_{self.iqr_multiplier}")

        col_group = self.groups['col']
        col_meta = col_group.attrs.get("meta", {})
        col_meta["mpv_data_exists"] = False
        col_group.attrs['meta'] = col_meta

    def _create_paths(self):
        base_path = BaseZARR.get_abs_path(group=self.groups['col'])
        
        self.paths = {}
        self.paths['base'] =  base_path

        self.paths["mpv_df"] =  base_path/"mpv_df.parquet"
        self.paths['marked_thresholds_df'] = base_path/"marked_thresholds_df.parquet"

        self.paths['plot_dashboard'] =  base_path/"plot_dashboard.png"
        self.paths['plot_median_follow_up'] =  base_path/'plot_median_follow_up.png'
        self.paths['plot_hr_vs_p_value_scatter'] = base_path/"plot_hr_vs_p_value_scatter.png"

    def _get_thresholds(self) -> np.ndarray:
        target_col = self.surv_df_mpv[self.target_col_stats["name"]]
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
                    thresholds = thresholds[n_trim: n - n_trim]
        n_removed_trim = n_before_trim - len(thresholds)
    
        logger.info(
            f"[Thresholds] Initial Candidates: {n_initial} | "
            f"Removed by IQR (k={iqr_multiplier}): {n_removed_iqr} | "
            f"Removed by percent-trim ({trunc_pct}%): {n_removed_trim} | "
            f"Final: {len(thresholds)}"
        )
    
        return thresholds

    def _get_mpv_data_at_threshold(self, threshold):
        low_label = f"<={threshold:.2f}"
        high_label = f">{threshold:.2f}"
        
        mpv_dict = {
            "threshold": threshold,
            "group0_label": low_label,
            "group1_label": high_label,
            "valid_split": False,  # overwritten to True on success below
        }
        
        surv_df_binary = self.surv_df_mpv.copy()
        is_low = surv_df_binary[self.target_col_stats["name"]] <= threshold
        surv_df_binary[self.target_col_stats["name"]] = np.where(is_low, low_label, high_label)
        
        if surv_df_binary[self.target_col_stats["name"]].nunique() != 2:
            # Degenerate split: every row landed on one side. Expected to
            # happen at the extremes of the threshold range -- not an
            # error, just not usable.
            return mpv_dict
        
        bcs = BinaryClassSurv(
            surv_df_binary=surv_df_binary,
            surv_label=self.surv_label,
            alpha=self.alpha,
            baseline_group=low_label,
        )

        tests_dict = bcs.get_tests_dict()
        #tests_dict['cox_ph']['hr_ci_low'] = tests_dict['cox_ph']['hr_ci'][0] 
        #tests_dict['cox_ph']['hr_ci_up'] = tests_dict['cox_ph']['hr_ci'][1]

        mpv_dict["split_ratio"] = tests_dict["split_ratio"]
        mpv_dict["cox_ph"] = tests_dict["cox_ph"]
        mpv_dict["log_rank"] =  tests_dict["log_rank"]
        
        return {"mpv_dict":mpv_dict, "binary_class_surv_object": bcs}
    
    def _require_mpv_df(self) -> None:
        if self.mpv_df is None:
            raise RuntimeError(
                "create_mpv_data() must be called before this method (no mpv_df "
                "available yet)."
            )

    def create_mpv_data(self, replace=False):

        col_group = self.groups['col']
        col_meta = col_group.attrs.get("meta", {})

        if col_meta["mpv_data_exists"] and not replace:
            logger.info(f"mpv data already exists for {self.surv_label}:{self.target_col_stats['name']}.\nSet replace=True to create a new one")
            return
            
        col_meta["mpv_data_exists"] = False
        
        self._create_mpv_df(replace=replace)
        
        self._save_marked_thresholds_data(replace=replace)
        
        _ = self.plot_dashboard(save_path=self.paths['plot_dashboard'])
        _ = self.plot_median_follow_up(save_path=self.paths['plot_median_follow_up'])
        _ = self.plot_hr_vs_pvalue_scatter(save_path=self.paths['plot_hr_vs_p_value_scatter'])

        col_meta["mpv_data_exists"] = True
        col_group.attrs['meta'] = col_meta

    def _create_mpv_df(self, replace):
        
        if self.paths['mpv_df'].exists() and not replace:
            logger.info(f"mpv_df already exists at {self.paths['mpv_df']}, set replace=True to create a new one")
            self.mpv_df = pd.read_parquet(self.paths['mpv_df'])
            self.marked_threshold_dicts = self._build_marked_threshold_dicts()
            return self.mpv_df
            
        thresholds = self._get_thresholds()
        iterator = tqdm(thresholds) if self.show_progress else thresholds
        mpv_dicts = []
        for threshold in iterator:
        
            mpv_data = self._get_mpv_data_at_threshold(threshold=threshold)
            mpv_dicts.append(mpv_data["mpv_dict"])

        if self.target_col_stats['median'] not in thresholds:
            median_mpv_data = self._get_mpv_data_at_threshold(threshold=self.target_col_stats['median'])
            mpv_dicts.append(median_mpv_data['mpv_dict']) 
        
        mpv_df = pd.json_normalize(data=mpv_dicts)
        # Sort thresholds sequentially and reset index to prevent scrambled index matching
        mpv_df = mpv_df.sort_values(by="threshold").reset_index(drop=True)
        mpv_df.to_parquet(self.paths['mpv_df'])
        mpv_df.to_csv(self.paths['mpv_df'].with_suffix(suffix=".csv"), index=False)
        
        self.mpv_df = mpv_df
        self.marked_threshold_dicts = self._build_marked_threshold_dicts()
        
        return self.mpv_df

    def _save_marked_thresholds_data(self, replace):
        
        self._require_mpv_df()
        
        if self.paths['marked_thresholds_df'].exists() and not replace:
            return
            
        tests_dicts = []
        for threshold_dict in self.marked_threshold_dicts:
            idx = threshold_dict['idx']
            threshold = self.mpv_df.iloc[idx]['threshold']
            mpv_data = self._get_mpv_data_at_threshold(threshold=threshold)
            bcs = mpv_data['binary_class_surv_object']
            
            save_path = self.paths['base']/f"km_curve_{threshold_dict['label']}:{threshold}.png"
            bcs.plot_km_curves(plot=False, save_path=save_path)
            
            tests_dict = bcs.get_tests_dict()
            tests_dict = pd.json_normalize(tests_dict).to_dict(orient='records')[0]
            tests_dict['threshold'] = threshold
        
            tests_dicts.append(tests_dict)
        
        tests_df = pd.DataFrame(data=tests_dicts).set_index(['threshold'])
        
        tests_df.to_parquet(path=self.paths['marked_thresholds_df'])
        tests_df.to_csv(self.paths['marked_thresholds_df'].with_suffix(suffix=".csv"), index=True)
    
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
        p_value_col = "cox_ph.p_value"
    
        median_matches = mpv_df.index[mpv_df["threshold"] == target_median]
        median_idx = median_matches[0] if len(median_matches) > 0 else (
            mpv_df["threshold"] - target_median
        ).abs().idxmin()
    
        min_p_val_idx = mpv_df[p_value_col].idxmin()
    
        sig = mpv_df[mpv_df[p_value_col] < self.alpha]
        closest_idx = None if sig.empty else (sig["threshold"] - target_median).abs().idxmin()
    
        return [
            {"label": "Median", "idx": median_idx, "color": "blue", "ls": "--"},
            {"label": "Min P-Val", "idx": min_p_val_idx, "color": "gray", "ls": "--"},
            {"label": "Closest to Median", "idx": closest_idx, "color": "green", "ls": "--"},
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
            ax.axvline(
                threshold_dict["idx"],
                color=threshold_dict["color"],
                ls=threshold_dict["ls"],
                lw=1,
                label=threshold_dict["label"],
            )
    
    
    def plot_hr_with_ci(self, ax=None, title=None, grid=True, figsize=(11, 5),
                         log_scale=True, save_path=None, dpi=300):
        """Cox hazard ratio (log scale) with shaded 95% CI, across thresholds."""
        self._require_mpv_df()
    
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=figsize)
    
        x = self.mpv_df.index
        ax.plot(x, self.mpv_df["cox_ph.hr.raw.hr"], color="tab:purple", lw=1.4, marker=".", label="Cox HR")
        ax.fill_between(x, self.mpv_df["cox_ph.hr.raw.ci_lower"], self.mpv_df["cox_ph.hr.raw.ci_upper"],
                         color="tab:purple", alpha=0.15, label="95% CI")
        ax.axhline(1.0, color="black", lw=0.8, alpha=0.5, label="HR = 1 (no effect)")
    
        self._add_threshold_markers(ax)
    
        ax.set_xlabel("Threshold")
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("Hazard Ratio (log scale)")
        else:
            ax.set_ylabel("Hazard Ratio")
    
        if title is None:
            ax.set_title(f"Cox proportional-hazards ratio across scanned thresholds\n{self.surv_label}:{self.target_col_stats['name']}")
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
    
    
    def plot_p_values(self, p_value="both", ax=None, title=None, grid=True, figsize=(11, 5),
                       log_scale=True, save_path=None, dpi=300):
        """
        Cox PH and/or log-rank p-values across the same threshold axis.
    
        p_value: "cox_ph", "log_rank", or "both" (default).
        """
        self._require_mpv_df()
    
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=figsize)
    
        x = self.mpv_df.index
        cox_p = (np.clip(self.mpv_df["cox_ph.p_value"].to_numpy(), 1e-300, None)
                  if log_scale else self.mpv_df["cox_ph.p_value"].to_numpy())
        lr_p = (np.clip(self.mpv_df["log_rank.p_value"].to_numpy(), 1e-300, None)
                 if log_scale else self.mpv_df["log_rank.p_value"].to_numpy())
    
        if p_value == "cox_ph":
            ax.plot(x, cox_p, color="tab:blue", lw=1.2, marker=".", label="Cox PH p-value")
            default_title = f"Cox PH p-value across scanned thresholds\n{self.surv_label}:{self.target_col_stats['name']}"
        elif p_value == "log_rank":
            ax.plot(x, lr_p, color="tab:orange", lw=1.2, marker=".", label="Log-rank p-value")
            default_title = f"Log-rank p-value across scanned thresholds\n{self.surv_label}:{self.target_col_stats['name']}"
        elif p_value == "both":
            ax.plot(x, cox_p, color="tab:blue", lw=1.2, marker=".", label="Cox PH p-value")
            ax.plot(x, lr_p, color="tab:orange", lw=1.2, marker=".", label="Log-rank p-value")
            default_title = f"Cox PH vs. log-rank p-value across scanned thresholds\n{self.surv_label}:{self.target_col_stats['name']}"
        else:
            raise ValueError('p_value must be one of "cox_ph", "log_rank", or "both"')
    
        ax.axhline(self.alpha, color="tab:red", ls="--", lw=1, label=f"alpha = {self.alpha}")
    
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
    
    
    def plot_ci_width(self, ax=None, title=None, grid=True, figsize=(11, 4.5),
                       log_scale=True, save_path=None, dpi=300):
        """
        Width of the Cox HR confidence interval across thresholds (upper /
        lower, log scale, since HR CIs are multiplicative). Narrower = more
        stable estimate at that threshold.
        """
        self._require_mpv_df()
    
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=figsize)
    
        x = self.mpv_df.index
        ci_ratio = self.mpv_df["cox_ph.hr.raw.ci_upper"] / self.mpv_df["cox_ph.hr.raw.ci_lower"]
    
        ax.plot(x, ci_ratio, color="tab:brown", lw=1.2, marker=".")
    
        self._add_threshold_markers(ax)
    
        ax.set_xlabel("Threshold")
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("CI Width\n(Upper / Lower, log scale)")
        else:
            ax.set_ylabel("CI Width\n(Upper / Lower)")
    
        if title is None:
            ax.set_title(f"Cox HR confidence-interval width across thresholds\n(narrower = more stable estimate)\n{self.surv_label}:{self.target_col_stats['name']}")
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
    
    
    def plot_median_survival(self, ax=None, cap_not_reached=True, title=None, grid=True,
                              figsize=(11, 5), save_path=None, dpi=300):
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
    
        x = self.mpv_df.index
        g0_est = self.mpv_df["log_rank.group0_median_survival.raw.median"]
        g0_lo = self.mpv_df["log_rank.group0_median_survival.raw.ci_lower"]
        g0_hi = self.mpv_df["log_rank.group0_median_survival.raw.ci_upper"]
        g1_est = self.mpv_df["log_rank.group1_median_survival.raw.median"]
        g1_lo = self.mpv_df["log_rank.group1_median_survival.raw.ci_lower"]
        g1_hi = self.mpv_df["log_rank.group1_median_survival.raw.ci_upper"]
    
        if cap_not_reached:
            finite_vals = pd.concat([s.replace([np.inf, -np.inf], np.nan).dropna()
                                      for s in (g0_est, g0_hi, g1_est, g1_hi)])
            cap = float(finite_vals.max()) * 1.15 if len(finite_vals) else 1.0
            g0_est_plot, g0_hi_plot = g0_est.replace(np.inf, cap), g0_hi.replace(np.inf, cap)
            g1_est_plot, g1_hi_plot = g1_est.replace(np.inf, cap), g1_hi.replace(np.inf, cap)
            ax.axhline(cap, color="gray", ls="--", lw=0.8, alpha=0.5, label='Capped = "Not Reached"')
        else:
            g0_est_plot, g0_hi_plot = g0_est.replace(np.inf, np.nan), g0_hi.replace(np.inf, np.nan)
            g1_est_plot, g1_hi_plot = g1_est.replace(np.inf, np.nan), g1_hi.replace(np.inf, np.nan)
    
        ax.plot(x, g0_est_plot, color="tab:blue", lw=1.4, marker=".", label="Group 0 median survival")
        ax.fill_between(x, g0_lo, g0_hi_plot, color="tab:blue", alpha=0.12)
    
        ax.plot(x, g1_est_plot, color="tab:orange", lw=1.4, marker=".", label="Group 1 median survival")
        ax.fill_between(x, g1_lo, g1_hi_plot, color="tab:orange", alpha=0.12)
    
        self._add_threshold_markers(ax)
    
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Median Survival Time")
    
        if title is None:
            subtitle = '("not reached" capped near top)' if cap_not_reached else '(gaps = "not reached")'
            ax.set_title(f"Median survival per group across scanned thresholds\n{subtitle}\n{self.surv_label}:{self.target_col_stats['name']}")
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
    
    
    def plot_median_follow_up(self, ax=None, title=None, grid=True, figsize=(11, 4.5),
                               save_path=None, dpi=300):
        """Median follow-up time per group across thresholds (sanity check)."""
        self._require_mpv_df()
    
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=figsize)
    
        x = self.mpv_df.index
    
        ax.plot(x, self.mpv_df["log_rank.group0_median_follow_up.raw.median"], color="tab:blue",
                lw=1.2, marker=".", label="Group 0 median follow-up")
        ax.fill_between(x, self.mpv_df["log_rank.group0_median_follow_up.raw.ci_lower"],
                         self.mpv_df["log_rank.group0_median_follow_up.raw.ci_upper"],
                         color="tab:blue", alpha=0.10)
    
        ax.plot(x, self.mpv_df["log_rank.group1_median_follow_up.raw.median"], color="tab:orange",
                lw=1.2, marker=".", label="Group 1 median follow-up")
        ax.fill_between(x, self.mpv_df["log_rank.group1_median_follow_up.raw.ci_lower"],
                         self.mpv_df["log_rank.group1_median_follow_up.raw.ci_upper"],
                         color="tab:orange", alpha=0.10)
    
        self._add_threshold_markers(ax)
    
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Median Follow-up Time")
    
        if title is None:
            ax.set_title(f"Median follow-up per group across scanned thresholds\n{self.surv_label}:{self.target_col_stats['name']}")
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
    
    
    def plot_group_sizes(self, ax=None, title=None, grid=True, figsize=(11, 4.5),
                          save_path=None, dpi=300):
        """Absolute group sizes (group0_n, group1_n) across thresholds."""
        self._require_mpv_df()
    
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=figsize)
    
        x = self.mpv_df.index
        ax.plot(x, self.mpv_df["log_rank.group0_n"], color="tab:blue", lw=1.4, marker=".", label="Group 0 n")
        ax.plot(x, self.mpv_df["log_rank.group1_n"], color="tab:orange", lw=1.4, marker=".", label="Group 1 n")
    
        total = self.mpv_df["log_rank.group0_n"] + self.mpv_df["log_rank.group1_n"]
        ax.plot(x, total, color="gray", lw=1.0, ls="--", alpha=0.6, label="Total n (sanity check)")
    
        self._add_threshold_markers(ax)
    
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Group size (n)")
    
        if title is None:
            ax.set_title(f"Absolute group sizes across scanned thresholds\n{self.surv_label}:{self.target_col_stats['name']}")
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
    
    
    def plot_split_ratio(self, imbalance_factor=10.0, ax=None, title=None, grid=True,
                          figsize=(11, 4.5), log_scale=True, save_path=None, dpi=300):
        """
        Group0_n / group1_n split ratio across thresholds (log scale), with
        a shaded "imbalanced" zone beyond `imbalance_factor`:1 in either
        direction.
        """
        self._require_mpv_df()
    
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=figsize)
    
        x = self.mpv_df.index
        ratio = self.mpv_df["split_ratio"]
    
        danger_lo, danger_hi = 1.0 / imbalance_factor, imbalance_factor
        ax.axhspan(danger_hi, max(ratio.max() * 1.1, danger_hi * 1.1), color="tab:red", alpha=0.08)
        ax.axhspan(min(ratio.min() * 0.9, danger_lo * 0.9), danger_lo, color="tab:red", alpha=0.08,
                   label=f"Imbalanced (>{imbalance_factor:g}:1)")
    
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
            ax.set_title(f"Group split ratio across scanned thresholds\n{self.surv_label}:{self.target_col_stats['name']}")
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
    
    
    def plot_hr_vs_pvalue_scatter(self, color_by="threshold", ax=None, title=None, grid=True,
                                   save_path=None, dpi=300):
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
    
        color_vals = self.mpv_df[color_by]
        sc = ax.scatter(
            self.mpv_df["cox_ph.hr.raw.hr"], np.clip(self.mpv_df["cox_ph.p_value"], 1e-300, None),
            c=color_vals, cmap="viridis", s=28, edgecolor="none", zorder=2,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(color_by)
    
        for threshold_dict in self.marked_threshold_dicts:
            if threshold_dict["idx"] is None:
                continue
            row = self.mpv_df.loc[threshold_dict["idx"]]
            ax.scatter(
                row["cox_ph.hr.raw.hr"], max(row["cox_ph.p_value"], 1e-300),
                s=140, facecolor="none", edgecolor=threshold_dict["color"],
                linewidth=1.8, zorder=3, label=threshold_dict["label"],
            )
    
        ax.axhline(self.alpha, color="tab:red", ls="--", lw=1, label=f"alpha = {self.alpha}")
        ax.axvline(1.0, color="black", lw=0.8, alpha=0.4, label="HR = 1")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Hazard Ratio (log scale)")
        ax.set_ylabel("Cox p-value (log scale)")
    
        if title is None:
            ax.set_title(f"HR vs. p-value across thresholds, colored by {color_by}\n{self.surv_label}:{self.target_col_stats['name']}")
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
    
    
    def plot_dashboard(self, imbalance_factor=10.0, title=None, grid=True, save_path=None, dpi=300):
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
            5, 1, figsize=(12, 16), sharex=True,
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
        fig.suptitle(f"Threshold scan dashboard\n{self.surv_label}:{self.target_col_stats['name']}" if title is None else title, fontsize=13, y=1.01)
        fig.tight_layout()
    
        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
        return fig