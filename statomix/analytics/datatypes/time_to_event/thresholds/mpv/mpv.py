import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from fileverse.logger import Logger
from fileverse.formats.zarr import BaseZARR
from fileverse.clean_path_name import clean_path_name

from statomix.analytics.datatypes.time_to_event.binary_class_surv import BinaryClassSurv

logger = Logger(name="MinimumPValue").get_logger()


class MinimumPValue:
    def __init__(
        self, 
        surv_label:str,
        surv_df_mpv:pd.DataFrame,
        root_group,
        trunc_pct = 2,
        iqr_multiplier = None,
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
        
    def _create_groups(self, root_group):
        self.groups = {}
        self.groups['root'] = root_group
        clean_col_name = clean_path_name(path=self.target_col_stats['name'])
        self.groups['col'] = self.groups['root'].root_group.require_group(f"{str(clean_col_name)}_trunc_pct_{self.trunc_pct}_iqr_multiplier_{self.iqr_multiplier}")

    def _create_paths(self):
        self.paths = {}
        self.paths['base'] =  BaseZARR.get_abs_path(group=self.groups['col'])
        self.paths["mpv_df"] =  self.paths['base']/"mpv_df.parquet"

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
        tests_dict['cox_ph']['hr_ci_low'] = tests_dict['cox_ph']['hr_ci'][0] 
        tests_dict['cox_ph']['hr_ci_up'] = tests_dict['cox_ph']['hr_ci'][1]

        mpv_dict["split_ratio"] = tests_dict["split_ratio"]
        mpv_dict["cox_ph"] = tests_dict["cox_ph"]
        mpv_dict["log_rank"] =  tests_dict["log_rank"]
        
        return {"mpv_dict":mpv_dict, "binary_class_surv_object": bcs}

    def get_mpv_df(self, replace=False):
        
        if self.paths['mpv_df'].exists() and not replace:
            logger.info(f"mpv_df already exists at {self.paths['mpv_df']}, set replace=True to create a new one")
            self.mpv_df = pd.read_parquet(self.paths['mpv_df'])
            return mpv_df
            
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
        
        self.mpv_df = mpv_df
        return self.mpv_df