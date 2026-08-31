"""Descriptive threshold scanning with optional maxstat inference."""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fileverse.clean_path_name import clean_path_name
from fileverse.formats.zarr import BaseZARR
from tqdm.auto import tqdm

from statomix.analysis.multiplicity import (
    CORRECTION_REGISTRY,
    adjust_p_values_with_missing,
    normalize_correction_methods,
)
from statomix.analysis.survival.binary import BinaryClassSurv
from statomix.analysis.survival.data import prepare_survival_data
from statomix.logging import get_logger

from .maxstat import (
    MaximallySelectedLogRank,
    MaxstatMethod,
    MaxstatResult,
)
from .scan import ThresholdScan

logger = get_logger(name="MinimumPValue")

_LINE_MARKER_SIZE = 2.5
_SCATTER_POINT_SIZE = 16
_SCAN_SCHEMA_VERSION = 2
_ANALYSIS_MODES = ("exploratory", "inferential")
_SELECTION_FAMILIES = ("cox_ph", "log_rank")


class MinimumPValue:
    """Orchestrate descriptive cutoff scans and optional global inference.

    ``mode`` must be chosen explicitly.  Exploratory mode exposes raw and
    optionally adjusted per-cutoff p-values, all visibly marked as
    post-selection descriptions.  Inferential mode adds one maximally selected
    log-rank p-value; row-wise values and Cox estimates remain secondary.
    """

    MODULE_NAME = "Survival -Threshold MPV"

    def __init__(
        self,
        surv_label: str,
        surv_df_mpv: pd.DataFrame,
        root_group,
        trunc_pct: float | None = None,
        iqr_multiplier: float | None = None,
        use_synthetic_cutoffs: bool = False,
        search_resolution: float = 0.5,
        show_progress: bool = True,
        alpha: float = 0.05,
        multiplicity_method: str | None = None,
        correction_methods: str | Sequence[str] | None = None,
        selection_method: str = "none",
        *,
        mode: Literal["exploratory", "inferential"] | None = None,
        minprop: float = 0.1,
        maxprop: float = 0.9,
        selection_family: Literal["cox_ph", "log_rank"] = "cox_ph",
        maxstat_method: MaxstatMethod = "lausen_1992",
        n_permutations: int = 9_999,
        random_state: int | None = 1_729,
        permutation_batch_size: int = 256,
        permutation_confidence_level: float = 0.95,
    ) -> None:
        if mode is None:
            raise ValueError(
                "mode must be explicitly set to 'exploratory' or 'inferential'."
            )
        if mode not in _ANALYSIS_MODES:
            raise ValueError(f"mode must be one of {_ANALYSIS_MODES}; got {mode!r}.")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        if use_synthetic_cutoffs and search_resolution <= 0:
            raise ValueError(
                f"search_resolution must be positive, got {search_resolution!r}"
            )
        if trunc_pct is not None or iqr_multiplier is not None:
            raise ValueError(
                "trunc_pct and iqr_multiplier belong to legacy threshold "
                "artifacts and are not accepted by scan schema 2. Use "
                "minprop/maxprop to constrain group sizes."
            )
        if selection_family not in _SELECTION_FAMILIES:
            raise ValueError(
                "selection_family must be 'cox_ph' or 'log_rank'; got "
                f"{selection_family!r}."
            )
        if maxstat_method not in {
            "lausen_1992",
            "conditional_monte_carlo",
        }:
            raise ValueError(
                "maxstat_method must be 'lausen_1992' or " "'conditional_monte_carlo'."
            )
        if not isinstance(n_permutations, int) or n_permutations < 1:
            raise ValueError("n_permutations must be a positive integer.")
        if permutation_batch_size < 1:
            raise ValueError("permutation_batch_size must be positive.")
        if not 0 < permutation_confidence_level < 1:
            raise ValueError("permutation_confidence_level must be in (0, 1).")
        if mode == "inferential" and use_synthetic_cutoffs:
            raise ValueError(
                "Inferential maxstat mode requires all eligible observed "
                "partitions; synthetic cutoff grids are exploratory only."
            )
        if mode == "inferential" and not 0 < minprop < maxprop < 1:
            raise ValueError(
                "Inferential maxstat mode requires " "0 < minprop < maxprop < 1."
            )
        if mode == "inferential" and selection_method != "none":
            raise ValueError(
                "selection_method applies only to exploratory row-wise "
                "results. Inferential selection is driven by log-rank maxstat."
            )
        if (
            mode == "inferential"
            and maxstat_method == "conditional_monte_carlo"
            and random_state is None
        ):
            raise ValueError(
                "random_state must be an integer for reproducible Monte Carlo "
                "artifacts."
            )
        if multiplicity_method is not None:
            if correction_methods is not None or selection_method != "none":
                raise ValueError(
                    "multiplicity_method cannot be combined with "
                    "correction_methods or a non-default selection_method."
                )
            warnings.warn(
                "multiplicity_method is deprecated; use correction_methods "
                "and selection_method instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            correction_methods = (multiplicity_method,)
            selection_method = multiplicity_method

        normalized_methods = normalize_correction_methods(
            correction_methods,
            selection_method=selection_method,
        )
        if selection_method not in normalized_methods:
            raise ValueError(
                f"selection_method={selection_method!r} is not available in "
                f"{normalized_methods}."
            )
        if mode == "inferential" and selection_method != "none":
            raise ValueError(
                "Legacy multiplicity_method cannot select an inferential "
                "cutoff. Use row-wise corrections only as sensitivity outputs."
            )

        self.alpha = alpha
        self.mode = mode
        self.minprop = float(minprop)
        self.maxprop = float(maxprop)
        self.trunc_pct = trunc_pct
        self.surv_label = surv_label
        self.show_progress = show_progress
        self.iqr_multiplier = iqr_multiplier
        self.search_resolution = search_resolution
        self.use_synthetic_cutoffs = use_synthetic_cutoffs
        self.correction_methods = normalized_methods
        self.selection_method = selection_method
        self.selection_family = (
            "log_rank" if mode == "inferential" else selection_family
        )
        self.maxstat_method = maxstat_method
        self.n_permutations = n_permutations
        self.random_state = random_state
        self.permutation_batch_size = permutation_batch_size
        self.permutation_confidence_level = permutation_confidence_level
        self.maxstat_analysis: MaximallySelectedLogRank | None = None
        self.maxstat_result: MaxstatResult | None = None
        # Compatibility attribute for callers that inspected the former
        # single-method configuration. New code should use selection_method.
        self.multiplicity_method = selection_method
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

        self.threshold_scan = ThresholdScan(
            target,
            minprop=self.minprop,
            maxprop=self.maxprop,
            use_synthetic_cutoffs=self.use_synthetic_cutoffs,
            search_resolution=self.search_resolution,
        )
        self.scan_config = self._build_scan_config()
        canonical_config = json.dumps(
            self.scan_config,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.scan_fingerprint = hashlib.sha256(
            canonical_config.encode("utf-8")
        ).hexdigest()

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

    def _build_scan_config(self) -> dict[str, object]:
        """Return every setting that can change scan output or interpretation."""

        analysis_frame = self.surv_df_mpv[
            ["time", "event", self.target_col_stats["name"]]
        ]
        row_hashes = pd.util.hash_pandas_object(
            analysis_frame,
            index=True,
        ).to_numpy(dtype=np.uint64)
        input_fingerprint = hashlib.sha256(row_hashes.tobytes()).hexdigest()
        return {
            "schema_version": _SCAN_SCHEMA_VERSION,
            "mode": self.mode,
            "survival_label": self.surv_label,
            "predictor": self.target_col_stats["name"],
            "n_observations": int(self.surv_df_mpv.shape[0]),
            "input_fingerprint": input_fingerprint,
            "minprop": self.minprop,
            "maxprop": self.maxprop,
            "use_synthetic_cutoffs": self.use_synthetic_cutoffs,
            "search_resolution": self.search_resolution,
            "alpha": self.alpha,
            "correction_methods": list(self.correction_methods),
            "selection_method": self.selection_method,
            "selection_family": self.selection_family,
            "maxstat_method": self.maxstat_method,
            "n_permutations": self.n_permutations,
            "random_state": self.random_state,
            "permutation_batch_size": self.permutation_batch_size,
            "permutation_confidence_level": (self.permutation_confidence_level),
            "legacy_trunc_pct": self.trunc_pct,
            "legacy_iqr_multiplier": self.iqr_multiplier,
        }

    def _create_groups(self, root_group):
        self.groups = {}
        self.groups["root"] = root_group
        clean_col_name = clean_path_name(path=self.target_col_stats["name"])
        self.groups["col_general"] = self.groups["root"].root_group.require_group(
            str(clean_col_name)
        )
        self.groups["scan"] = self.groups["col_general"].require_group(
            f"scan_v{_SCAN_SCHEMA_VERSION}_{self.scan_fingerprint[:16]}"
        )
        clean_surv_label = clean_path_name(path=self.surv_label)
        self.groups["col"] = self.groups["scan"].require_group(str(clean_surv_label))

        col_group = self.groups["col"]
        col_meta = dict(col_group.attrs.get("meta", {}))
        col_meta.setdefault("mpv_data_exists", False)
        col_meta["correction_methods"] = list(self.correction_methods)
        col_meta["selection_method"] = self.selection_method
        col_meta["selection_family"] = self.selection_family
        col_meta["multiplicity_method"] = self.selection_method
        col_meta["scan_schema_version"] = _SCAN_SCHEMA_VERSION
        col_meta["scan_fingerprint"] = self.scan_fingerprint
        col_meta["scan_config"] = self.scan_config
        col_meta["analysis_mode"] = self.mode
        col_meta["maxstat_method"] = (
            self.maxstat_method if self.mode == "inferential" else None
        )
        col_group.attrs["meta"] = col_meta

    def _create_paths(self):
        base_path = BaseZARR.get_abs_path(group=self.groups["col"])

        self.paths = {}
        self.paths["base"] = base_path

        self.paths["mpv_df"] = base_path / "mpv_df.parquet"
        self.paths["marked_thresholds_df"] = base_path / "marked_thresholds_df.parquet"
        self.paths["scan_config"] = base_path / "scan_config.json"
        self.paths["maxstat_result"] = base_path / "maxstat_result.json"
        self.paths["maxstat_process"] = base_path / "maxstat_process.parquet"

        self.paths["plot_dashboard"] = base_path / "plot_dashboard.png"
        self.paths["plot_median_follow_up"] = base_path / "plot_median_follow_up.png"
        self.paths["plot_hr_vs_p_value_scatter"] = (
            base_path / "plot_hr_vs_p_value_scatter.png"
        )
        self.paths["plot_p_values_by_correction"] = {
            correction: base_path / f"plot_p_values_{correction}.png"
            for correction in self.correction_methods
        }
        self.paths["plot_p_values_all_corrections"] = (
            base_path / "plot_p_values_all_corrections.png"
        )
        self.paths["plot_maxstat_process"] = base_path / "plot_maxstat_process.png"

    def _get_thresholds(self) -> np.ndarray:
        logger.info(
            "[Thresholds] %s unique patient partitions within "
            "minprop=%s and maxprop=%s.",
            len(self.threshold_scan.candidates),
            self.minprop,
            self.maxprop,
        )
        return self.threshold_scan.thresholds.copy()

    def _get_mpv_data_at_threshold(self, threshold: float) -> dict:
        threshold_label = f"{threshold:.12g}"
        low_label = f"<= {threshold_label}"
        high_label = f"> {threshold_label}"

        mpv_dict = {
            "threshold": threshold,
            "group0_label": low_label,
            "group1_label": high_label,
            "valid_split": False,
            "cox_ph_valid_split": False,
            "log_rank_valid_split": False,
            "cox_ph_invalid_reason": None,
            "log_rank_invalid_reason": None,
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
        cox_result = tests_dict["cox_ph"]
        log_rank_result = tests_dict["log_rank"]
        cox_valid = bool(cox_result.get("split_valid", False)) and np.isfinite(
            pd.to_numeric(
                pd.Series([cox_result.get("p_value")]),
                errors="coerce",
            ).iloc[0]
        )
        log_rank_valid = bool(log_rank_result.get("valid", False)) and np.isfinite(
            pd.to_numeric(
                pd.Series([log_rank_result.get("p_value")]),
                errors="coerce",
            ).iloc[0]
        )
        mpv_dict["valid_split"] = cox_valid
        mpv_dict["cox_ph_valid_split"] = cox_valid
        mpv_dict["log_rank_valid_split"] = log_rank_valid
        mpv_dict["cox_ph_invalid_reason"] = cox_result.get("split_invalid_reason")
        mpv_dict["log_rank_invalid_reason"] = log_rank_result.get("invalid_reason")
        mpv_dict["invalid_reason"] = mpv_dict["cox_ph_invalid_reason"]
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
                    "group0_label": f"<= {threshold:.12g}",
                    "group1_label": f"> {threshold:.12g}",
                    "valid_split": False,
                    "cox_ph_valid_split": False,
                    "log_rank_valid_split": False,
                    "cox_ph_invalid_reason": "analysis_error",
                    "log_rank_invalid_reason": "analysis_error",
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

    def _resolve_correction(self, correction: str | None) -> str:
        # Omitted plot/runtime views are always raw. ``selection_method`` is
        # used only when a caller or artifact-generation path names it.
        resolved = "none" if correction is None else correction
        if resolved not in CORRECTION_REGISTRY:
            raise ValueError(
                f"Unknown correction method {resolved!r}. Supported methods: "
                f"{list(CORRECTION_REGISTRY)}"
            )
        if resolved not in self.correction_methods:
            raise ValueError(
                f"Correction method {resolved!r} was not configured for this "
                f"MPV run. Available methods: {list(self.correction_methods)}"
            )
        return resolved

    def _p_value_column(self, *, family: str, correction: str) -> str:
        raw_column = f"{family}.p_value"
        column = raw_column if correction == "none" else f"{raw_column}_{correction}"
        if column not in self.mpv_df.columns:
            raise RuntimeError(
                f"MPV artifact does not contain {column!r}. Existing artifacts "
                "are immutable; regenerate with replace=True to calculate the "
                "requested correction."
            )
        return column

    def _eligible_p_values(self, *, family: str, column: str) -> np.ndarray:
        """Return numeric p-values with every ineligible row masked to NaN."""

        values = pd.to_numeric(
            self.mpv_df[column],
            errors="coerce",
        ).to_numpy(dtype=float, na_value=np.nan)
        eligibility_column = f"{family}.multiplicity.eligible"
        if eligibility_column in self.mpv_df.columns:
            eligibility = (
                self.mpv_df[eligibility_column].fillna(False).to_numpy(dtype=bool)
            )
            values = np.where(eligibility, values, np.nan)
        return values

    def _refresh_marker_cache(self) -> None:
        selection_family = getattr(self, "selection_family", "cox_ph")
        self.marked_thresholds_by_family = {
            family: {
                correction: self._build_marked_threshold_dicts(
                    correction=correction,
                    family=family,
                )
                for correction in self.correction_methods
            }
            for family in _SELECTION_FAMILIES
        }
        # Compatibility view for callers that used the old Cox-only cache.
        self.marked_thresholds_by_correction = self.marked_thresholds_by_family[
            selection_family
        ]
        self.marked_threshold_dicts = self.marked_thresholds_by_correction[
            self.selection_method
        ]

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
            if self.mode == "inferential":
                self._create_maxstat_result(replace=False)
            self._refresh_marker_cache()
            return self.mpv_df

        col_meta["mpv_data_exists"] = False
        col_meta["status"] = "pending"
        col_group.attrs["meta"] = col_meta

        self._create_mpv_df(replace=replace)

        if self.mode == "inferential":
            try:
                self._create_maxstat_result(replace=replace)
            except Exception as exc:
                col_meta["status"] = "failed"
                col_meta["failure_reason"] = "maxstat_inference_error"
                col_meta["error_type"] = type(exc).__name__
                col_meta["error_message"] = str(exc)
                col_group.attrs["meta"] = col_meta
                raise

        available_families = [
            family
            for family in ("cox_ph", "log_rank")
            if f"{family}.p_value" in self.mpv_df.columns
            and np.isfinite(
                pd.to_numeric(
                    self.mpv_df[f"{family}.p_value"],
                    errors="coerce",
                ).to_numpy(dtype=float, na_value=np.nan)
            ).any()
        ]
        if not available_families and self.maxstat_result is None:
            message = (
                f"No valid threshold tests for {self.surv_label}:"
                f"{self.target_col_stats['name']}."
            )
            logger.warning(message)
            col_meta["status"] = "failed"
            col_meta["failure_reason"] = "no_valid_threshold_tests"
            col_group.attrs["meta"] = col_meta
            return self.mpv_df

        self._refresh_marker_cache()
        self._save_marked_thresholds_data(replace=replace)
        self._create_plots()

        col_meta["mpv_data_exists"] = True
        col_meta["status"] = "completed"
        col_meta["failure_reason"] = None
        col_meta["available_descriptive_families"] = available_families
        col_meta["maxstat_result"] = (
            self.maxstat_result.to_dict() if self.maxstat_result is not None else None
        )
        col_group.attrs["meta"] = col_meta
        return self.mpv_df

    def _write_scan_config(self, *, replace: bool) -> None:
        config_path = self.paths["scan_config"]
        payload = {
            "scan_fingerprint": self.scan_fingerprint,
            "scan_config": self.scan_config,
        }
        if config_path.exists() and not replace:
            stored = json.loads(config_path.read_text(encoding="utf-8"))
            if stored != payload:
                raise RuntimeError(
                    "Stored scan provenance does not match the requested "
                    "configuration. Create a new fingerprinted artifact."
                )
            return
        config_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _create_maxstat_result(self, *, replace: bool) -> MaxstatResult:
        """Create or load the primary scan-level inferential result."""

        result_path = self.paths["maxstat_result"]
        predictor = self.surv_df_mpv[self.target_col_stats["name"]]
        self.maxstat_analysis = MaximallySelectedLogRank(
            predictor=predictor,
            time=self.surv_df_mpv["time"],
            event=self.surv_df_mpv["event"],
            minprop=self.minprop,
            maxprop=self.maxprop,
        )

        if result_path.exists() and not replace:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            expected_identity = {
                "schema_version": _SCAN_SCHEMA_VERSION,
                "p_value_method": self.maxstat_method,
                "minprop": self.minprop,
                "maxprop": self.maxprop,
                "n_observations": int(self.surv_df_mpv.shape[0]),
            }
            actual_identity = {key: payload.get(key) for key in expected_identity}
            if actual_identity != expected_identity:
                raise RuntimeError(
                    "Stored maxstat result does not match the requested "
                    "scan identity. Create a new fingerprinted artifact."
                )
            payload["assumptions"] = tuple(payload["assumptions"])
            self.maxstat_result = MaxstatResult(**payload)
            return self.maxstat_result

        self.maxstat_result = self.maxstat_analysis.fit(
            method=self.maxstat_method,
            n_permutations=self.n_permutations,
            random_state=self.random_state,
            batch_size=self.permutation_batch_size,
            confidence_level=self.permutation_confidence_level,
        )
        result_payload = self.maxstat_result.to_dict()
        result_path.write_text(
            json.dumps(result_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        pd.DataFrame([result_payload]).to_parquet(
            self.paths["maxstat_result"].with_suffix(".parquet"),
            index=False,
        )

        process = self.maxstat_analysis.process_df
        process["scan_fingerprint"] = self.scan_fingerprint
        process["p_value_method"] = self.maxstat_method
        process.to_parquet(self.paths["maxstat_process"], index=False)
        process.to_csv(
            self.paths["maxstat_process"].with_suffix(".csv"),
            index=False,
        )
        return self.maxstat_result

    @staticmethod
    def load_legacy_artifact(path) -> pd.DataFrame:
        """Load a pre-schema-2 MPV table without rewriting it."""

        return pd.read_parquet(path)

    def _create_plots(self) -> None:
        """Create the standard, correction-specific, and comparison figures."""

        has_p_values = any(
            f"{family}.p_value" in self.mpv_df.columns for family in _SELECTION_FAMILIES
        )
        if has_p_values:
            for correction in self.correction_methods:
                figure = self.plot_p_values(
                    correction=correction,
                    save_path=(self.paths["plot_p_values_by_correction"][correction]),
                )
                plt.close(figure)

            figure = self.plot_p_values_all_corrections(
                save_path=self.paths["plot_p_values_all_corrections"]
            )
            plt.close(figure)

        cox_columns = {
            "cox_ph.p_value",
            "cox_ph.hr.raw.hr",
            "cox_ph.hr.raw.ci_lower",
            "cox_ph.hr.raw.ci_upper",
        }
        log_rank_columns = {
            "log_rank.p_value",
            "log_rank.group0_n",
            "log_rank.group1_n",
            "log_rank.group0_median_survival.raw.median",
            "log_rank.group0_median_survival.raw.ci_lower",
            "log_rank.group0_median_survival.raw.ci_upper",
            "log_rank.group1_median_survival.raw.median",
            "log_rank.group1_median_survival.raw.ci_lower",
            "log_rank.group1_median_survival.raw.ci_upper",
            "log_rank.group0_median_follow_up.raw.median",
            "log_rank.group0_median_follow_up.raw.ci_lower",
            "log_rank.group0_median_follow_up.raw.ci_upper",
            "log_rank.group1_median_follow_up.raw.median",
            "log_rank.group1_median_follow_up.raw.ci_lower",
            "log_rank.group1_median_follow_up.raw.ci_upper",
            "split_ratio",
        }
        if (cox_columns | log_rank_columns) <= set(self.mpv_df.columns):
            figure = self.plot_dashboard(
                correction=self.selection_method,
                save_path=self.paths["plot_dashboard"],
            )
            plt.close(figure)

        if log_rank_columns <= set(self.mpv_df.columns):
            figure = self.plot_median_follow_up(
                correction=self.selection_method,
                save_path=self.paths["plot_median_follow_up"],
            )
            plt.close(figure)

        if cox_columns <= set(self.mpv_df.columns):
            figure = self.plot_hr_vs_pvalue_scatter(
                correction="none",
                save_path=self.paths["plot_hr_vs_p_value_scatter"],
            )
            plt.close(figure)

        if self.maxstat_result is not None:
            figure = self.plot_maxstat_process(
                save_path=self.paths["plot_maxstat_process"]
            )
            plt.close(figure)

    def _create_mpv_df(self, replace: bool) -> pd.DataFrame:

        self._write_scan_config(replace=replace)

        if self.paths["mpv_df"].exists() and not replace:
            logger.info(
                "mpv_df already exists at %s; set replace=True to create " "a new one.",
                self.paths["mpv_df"],
            )
            self.mpv_df = pd.read_parquet(self.paths["mpv_df"])
            self._refresh_marker_cache()
            return self.mpv_df

        thresholds = self._get_thresholds()
        iterator = tqdm(thresholds) if self.show_progress else thresholds
        mpv_dicts: list[dict] = []
        for threshold in iterator:
            mpv_data = self._evaluate_threshold(threshold=float(threshold))
            mpv_dicts.append(mpv_data["mpv_dict"])

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

        candidate_metadata = self.threshold_scan.to_frame()
        if not mpv_df.empty:
            metadata_columns = [
                "threshold",
                "partition_index",
                "lower_n",
                "upper_n",
                "lower_proportion",
                "upper_proportion",
                "source",
            ]
            mpv_df = mpv_df.merge(
                candidate_metadata[metadata_columns],
                on="threshold",
                how="left",
                validate="one_to_one",
            )

        mpv_df["analysis.schema_version"] = _SCAN_SCHEMA_VERSION
        mpv_df["analysis.mode"] = self.mode
        mpv_df["analysis.scan_fingerprint"] = self.scan_fingerprint
        mpv_df["analysis.cutoff_search_status"] = (
            "exploratory_post_selection"
            if self.mode == "exploratory"
            else "inferential_maxstat_primary"
        )
        mpv_df["cox_ph.inference_scope"] = "descriptive_post_selection"
        mpv_df["log_rank.inference_scope"] = (
            "descriptive_per_cutoff; global_p_in_maxstat_result"
            if self.mode == "inferential"
            else "exploratory_per_cutoff"
        )

        self._add_multiplicity_columns(mpv_df=mpv_df)
        mpv_df.to_parquet(self.paths["mpv_df"], index=False)
        mpv_df.to_csv(self.paths["mpv_df"].with_suffix(suffix=".csv"), index=False)

        self.mpv_df = mpv_df
        self._refresh_marker_cache()

        return self.mpv_df

    def _add_multiplicity_columns(self, *, mpv_df: pd.DataFrame) -> None:
        """Add separate multiplicity results and family sizes to ``mpv_df``.

        Cox-PH and log-rank p-values are separate correction families. Their
        reported family sizes must therefore be computed independently using
        the same finite-value rule as the correction adapter.

        This method deliberately does not rewrite historical artifacts. New
        MPV artifacts contain the per-family fields; existing artifacts loaded
        with ``replace=False`` retain their original schema.
        """

        p_value_families = (
            (
                "cox_ph.p_value",
                "cox_ph.multiplicity.n_tests",
                "cox_ph_valid_split",
                "cox_ph.multiplicity.eligible",
            ),
            (
                "log_rank.p_value",
                "log_rank.multiplicity.n_tests",
                "log_rank_valid_split",
                "log_rank.multiplicity.eligible",
            ),
        )

        for (
            p_value_column,
            count_column,
            validity_column,
            eligibility_column,
        ) in p_value_families:
            if p_value_column not in mpv_df.columns:
                mpv_df[count_column] = 0
                mpv_df[eligibility_column] = False
                continue

            values = pd.to_numeric(mpv_df[p_value_column], errors="coerce")
            raw_values = values.to_numpy(dtype=float, na_value=np.nan)
            eligible = np.isfinite(raw_values)
            if validity_column in mpv_df.columns:
                eligible &= mpv_df[validity_column].fillna(False).to_numpy(dtype=bool)
            mpv_df[eligibility_column] = eligible
            mpv_df[count_column] = int(eligible.sum())
            eligible_values = np.where(eligible, raw_values, np.nan)

            for correction in self.correction_methods:
                if correction == "none":
                    continue
                adjusted = adjust_p_values_with_missing(
                    eligible_values,
                    method=correction,
                )
                mpv_df[f"{p_value_column}_{correction}"] = adjusted

        methods_label = "|".join(self.correction_methods)
        mpv_df["multiplicity.methods"] = methods_label
        mpv_df["multiplicity.selection_method"] = self.selection_method

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

            threshold = self.mpv_df.loc[idx, "threshold"]
            mpv_data = self._evaluate_threshold(threshold=float(threshold))
            bcs = mpv_data["binary_class_surv_object"]
            if bcs is None:
                logger.warning(
                    "Marked threshold %s is not analyzable; skipping its KM "
                    "curve and detailed test artifact.",
                    threshold,
                )
                continue

            marker_name = clean_path_name(
                path=(
                    f"km_curve_{threshold_dict['label']}_" f"threshold_{threshold}.png"
                )
            )
            save_path = self.paths["base"] / marker_name
            try:
                bcs.plot_km_curves(plot=False, save_path=save_path)
            except Exception as exc:
                logger.warning(
                    "KM plot for marked threshold %s failed with %s: %s",
                    threshold,
                    type(exc).__name__,
                    exc,
                )

            tests_dict = bcs.get_tests_dict()
            tests_dict = pd.json_normalize(tests_dict).to_dict(orient="records")[0]
            tests_dict["threshold"] = threshold
            tests_dict["marker_label"] = threshold_dict["label"]
            tests_dict["selection_family"] = threshold_dict.get("family")
            tests_dict["analysis_mode"] = self.mode
            tests_dict["post_selection_warning"] = (
                "Cutoff-specific Cox estimates and confidence intervals are "
                "descriptive after data-driven cutoff selection."
            )
            if self.maxstat_result is not None:
                tests_dict["maxstat.global_p_value"] = self.maxstat_result.p_value
                tests_dict["maxstat.p_value_method"] = (
                    self.maxstat_result.p_value_method
                )

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

    def _build_marked_threshold_dicts(
        self,
        *,
        correction: str | None = "none",
        family: Literal["cox_ph", "log_rank"] = "cox_ph",
    ) -> list[dict]:
        """Compute explicitly family-specific descriptive/inferential markers."""

        correction = self._resolve_correction(correction)
        if family not in _SELECTION_FAMILIES:
            raise ValueError(
                f"family must be one of {_SELECTION_FAMILIES}; got {family!r}."
            )
        mpv_df = self.mpv_df
        target_median = self.target_col_stats["median"]
        family_label = "Cox-Wald" if family == "cox_ph" else "log-rank"

        if mpv_df.empty or "threshold" not in mpv_df.columns:
            return [
                {
                    "label": "Median partition",
                    "idx": None,
                    "color": "blue",
                    "ls": "--",
                    "family": family,
                },
                {
                    "label": f"Min {family_label} p (descriptive)",
                    "idx": None,
                    "color": "gray",
                    "ls": "--",
                    "family": family,
                },
            ]

        median_idx = (mpv_df["threshold"] - target_median).abs().idxmin()

        mode = getattr(self, "mode", "exploratory")
        maxstat_result = getattr(self, "maxstat_result", None)
        if mode == "inferential" and family == "log_rank" and maxstat_result:
            distance = (mpv_df["threshold"] - maxstat_result.optimal_threshold).abs()
            maxstat_idx = distance.idxmin()
            return [
                {
                    "label": "Median partition",
                    "idx": median_idx,
                    "color": "blue",
                    "ls": "--",
                    "family": family,
                },
                {
                    "label": "Maxstat optimum (log-rank)",
                    "idx": maxstat_idx,
                    "color": "black",
                    "ls": "--",
                    "family": family,
                },
            ]

        if f"{family}.p_value" not in mpv_df.columns:
            return [
                {
                    "label": "Median partition",
                    "idx": median_idx,
                    "color": "blue",
                    "ls": "--",
                    "family": family,
                }
            ]

        p_value_col = self._p_value_column(
            family=family,
            correction=correction,
        )

        valid_rows = mpv_df
        validity_column = f"{family}_valid_split"
        if validity_column in valid_rows.columns:
            valid_rows = valid_rows[valid_rows[validity_column].fillna(False)]
        if p_value_col in valid_rows.columns:
            valid_rows = valid_rows[
                np.isfinite(
                    pd.to_numeric(
                        valid_rows[p_value_col],
                        errors="coerce",
                    )
                )
            ]

        if not valid_rows.empty and p_value_col in valid_rows.columns:
            min_p_val_idx = valid_rows[p_value_col].idxmin()
            sig = valid_rows[valid_rows[p_value_col] < self.alpha]
            closest_idx = (
                None if sig.empty else (sig["threshold"] - target_median).abs().idxmin()
            )
        else:
            closest_idx = None
            min_p_val_idx = None
            logger.warning(
                "No valid splits for %s; skipping its p-value markers.",
                family_label,
            )

        return [
            {
                "label": "Median partition",
                "idx": median_idx,
                "color": "blue",
                "ls": "--",
                "family": family,
            },
            {
                "label": f"Min {family_label} p (descriptive)",
                "idx": min_p_val_idx,
                "color": "gray",
                "ls": "--",
                "family": family,
            },
            {
                "label": f"Closest significant {family_label} (descriptive)",
                "idx": closest_idx,
                "color": "green",
                "ls": "--",
                "family": family,
            },
        ]

    def _add_threshold_markers(
        self,
        ax,
        *,
        correction: str | None = None,
        family: Literal["cox_ph", "log_rank"] | None = None,
    ) -> None:
        """Draw correction- and family-specific reference markers."""

        correction = self._resolve_correction(correction)
        if family is None:
            family = getattr(self, "selection_family", "cox_ph")
        marker_cache = getattr(self, "marked_thresholds_by_family", {})
        threshold_dicts = marker_cache.get(family, {}).get(correction)
        if threshold_dicts is None:
            threshold_dicts = self._build_marked_threshold_dicts(
                correction=correction,
                family=family,
            )
        for threshold_dict in threshold_dicts:
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

    def _annotate_analysis_status(
        self,
        ax,
        *,
        family: Literal["cox_ph", "log_rank"],
    ) -> None:
        """Place a visible inference-scope label on an output axis."""

        if getattr(self, "mode", "exploratory") == "exploratory":
            label = (
                "EXPLORATORY / POST-SELECTION: per-cutoff p-values do not "
                "provide a maxstat global test"
            )
        elif family == "cox_ph":
            label = (
                "POST-SELECTION DESCRIPTIVE: Cox HR/CI and Wald p-values are "
                "not maxstat-adjusted"
            )
        elif self.maxstat_result is None:
            label = "INFERENTIAL MODE: maxstat global result not yet available"
        else:
            label = (
                f"PRIMARY GLOBAL TEST: {self.maxstat_result.p_value_method}; "
                f"p={self.maxstat_result.p_value:.4g}"
            )
        ax.text(
            0.99,
            0.01,
            label,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="dimgray",
            bbox={
                "facecolor": "white",
                "edgecolor": "lightgray",
                "alpha": 0.85,
                "pad": 2,
            },
            zorder=10,
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
        correction: str | None = None,
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
            markersize=_LINE_MARKER_SIZE,
            label="Cox HR (post-selection descriptive)",
        )
        ax.fill_between(
            x,
            self.mpv_df["cox_ph.hr.raw.ci_lower"],
            self.mpv_df["cox_ph.hr.raw.ci_upper"],
            color="tab:purple",
            alpha=0.15,
            label="Naive 95% CI",
        )
        ax.axhline(1.0, color="black", lw=0.8, alpha=0.5, label="HR = 1 (no effect)")

        marker_family = "log_rank" if self.mode == "inferential" else "cox_ph"
        self._add_threshold_markers(
            ax,
            correction=correction,
            family=marker_family,
        )
        self._annotate_analysis_status(ax, family="cox_ph")

        ax.set_xlabel("Threshold")
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("Hazard Ratio (log scale)")
        else:
            ax.set_ylabel("Hazard Ratio")

        if title is None:
            ax.set_title(
                "Post-selection Cox proportional-hazards ratio across "
                f"scanned thresholds\n{self.surv_label}:"
                f"{self.target_col_stats['name']}"
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
        correction: str | None = None,
    ):
        """Plot raw or one corrected p-value view across thresholds.

        A correction-specific view retains the raw values as a thin reference
        and overlays the selected adjusted values. ``correction="none"`` plots
        raw p-values only.
        """
        self._require_mpv_df()
        correction = self._resolve_correction(correction)
        if p_value not in {"cox_ph", "log_rank", "both"}:
            raise ValueError('p_value must be one of "cox_ph", "log_rank", or "both"')

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=figsize)

        x = self.mpv_df["threshold"]
        families = {
            "cox_ph": ("Cox PH", "tab:blue", "navy"),
            "log_rank": ("Log-rank", "tab:orange", "darkorange"),
        }
        selected_families = tuple(families) if p_value == "both" else (p_value,)

        for family in selected_families:
            family_label, raw_color, adjusted_color = families[family]
            raw_column = f"{family}.p_value"
            if raw_column not in self.mpv_df.columns:
                logger.warning(
                    "No valid splits for %s; skipping its p-value plot.",
                    family_label,
                )
                continue

            raw_values = self._eligible_p_values(
                family=family,
                column=raw_column,
            )
            if log_scale:
                raw_values = np.clip(raw_values, 1e-300, None)
            raw_label = (
                f"{family_label} raw"
                if correction != "none"
                else f"{family_label} p-value"
            )
            ax.plot(
                x,
                raw_values,
                color=raw_color,
                lw=1.2,
                marker=".",
                markersize=_LINE_MARKER_SIZE,
                alpha=0.7 if correction != "none" else 1.0,
                label=raw_label,
            )

            if correction != "none":
                adjusted_column = self._p_value_column(
                    family=family,
                    correction=correction,
                )
                adjusted_values = self._eligible_p_values(
                    family=family,
                    column=adjusted_column,
                )
                if log_scale:
                    adjusted_values = np.clip(adjusted_values, 1e-300, None)
                ax.plot(
                    x,
                    adjusted_values,
                    color=adjusted_color,
                    lw=1.7,
                    ls="--",
                    label=f"{family_label} ({correction}-adjusted)",
                )

        family_title = {
            "cox_ph": "Cox PH",
            "log_rank": "Log-rank",
            "both": "Cox PH vs. log-rank",
        }[p_value]
        view_label = "raw (no correction)" if correction == "none" else correction
        default_title = (
            f"Per-cutoff {family_title} p-values (post-selection descriptive) "
            f"[{view_label}]\n{self.surv_label}:"
            f"{self.target_col_stats['name']}"
        )

        ax.axhline(
            self.alpha, color="tab:red", ls="--", lw=1, label=f"alpha = {self.alpha}"
        )

        marker_family = (
            p_value
            if p_value in _SELECTION_FAMILIES
            else getattr(self, "selection_family", "cox_ph")
        )
        self._add_threshold_markers(
            ax,
            correction=correction,
            family=marker_family,
        )
        self._annotate_analysis_status(ax, family=marker_family)

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

    def plot_p_values_all_corrections(
        self,
        *,
        title=None,
        grid=True,
        figsize=(12, 9),
        log_scale=True,
        save_path=None,
        dpi=300,
    ):
        """Plot every configured correction in separate test-family panels."""

        self._require_mpv_df()
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        x = self.mpv_df["threshold"]
        colors = plt.get_cmap("tab10")
        family_specs = (
            ("cox_ph", "Cox PH", axes[0]),
            ("log_rank", "Log-rank", axes[1]),
        )

        for family, family_label, ax in family_specs:
            raw_column = f"{family}.p_value"
            if raw_column not in self.mpv_df.columns:
                logger.warning(
                    "No valid splits for %s; skipping its comparison panel.",
                    family_label,
                )
            else:
                for index, correction in enumerate(self.correction_methods):
                    column = self._p_value_column(
                        family=family,
                        correction=correction,
                    )
                    values = self._eligible_p_values(
                        family=family,
                        column=column,
                    )
                    if log_scale:
                        values = np.clip(values, 1e-300, None)
                    label = "Raw (none)" if correction == "none" else correction
                    ax.plot(
                        x,
                        values,
                        color=colors(index % 10),
                        lw=1.5,
                        marker="." if correction == "none" else None,
                        markersize=(
                            _LINE_MARKER_SIZE if correction == "none" else None
                        ),
                        ls="-" if correction == "none" else "--",
                        label=label,
                    )

            ax.axhline(
                self.alpha,
                color="tab:red",
                ls=":",
                lw=1,
                label=f"alpha = {self.alpha}",
            )
            self._add_threshold_markers(
                ax,
                correction=self.selection_method,
                family=family,
            )
            self._annotate_analysis_status(ax, family=family)
            ax.set_title(f"{family_label} p-value family")
            ax.set_ylabel("p-value (log scale)" if log_scale else "p-value")
            if log_scale:
                ax.set_yscale("log")
            if grid:
                ax.grid(True, which="both", alpha=0.2)
            ax.legend(loc="upper right", fontsize=8, ncols=2)

        axes[-1].set_xlabel("Threshold")
        selection_label = (
            "raw" if self.selection_method == "none" else self.selection_method
        )
        fig.suptitle(
            (
                "Row-wise correction sensitivity analyses "
                "(post-selection descriptive)\n"
                f"{self.surv_label}:{self.target_col_stats['name']} "
                f"(threshold markers: {selection_label})"
                if title is None
                else title
            ),
            fontsize=13,
        )
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return fig

    def plot_maxstat_process(
        self,
        *,
        ax=None,
        title=None,
        grid=True,
        figsize=(11, 5),
        save_path=None,
        dpi=300,
    ):
        """Plot the standardized log-rank score process and global result."""

        if self.maxstat_result is None or self.maxstat_analysis is None:
            raise RuntimeError(
                "A maxstat result is available only after inferential "
                "create_mpv_data()."
            )

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=figsize)

        process = self.maxstat_analysis.process_df
        ax.plot(
            process["threshold"],
            process["standardized_statistic"],
            color="tab:orange",
            marker=".",
            markersize=_LINE_MARKER_SIZE,
            lw=1.4,
            label="Absolute standardized log-rank score",
        )
        ax.axvline(
            self.maxstat_result.optimal_threshold,
            color="black",
            ls="--",
            lw=1,
            label=(
                "Selected maxstat cutoff = "
                f"{self.maxstat_result.optimal_threshold:.6g}"
            ),
        )
        ax.scatter(
            [self.maxstat_result.optimal_threshold],
            [self.maxstat_result.statistic],
            color="black",
            s=30,
            zorder=4,
        )
        ax.set_xlabel("Canonical threshold (unique patient partition)")
        ax.set_ylabel("Absolute standardized statistic")
        default_title = (
            "Maximally selected log-rank process\n"
            f"global p={self.maxstat_result.p_value:.4g} "
            f"({self.maxstat_result.p_value_method}); "
            f"{self.surv_label}:{self.target_col_stats['name']}"
        )
        ax.set_title(default_title if title is None else title)
        if grid:
            ax.grid(True, alpha=0.2)
        ax.legend(loc="best", fontsize=8)
        self._annotate_analysis_status(ax, family="log_rank")

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
        correction: str | None = None,
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

        ax.plot(
            x,
            ci_ratio,
            color="tab:brown",
            lw=1.2,
            marker=".",
            markersize=_LINE_MARKER_SIZE,
        )

        marker_family = "log_rank" if self.mode == "inferential" else "cox_ph"
        self._add_threshold_markers(
            ax,
            correction=correction,
            family=marker_family,
        )

        ax.set_xlabel("Threshold")
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("CI Width\n(Upper / Lower, log scale)")
        else:
            ax.set_ylabel("CI Width\n(Upper / Lower)")

        if title is None:
            ax.set_title(
                "Cox HR confidence-interval width across thresholds\n"
                "(narrower = more stable estimate)\n"
                f"{self.surv_label}:{self.target_col_stats['name']}"
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
        correction: str | None = None,
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
            markersize=_LINE_MARKER_SIZE,
            label="Group 0 median survival",
        )
        ax.fill_between(x, g0_lo, g0_hi_plot, color="tab:blue", alpha=0.12)

        ax.plot(
            x,
            g1_est_plot,
            color="tab:orange",
            lw=1.4,
            marker=".",
            markersize=_LINE_MARKER_SIZE,
            label="Group 1 median survival",
        )
        ax.fill_between(x, g1_lo, g1_hi_plot, color="tab:orange", alpha=0.12)

        self._add_threshold_markers(
            ax,
            correction=correction,
            family=getattr(self, "selection_family", "cox_ph"),
        )

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Median Survival Time")

        if title is None:
            subtitle = (
                '("not reached" capped near top)'
                if cap_not_reached
                else '(gaps = "not reached")'
            )
            ax.set_title(
                "Median survival per group across scanned thresholds\n"
                f"{subtitle}\n{self.surv_label}:"
                f"{self.target_col_stats['name']}"
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
        self,
        ax=None,
        title=None,
        grid=True,
        figsize=(11, 4.5),
        save_path=None,
        dpi=300,
        correction: str | None = None,
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
            markersize=_LINE_MARKER_SIZE,
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
            markersize=_LINE_MARKER_SIZE,
            label="Group 1 median follow-up",
        )
        ax.fill_between(
            x,
            self.mpv_df["log_rank.group1_median_follow_up.raw.ci_lower"],
            self.mpv_df["log_rank.group1_median_follow_up.raw.ci_upper"],
            color="tab:orange",
            alpha=0.10,
        )

        self._add_threshold_markers(
            ax,
            correction=correction,
            family=getattr(self, "selection_family", "cox_ph"),
        )

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Median Follow-up Time")

        if title is None:
            ax.set_title(
                "Median follow-up per group across scanned thresholds\n"
                f"{self.surv_label}:{self.target_col_stats['name']}"
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
        self,
        ax=None,
        title=None,
        grid=True,
        figsize=(11, 4.5),
        save_path=None,
        dpi=300,
        correction: str | None = None,
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
            markersize=_LINE_MARKER_SIZE,
            label="Group 0 n",
        )
        ax.plot(
            x,
            self.mpv_df["log_rank.group1_n"],
            color="tab:orange",
            lw=1.4,
            marker=".",
            markersize=_LINE_MARKER_SIZE,
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

        self._add_threshold_markers(
            ax,
            correction=correction,
            family=getattr(self, "selection_family", "cox_ph"),
        )

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Group size (n)")

        if title is None:
            ax.set_title(
                "Absolute group sizes across scanned thresholds\n"
                f"{self.surv_label}:{self.target_col_stats['name']}"
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
        correction: str | None = None,
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

        ax.plot(
            x,
            ratio,
            color="tab:green",
            lw=1.2,
            marker=".",
            markersize=_LINE_MARKER_SIZE,
        )
        ax.axhline(1.0, color="black", lw=0.8, alpha=0.4, label="Balanced (1:1)")

        self._add_threshold_markers(
            ax,
            correction=correction,
            family=getattr(self, "selection_family", "cox_ph"),
        )

        ax.set_xlabel("Threshold")
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("Split ratio\n(group0_n / group1_n, log)")
        else:
            ax.set_ylabel("Split ratio\n(group0_n / group1_n)")

        if title is None:
            ax.set_title(
                "Group split ratio across scanned thresholds\n"
                f"{self.surv_label}:{self.target_col_stats['name']}"
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
        correction: str | None = None,
    ):
        """
        Scatter of Cox HR (x, log scale) vs. Cox p-value (y, log scale),
        colored by `color_by` (default: "threshold"; alternative:
        "split_ratio"). The three reference thresholds are drawn as open
        circles at their (HR, p-value) position rather than axvlines, since
        the x-axis here is HR, not threshold/index. The default scatter uses
        raw p-values; pass a configured correction explicitly for an adjusted
        view.
        """
        self._require_mpv_df()
        correction = self._resolve_correction(correction)

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(7, 6))
        else:
            fig = ax.figure

        if color_by not in self.mpv_df.columns:
            raise ValueError(f"color_by={color_by!r} is not a result column.")
        color_vals = self.mpv_df[color_by]
        p_value_column = self._p_value_column(
            family="cox_ph",
            correction=correction,
        )
        sc = ax.scatter(
            self.mpv_df["cox_ph.hr.raw.hr"],
            np.clip(
                self._eligible_p_values(
                    family="cox_ph",
                    column=p_value_column,
                ),
                1e-300,
                None,
            ),
            c=color_vals,
            cmap="viridis",
            s=_SCATTER_POINT_SIZE,
            edgecolor="none",
            zorder=2,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(color_by)

        marker_family = "log_rank" if self.mode == "inferential" else "cox_ph"
        threshold_dicts = self.marked_thresholds_by_family[marker_family][correction]
        for threshold_dict in threshold_dicts:
            if threshold_dict["idx"] is None:
                continue
            row = self.mpv_df.loc[threshold_dict["idx"]]
            hr_value = pd.to_numeric(
                pd.Series([row["cox_ph.hr.raw.hr"]]),
                errors="coerce",
            ).iloc[0]
            marker_p_value = pd.to_numeric(
                pd.Series([row[p_value_column]]),
                errors="coerce",
            ).iloc[0]
            if not np.isfinite(hr_value) or not np.isfinite(marker_p_value):
                continue
            ax.scatter(
                hr_value,
                max(marker_p_value, 1e-300),
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
        p_value_label = "raw" if correction == "none" else f"{correction}-adjusted"
        ax.set_ylabel(f"Cox p-value ({p_value_label}, log scale)")
        self._annotate_analysis_status(ax, family="cox_ph")

        if title is None:
            ax.set_title(
                "Post-selection Cox HR vs. Wald p-value across thresholds, "
                f"colored by {color_by}\n{self.surv_label}:"
                f"{self.target_col_stats['name']}"
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
        self,
        imbalance_factor=10.0,
        title=None,
        grid=True,
        save_path=None,
        dpi=300,
        correction: str | None = None,
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
        # The dashboard is intentionally a raw overview. Corrected views are
        # kept in the dedicated correction-specific and comparison figures.
        self._resolve_correction(correction)
        correction = "none"

        fig, axes = plt.subplots(
            5,
            1,
            figsize=(12, 16),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 2, 1.3, 1.3, 2]},
        )
        ax_p, ax_hr, ax_n, ax_ratio, ax_surv = axes

        self.plot_p_values(
            p_value="both",
            ax=ax_p,
            grid=grid,
            correction=correction,
        )
        ax_p.set_xlabel("")

        self.plot_hr_with_ci(ax=ax_hr, grid=grid, correction=correction)
        ax_hr.set_xlabel("")
        ax_hr.set_title("Cox Hazard Ratio (log scale) with 95% CI")

        self.plot_group_sizes(ax=ax_n, grid=grid, correction=correction)
        ax_n.set_title("")
        ax_n.set_xlabel("")

        self.plot_split_ratio(
            imbalance_factor=imbalance_factor,
            ax=ax_ratio,
            grid=grid,
            correction=correction,
        )
        ax_ratio.set_title("")
        ax_ratio.set_xlabel("")

        self.plot_median_survival(
            ax=ax_surv,
            cap_not_reached=True,
            grid=grid,
            correction=correction,
        )
        ax_surv.set_title("")

        axes[-1].set_xlabel("Threshold")
        inference_suffix = (
            ""
            if self.maxstat_result is None
            else (
                f"; maxstat global p={self.maxstat_result.p_value:.4g} "
                f"({self.maxstat_result.p_value_method})"
            )
        )
        fig.suptitle(
            (
                f"Threshold scan dashboard [raw descriptive{inference_suffix}]\n"
                f"{self.surv_label}:{self.target_col_stats['name']}"
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
