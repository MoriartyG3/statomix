from __future__ import annotations

import pandas as pd

from statomix.pipelines.artifacts import frame_from_rows

CATEGORICAL_SUMMARY_SCHEMA = {
    "count": "int64",
    "percentage": "float64",
}

NUMERICAL_SUMMARY_SCHEMA = {
    "n": "int64",
    "unique_n": "int64",
    "missing_n": "int64",
    "missing_pct": "float64",
    "mean": "float64",
    "std": "float64",
    "q1": "float64",
    "median": "float64",
    "q3": "float64",
    "iqr": "float64",
    "p01": "float64",
    "p05": "float64",
    "p95": "float64",
    "p99": "float64",
    "min": "float64",
    "max": "float64",
}

NORMALITY_DIAGNOSTIC_SCHEMA = {
    "test_type": "object",
    "stat": "float64",
    "p": "float64",
    "normal": "boolean",
    "recommended_sample_size": "object",
    "p_value_reliable": "boolean",
    "power.n": "int64",
    "power.regime": "object",
    "power.note": "object",
    "ties.n_unique": "int64",
    "ties.n_total": "int64",
    "ties.tie_fraction": "float64",
    "ties.ks_family_reliable": "boolean",
    "outliers.method": "object",
    "outliers.threshold": "object",
    "outliers.n_outliers": "int64",
    "outliers.outlier_fraction": "float64",
    "outliers.computable": "boolean",
    "outliers.note": "object",
    "shape.skewness": "float64",
    "shape.kurtosis_excess": "float64",
    "shape.skew_interpretation": "object",
    "shape.kurtosis_interpretation": "object",
    "outliers.outlier_values": "object",
}

DEFAULT_SURVIVAL_TIME_POINTS = (12, 24, 36, 48, 60)


def _build_survival_descriptive_schema() -> dict[str, str]:
    schema = {
        "median_survival.label": "object",
        "median_survival.raw.median": "float64",
        "median_survival.raw.ci_lower": "float64",
        "median_survival.raw.ci_upper": "float64",
        "median_follow_up.label": "object",
        "median_follow_up.raw.median": "float64",
        "median_follow_up.raw.ci_lower": "float64",
        "median_follow_up.raw.ci_upper": "float64",
    }

    for time_point in DEFAULT_SURVIVAL_TIME_POINTS:
        probability_prefix = f"Surv Probability.{time_point}"
        schema.update(
            {
                f"{probability_prefix}.label": "object",
                f"{probability_prefix}.raw.survival_prob": "float64",
                f"{probability_prefix}.raw.ci_lower": "float64",
                f"{probability_prefix}.raw.ci_upper": "float64",
            }
        )

    for time_point in DEFAULT_SURVIVAL_TIME_POINTS:
        schema.update(
            {
                f"rmst.{time_point}.restricted_time": "float64",
                f"rmst.{time_point}.rmst": "float64",
            }
        )

    return schema


SURVIVAL_DESCRIPTIVE_SCHEMA = _build_survival_descriptive_schema()


def empty_indexed_frame(
    schema: dict[str, str],
    index_name: str,
) -> pd.DataFrame:
    frame = frame_from_rows(rows=[], schema=schema)
    frame.index = pd.Index([], name=index_name)
    return frame


def empty_numerical_summary() -> pd.DataFrame:
    return empty_indexed_frame(
        schema=NUMERICAL_SUMMARY_SCHEMA,
        index_name="name",
    )


def empty_categorical_summary() -> pd.DataFrame:
    frame = frame_from_rows(
        rows=[],
        schema=CATEGORICAL_SUMMARY_SCHEMA,
    )
    frame.index = pd.MultiIndex.from_arrays(
        arrays=[[], []],
        names=["col_name", "category"],
    )
    return frame


def empty_normality_diagnostics() -> pd.DataFrame:
    return empty_indexed_frame(
        schema=NORMALITY_DIAGNOSTIC_SCHEMA,
        index_name="name",
    )


def empty_survival_descriptives() -> pd.DataFrame:
    return empty_indexed_frame(
        schema=SURVIVAL_DESCRIPTIVE_SCHEMA,
        index_name="surv_label",
    )
