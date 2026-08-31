"""Dataset pipeline interface for analyzer result schemas."""

from statomix.core.analysis_results import (
    CATEGORICAL_SUMMARY_SCHEMA,
    DEFAULT_SURVIVAL_TIME_POINTS,
    NORMALITY_DIAGNOSTIC_SCHEMA,
    NUMERICAL_SUMMARY_SCHEMA,
    SURVIVAL_DESCRIPTIVE_SCHEMA,
    empty_categorical_summary,
    empty_indexed_frame,
    empty_normality_diagnostics,
    empty_numerical_summary,
    empty_survival_descriptives,
)

__all__ = [
    "CATEGORICAL_SUMMARY_SCHEMA",
    "DEFAULT_SURVIVAL_TIME_POINTS",
    "NORMALITY_DIAGNOSTIC_SCHEMA",
    "NUMERICAL_SUMMARY_SCHEMA",
    "SURVIVAL_DESCRIPTIVE_SCHEMA",
    "empty_categorical_summary",
    "empty_indexed_frame",
    "empty_normality_diagnostics",
    "empty_numerical_summary",
    "empty_survival_descriptives",
]
