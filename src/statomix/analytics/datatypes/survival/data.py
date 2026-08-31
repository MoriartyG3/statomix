"""Validation and normalization of right-censored survival data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedSurvivalData:
    """Validated survival frame and its missing-row audit count."""

    frame: pd.DataFrame
    dropped_rows: int


def coerce_event_to_bool(*, event: pd.Series) -> pd.Series:
    """Coerce only boolean or exact numeric 0/1 event indicators."""

    if pd.api.types.is_bool_dtype(event):
        return event.astype(bool)
    unique_values = set(pd.unique(event))
    if unique_values <= {0, 1}:
        return event.astype(bool)
    raise ValueError(
        "'event' column must be boolean, or numeric containing only 0/1. "
        f"Found unexpected values: {sorted(unique_values, key=str)}"
    )


def prepare_survival_data(
    *,
    frame: pd.DataFrame,
    grouping_column: str | None = None,
) -> PreparedSurvivalData:
    """Validate durations/events and drop rows missing required values."""

    required = ["time", "event"]
    if grouping_column is not None:
        required.append(grouping_column)
    missing_columns = set(required) - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Survival data is missing required columns: {sorted(missing_columns)}"
        )

    prepared = frame[required].copy()
    before = len(prepared)
    prepared = prepared.dropna(subset=required)
    dropped = before - len(prepared)
    if prepared.empty:
        raise ValueError(
            "No complete survival observations remain after dropping NaNs."
        )

    if pd.api.types.is_bool_dtype(
        prepared["time"]
    ) or not pd.api.types.is_numeric_dtype(prepared["time"]):
        raise ValueError("'time' column must be numeric.")
    time_values = prepared["time"].to_numpy(dtype=float)
    if not np.isfinite(time_values).all():
        raise ValueError("'time' column must contain finite durations.")
    if (time_values < 0).any():
        raise ValueError("'time' column contains negative durations.")

    prepared["event"] = coerce_event_to_bool(event=prepared["event"])
    return PreparedSurvivalData(
        frame=prepared.reset_index(drop=True),
        dropped_rows=dropped,
    )
