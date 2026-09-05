"""Strict, auditable row selection through curated Identifier columns."""

from __future__ import annotations

from collections import Counter
from numbers import Integral, Real

import numpy as np

from statomix.core.artifacts import canonical_json
from statomix.curation.columns import DataTypes

from .metadata import ArtifactData, refresh_profiles, validate_state
from .specifications import ExcludeRows, encode_identifier_value


def _observed_identifier(value):
    """Return a stable typed identity for a scalar read from pandas."""

    if isinstance(value, str):
        return "string", value
    if isinstance(value, (bool, np.bool_)):
        return "boolean", bool(value)
    if isinstance(value, Integral):
        return "integer", int(value)
    if isinstance(value, Real):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("Identifier columns cannot contain nonfinite values.")
        return "real", result
    raise TypeError(
        "Identifier columns must contain strings, booleans, integers, or real "
        f"numbers, not {type(value).__name__}."
    )


def _requested_identifier(value):
    record = encode_identifier_value(value)
    return record["type"], record["value"]


def _audit_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def apply_row_exclusion(
    parent: ArtifactData,
    operation: ExcludeRows,
    *,
    step: int = 1,
):
    """Return a filtered copy plus one summary and one record per removed row."""

    state = parent.copy()
    identifier = operation.identifier
    if identifier not in state.df.columns:
        raise KeyError(f"Unknown identifier column: {identifier!r}.")
    if state.profiles[identifier].col_type != DataTypes.IDENTIFIER:
        raise TypeError(
            f"Row exclusion requires an Identifier column, not {identifier!r}."
        )

    series = state.df[identifier]
    if series.isna().any():
        raise ValueError(
            f"Identifier column {identifier!r} contains missing values; "
            "row selection would not be fully auditable."
        )

    observed = [_observed_identifier(value) for value in series.tolist()]
    counts = Counter(observed)
    if len(counts) != len(observed):
        duplicates = sorted(
            {
                canonical_json({"type": kind, "value": value})
                for (kind, value), count in counts.items()
                if count > 1
            }
        )
        raise ValueError(
            f"Identifier column {identifier!r} is not unique: {duplicates!r}."
        )

    requested = {_requested_identifier(value) for value in operation.values}
    found = set(observed) & requested
    missing = requested - found
    if missing:
        labels = sorted(
            canonical_json({"type": kind, "value": value}) for kind, value in missing
        )
        raise KeyError(
            f"Requested identifiers are absent from {identifier!r}: {labels!r}."
        )

    excluded_positions = [
        position for position, identity in enumerate(observed) if identity in requested
    ]
    if len(excluded_positions) == len(state.df):
        raise ValueError("A row-exclusion operation cannot remove every row.")

    excluded_records = []
    lineage = state.lineage.reset_index(drop=True)
    for exclusion_order, input_row in enumerate(excluded_positions, start=1):
        kind, value = observed[input_row]
        parent_record = lineage.iloc[input_row]
        excluded_records.append(
            {
                "step": step,
                "exclusion_order": exclusion_order,
                "identifier_column": identifier,
                "identifier_type": kind,
                "identifier_value": canonical_json({"type": kind, "value": value}),
                "input_row": input_row,
                "dataframe_index": repr(state.df.index[input_row]),
                "parent_artifact": str(parent_record["parent_artifact"]),
                "parent_row": int(_audit_scalar(parent_record["parent_row"])),
                "source_dataset": str(parent_record["source_dataset"]),
                "reason": operation.reason,
            }
        )

    keep = np.ones(len(state.df), dtype=bool)
    keep[excluded_positions] = False
    state.df = state.df.iloc[keep].copy()
    state.lineage = state.lineage.iloc[keep].copy().reset_index(drop=True)
    state.lineage["output_row"] = range(len(state.lineage))

    types = {name: profile.col_type for name, profile in state.profiles.items()}
    refresh_profiles(state, types)
    validate_state(state)

    summary = {
        "step": step,
        "operation": "exclude_rows",
        "identifier_column": identifier,
        "requested_identifiers": len(requested),
        "input_rows": len(parent.df),
        "excluded_rows": len(excluded_positions),
        "output_rows": len(state.df),
        "reason": operation.reason,
    }
    return state, summary, excluded_records
