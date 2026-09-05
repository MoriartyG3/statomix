"""Two-parent column replacement using exact Identifier values."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_numeric_dtype,
)

from statomix.core.artifacts import canonical_json
from statomix.curation.columns import DataTypes

from .metadata import (
    ArtifactData,
    refresh_profiles,
    validate_state,
)
from .specifications import UpdateColumnsByKey


def _identifier_identity(
    value: Any,
) -> tuple[str, object]:
    """Create an exact, hashable identifier without coercing its type."""

    if isinstance(value, str):
        if not value:
            raise ValueError("Identifier values must not be empty strings.")

        return "string", value

    if isinstance(value, (bool, np.bool_)):
        return "boolean", bool(value)

    if isinstance(value, Integral):
        return "integer", int(value)

    if isinstance(value, Real):
        result = float(value)

        if not np.isfinite(result):
            raise ValueError("Identifier values must be finite.")

        return "real", result

    raise TypeError(
        "Identifier columns must contain strings, booleans, "
        f"integers, or real numbers, not "
        f"{type(value).__name__}."
    )


def _identifier_json(
    identity: tuple[str, object],
) -> str:
    kind, value = identity

    return canonical_json(
        {
            "type": kind,
            "value": value,
        }
    )


def _identifier_index(
    *,
    state: ArtifactData,
    column: str,
    artifact_role: str,
) -> tuple[
    tuple[tuple[str, object], ...],
    dict[tuple[str, object], int],
]:
    if column not in state.df.columns:
        raise KeyError(
            f"{artifact_role} artifact does not contain "
            f"Identifier column {column!r}."
        )

    if state.profiles[column].col_type != DataTypes.IDENTIFIER:
        raise TypeError(
            f"{artifact_role} key {column!r} is not curated " "as an Identifier."
        )

    series = state.df[column]

    if series.isna().any():
        raise ValueError(
            f"{artifact_role} Identifier column {column!r} " "contains missing values."
        )

    identities = tuple(_identifier_identity(value) for value in series.tolist())

    counts = Counter(identities)

    duplicates = [
        _identifier_json(identity) for identity, count in counts.items() if count > 1
    ]

    if duplicates:
        raise ValueError(
            f"{artifact_role} Identifier column {column!r} "
            f"is not unique: {sorted(duplicates)!r}."
        )

    positions = {identity: position for position, identity in enumerate(identities)}

    return identities, positions


def _audit_value(
    value: Any,
) -> str:
    """Serialize one value without losing missingness or scalar type."""

    if value is None or value is pd.NA or value is pd.NaT:
        return canonical_json(
            {
                "type": "missing",
                "value": None,
            }
        )

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return canonical_json(
                {
                    "type": "missing",
                    "value": None,
                }
            )

        return canonical_json(
            {
                "type": "datetime",
                "value": value.isoformat(),
            }
        )

    if isinstance(value, datetime):
        return canonical_json(
            {
                "type": "datetime",
                "value": value.isoformat(),
            }
        )

    if isinstance(value, date):
        return canonical_json(
            {
                "type": "date",
                "value": value.isoformat(),
            }
        )

    if isinstance(value, str):
        return canonical_json(
            {
                "type": "string",
                "value": value,
            }
        )

    if isinstance(value, bool):
        return canonical_json(
            {
                "type": "boolean",
                "value": value,
            }
        )

    if isinstance(value, Integral):
        return canonical_json(
            {
                "type": "integer",
                "value": int(value),
            }
        )

    if isinstance(value, Real):
        result = float(value)

        if np.isnan(result):
            return canonical_json(
                {
                    "type": "missing",
                    "value": None,
                }
            )

        if not np.isfinite(result):
            raise ValueError("Audit values must be finite.")

        return canonical_json(
            {
                "type": "real",
                "value": result,
            }
        )

    raise TypeError(
        "Keyed-update audit cannot serialize values of type " f"{type(value).__name__}."
    )


def _lineage_integer(
    value: Any,
) -> int:
    if isinstance(value, np.generic):
        value = value.item()

    if type(value) is not int:
        raise TypeError("Artifact lineage row numbers must be integers.")

    return value


def _endpoint_columns(
    state: ArtifactData,
) -> dict[str, dict[str, str]]:
    return {
        label: {
            "event": pair.event_profile.col_name,
            "duration": pair.time_profile.col_name,
        }
        for label, pair in state.pairs.pairs.items()
    }


def _validate_column_roles(
    *,
    base: ArtifactData,
    updates: ArtifactData,
    column_mapping: dict[str, str],
) -> None:
    for target_column, update_column in column_mapping.items():
        if target_column not in base.df.columns:
            raise KeyError(
                f"Base artifact does not contain target column " f"{target_column!r}."
            )

        if update_column not in updates.df.columns:
            raise KeyError(
                f"Update artifact does not contain source column " f"{update_column!r}."
            )

        base_role = base.profiles[target_column].col_type

        update_role = updates.profiles[update_column].col_type

        if base_role != update_role:
            raise TypeError(
                f"Semantic datatype mismatch for "
                f"{target_column!r} <- {update_column!r}: "
                f"{base_role!r} != {update_role!r}."
            )


def _validate_endpoint_mapping(
    *,
    base: ArtifactData,
    updates: ArtifactData,
    column_mapping: dict[str, str],
    endpoint_mapping: dict[str, str],
) -> None:
    base_endpoints = _endpoint_columns(base)

    update_endpoints = _endpoint_columns(updates)

    replaced_targets = set(column_mapping)

    affected_base_endpoints = {
        label
        for label, columns in base_endpoints.items()
        if (
            columns["event"] in replaced_targets
            or columns["duration"] in replaced_targets
        )
    }

    if affected_base_endpoints != set(endpoint_mapping):
        raise ValueError(
            "Endpoint mapping must cover exactly every base "
            "survival endpoint affected by the column update. "
            f"Affected={sorted(affected_base_endpoints)!r}; "
            f"mapped={sorted(endpoint_mapping)!r}."
        )

    base_definitions = base.metadata.get("endpoint_definitions", {})

    update_definitions = updates.metadata.get("endpoint_definitions", {})

    for base_label, update_label in endpoint_mapping.items():
        if base_label not in base_endpoints:
            raise KeyError(f"Unknown base endpoint: {base_label!r}.")

        if update_label not in update_endpoints:
            raise KeyError(f"Unknown update endpoint: {update_label!r}.")

        base_pair = base.pairs.pairs[base_label]

        update_pair = updates.pairs.pairs[update_label]

        if base_pair.event_structure != update_pair.event_structure:
            raise ValueError(
                f"Event-structure mismatch for " f"{base_label!r} <- {update_label!r}."
            )

        if base_pair.observation_scheme != update_pair.observation_scheme:
            raise ValueError(
                f"Observation-scheme mismatch for "
                f"{base_label!r} <- {update_label!r}."
            )

        base_columns = base_endpoints[base_label]

        update_columns = update_endpoints[update_label]

        if column_mapping.get(base_columns["event"]) != update_columns["event"]:
            raise ValueError(
                f"Endpoint {base_label!r} requires event mapping "
                f"{base_columns['event']!r} <- "
                f"{update_columns['event']!r}."
            )

        if column_mapping.get(base_columns["duration"]) != update_columns["duration"]:
            raise ValueError(
                f"Endpoint {base_label!r} requires duration mapping "
                f"{base_columns['duration']!r} <- "
                f"{update_columns['duration']!r}."
            )

        if not is_bool_dtype(updates.df[update_columns["event"]].dtype):
            raise TypeError(
                f"Update event column "
                f"{update_columns['event']!r} must use a "
                "Boolean dtype."
            )

        duration_series = updates.df[update_columns["duration"]]

        if (
            not is_numeric_dtype(duration_series.dtype)
            or is_bool_dtype(duration_series.dtype)
            or is_complex_dtype(duration_series.dtype)
        ):
            raise TypeError(
                f"Update duration column "
                f"{update_columns['duration']!r} must be a "
                "real numeric column."
            )

        duration_values = duration_series.dropna().to_numpy(dtype=float)

        if not np.isfinite(duration_values).all() or (duration_values < 0).any():
            raise ValueError(
                f"Update duration column "
                f"{update_columns['duration']!r} contains "
                "invalid survival times."
            )

        duration_metadata = updates.metadata["columns"][update_columns["duration"]]

        duration_unit = duration_metadata.get("unit")

        if duration_unit is None or duration_unit.get("dimension") != "time":
            raise ValueError(
                f"Update duration column "
                f"{update_columns['duration']!r} requires an "
                "explicit time unit."
            )

        base_definition = base_definitions.get(base_label)

        update_definition = update_definitions.get(update_label)

        if not isinstance(base_definition, str) or not base_definition.strip():
            raise ValueError(
                f"Base endpoint {base_label!r} does not have " "a reviewed definition."
            )

        if not isinstance(update_definition, str) or not update_definition.strip():
            raise ValueError(
                f"Update endpoint {update_label!r} does not have "
                "a reviewed definition."
            )

        if base_definition.strip() != update_definition.strip():
            raise ValueError(
                f"Endpoint definitions differ for "
                f"{base_label!r} <- {update_label!r}. "
                "Resolve the scientific definition explicitly "
                "before replacing values."
            )


def apply_keyed_update(
    *,
    base: ArtifactData,
    updates: ArtifactData,
    operation: UpdateColumnsByKey,
):
    """Create a base-shaped artifact with selected columns replaced."""

    base_state = base.copy()
    update_state = updates.copy()

    base_identities, base_positions = _identifier_index(
        state=base_state,
        column=operation.base_key,
        artifact_role="Base",
    )

    update_identities, update_positions = _identifier_index(
        state=update_state,
        column=operation.update_key,
        artifact_role="Update",
    )

    base_identity_set = set(base_identities)

    update_identity_set = set(update_identities)

    missing_base_identities = base_identity_set - update_identity_set

    if missing_base_identities:
        missing_labels = sorted(
            _identifier_json(identity) for identity in missing_base_identities
        )

        raise KeyError(
            "The update artifact does not cover every base "
            f"patient: {missing_labels!r}."
        )

    unused_update_identities = update_identity_set - base_identity_set

    column_mapping = dict(operation.column_mapping)

    endpoint_mapping = dict(operation.endpoint_mapping)

    _validate_column_roles(
        base=base_state,
        updates=update_state,
        column_mapping=column_mapping,
    )

    _validate_endpoint_mapping(
        base=base_state,
        updates=update_state,
        column_mapping=column_mapping,
        endpoint_mapping=endpoint_mapping,
    )

    matched_update_rows = [update_positions[identity] for identity in base_identities]

    base_lineage = base_state.lineage.reset_index(drop=True)

    update_lineage = update_state.lineage.reset_index(drop=True)

    column_update_records = []
    changed_cell_count = 0

    original_types = {
        column_name: profile.col_type
        for column_name, profile in base_state.profiles.items()
    }

    for target_column, update_column in operation.column_mapping:
        old_values = base_state.df[target_column].tolist()

        replacement_series = (
            update_state.df[update_column]
            .iloc[matched_update_rows]
            .reset_index(drop=True)
        )

        replacement_values = replacement_series.tolist()

        for base_row, update_row in enumerate(matched_update_rows):
            identity = base_identities[base_row]

            old_value = old_values[base_row]

            replacement_value = replacement_values[base_row]

            old_value_json = _audit_value(old_value)

            replacement_value_json = _audit_value(replacement_value)

            changed = old_value_json != replacement_value_json

            changed_cell_count += int(changed)

            base_parent = base_lineage.iloc[base_row]

            update_parent = update_lineage.iloc[update_row]

            column_update_records.append(
                {
                    "operation": "keyed_update",
                    "base_key_column": (operation.base_key),
                    "update_key_column": (operation.update_key),
                    "key_type": identity[0],
                    "key_value": (_identifier_json(identity)),
                    "base_row": base_row,
                    "update_row": update_row,
                    "base_parent_artifact": str(base_parent["parent_artifact"]),
                    "base_parent_row": (_lineage_integer(base_parent["parent_row"])),
                    "update_parent_artifact": str(update_parent["parent_artifact"]),
                    "update_parent_row": (
                        _lineage_integer(update_parent["parent_row"])
                    ),
                    "target_column": (target_column),
                    "update_column": (update_column),
                    "old_value": (old_value_json),
                    "replacement_value": (replacement_value_json),
                    "changed": changed,
                    "reason": operation.reason,
                }
            )

        # Passing the array—not the indexed Series—prevents
        # accidental index alignment against the base dataframe.
        base_state.df[target_column] = pd.Series(
            replacement_series.array,
            index=base_state.df.index,
            name=target_column,
        )

        source_metadata = update_state.metadata["columns"][update_column]

        base_state.metadata["columns"][target_column] = json_clone(source_metadata)

        source_ranks = update_state.ranks["columns"].get(update_column)

        if source_ranks is None:
            base_state.ranks["columns"].pop(
                target_column,
                None,
            )
        else:
            base_state.ranks["columns"][target_column] = json_clone(source_ranks)

    refresh_profiles(
        base_state,
        original_types,
    )

    validate_state(base_state)

    unused_update_records = []

    for identity in sorted(
        unused_update_identities,
        key=_identifier_json,
    ):
        update_row = update_positions[identity]

        update_parent = update_lineage.iloc[update_row]

        unused_update_records.append(
            {
                "update_key_column": (operation.update_key),
                "key_type": identity[0],
                "key_value": (_identifier_json(identity)),
                "update_row": update_row,
                "update_parent_artifact": str(update_parent["parent_artifact"]),
                "update_parent_row": (_lineage_integer(update_parent["parent_row"])),
                "reason": ("Update-source patient is not part of " "the base cohort."),
            }
        )

    audit = [
        {
            "step": 1,
            "operation": "keyed_update",
            "base_key_column": (operation.base_key),
            "update_key_column": (operation.update_key),
            "base_rows": len(base_state.df),
            "update_rows": len(update_state.df),
            "matched_base_rows": len(base_positions),
            "unused_update_rows": len(unused_update_records),
            "updated_columns": len(column_mapping),
            "audited_cells": len(column_update_records),
            "changed_cells": (changed_cell_count),
            "reason": operation.reason,
        }
    ]

    return (
        base_state,
        audit,
        column_update_records,
        unused_update_records,
    )


def json_clone(
    value: Any,
) -> Any:
    """Make a detached JSON-compatible metadata copy."""

    return json.loads(canonical_json(value))
