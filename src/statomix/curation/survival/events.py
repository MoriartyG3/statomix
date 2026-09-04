"""Typed parsing, mapping, and normalization of survival-event values."""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Mapping
from numbers import Integral, Real
from typing import Protocol

import numpy as np
import pandas as pd


class SurvivalEventEdit(Protocol):
    """Fields needed to apply one survival-event curation instruction."""

    category: object
    category_encoding: str
    event_observed: bool | None
    remove: bool


def _python_scalar(value: object) -> object:
    """Convert NumPy scalars without converting text into numbers."""

    if isinstance(value, np.generic):
        return value.item()
    return value


def is_missing_scalar(value: object) -> bool:
    """Recognize missing scalar values without treating text as missing."""

    result = pd.isna(value)
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False


def is_blank_cell(value: object) -> bool:
    """Recognize an empty curation-instruction cell."""

    return is_missing_scalar(value) or (isinstance(value, str) and not value.strip())


def _parse_optional_boolean(
    value: object,
    *,
    field_name: str,
    col_name: str,
    category: object,
) -> bool | None:
    value = _python_scalar(value)

    if is_blank_cell(value):
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, Real) and value in (0, 1):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False

    raise ValueError(
        f"{field_name} for column {col_name!r}, category {category!r} "
        f"must be True/False or exact 1/0; received {value!r}."
    )


def parse_optional_event_observed(
    value: object,
    *,
    col_name: str,
    category: object,
) -> bool | None:
    """Convert an Excel instruction into bool or None, never into text."""

    return _parse_optional_boolean(
        value,
        field_name="event_observed",
        col_name=col_name,
        category=category,
    )


def parse_remove_instruction(
    value: object,
    *,
    col_name: str,
    category: object,
) -> bool:
    """A blank removal instruction means do not remove the value."""

    parsed = _parse_optional_boolean(
        value,
        field_name="remove",
        col_name=col_name,
        category=category,
    )
    return False if parsed is None else parsed


def is_canonical_event_value(value: object) -> bool:
    """Accept typed Boolean or exact numeric 0/1, but not text labels."""

    value = _python_scalar(value)

    if is_missing_scalar(value):
        return False
    if isinstance(value, bool):
        return True
    return isinstance(value, Real) and value in (0, 1)


def encode_category_scalar(value: object) -> str:
    """Preserve source category identity across Excel and Parquet.

    Numeric 1 and 1.0 share an identity. Text "1" and Boolean True have
    different identities.
    """

    value = _python_scalar(value)

    if is_missing_scalar(value):
        raise ValueError("A source category must not be missing.")

    if isinstance(value, bool):
        payload = {"type": "boolean", "value": value}
    elif isinstance(value, Integral):
        payload = {"type": "number", "value": int(value)}
    elif isinstance(value, Real):
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError("Source categories must contain finite numbers.")

        normalized_number = (
            int(numeric_value) if numeric_value.is_integer() else numeric_value
        )
        payload = {"type": "number", "value": normalized_number}
    elif isinstance(value, str):
        payload = {"type": "string", "value": value}
    else:
        raise TypeError(
            "Survival-event categories must be strings, finite numbers, "
            f"or Booleans; received {type(value).__name__}."
        )

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def decode_category_scalar(encoded: str) -> object:
    """Read a validated typed category identity."""

    if not isinstance(encoded, str) or not encoded.strip():
        raise ValueError("category_encoding must be a non-empty JSON string.")

    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid survival-event category encoding: {encoded!r}."
        ) from exc

    if not isinstance(payload, dict) or set(payload) != {"type", "value"}:
        raise ValueError(f"Invalid category encoding: {encoded!r}.")

    category_type = payload["type"]
    value = payload["value"]

    if category_type == "boolean" and isinstance(value, bool):
        return value

    if category_type == "number" and not isinstance(value, bool):
        if isinstance(value, int):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value

    if category_type == "string" and isinstance(value, str):
        return value

    raise ValueError(f"Invalid category encoding: {encoded!r}.")


def normalize_survival_event_series(
    series: pd.Series,
    *,
    column_name: str | None = None,
) -> pd.Series:
    """Return nullable Boolean values while preserving missing observations."""

    resolved_name = column_name if column_name is not None else str(series.name)
    normalized_values: list[object] = []

    for value in series.array:
        value = _python_scalar(value)

        if is_missing_scalar(value):
            normalized_values.append(pd.NA)
        elif is_canonical_event_value(value):
            normalized_values.append(bool(value))
        else:
            raise ValueError(
                f"Survival event column {resolved_name!r} contains "
                f"noncanonical value {value!r}. Map every retained text "
                "label explicitly in SurvCatMeta."
            )

    return pd.Series(
        pd.array(normalized_values, dtype="boolean"),
        index=series.index,
        name=series.name,
    )


def apply_survival_event_edits(
    *,
    df: pd.DataFrame,
    category_edits: Mapping[str, Mapping[str, SurvivalEventEdit]],
    require_all_categories: bool = True,
) -> pd.DataFrame:
    """Apply event mappings against original values, not intermediate results."""

    if not df.columns.is_unique:
        raise ValueError("Survival-event curation requires unique column names.")
    if not isinstance(require_all_categories, bool):
        raise TypeError("require_all_categories must be Boolean.")

    curated_df = df.copy()

    for col_name, column_edits in category_edits.items():
        if col_name not in curated_df.columns:
            raise KeyError(
                f"Survival-event schema refers to missing column {col_name!r}."
            )

        actions: dict[str, tuple[bool | None, bool]] = {}

        for edit in column_edits.values():
            identity = encode_category_scalar(edit.category)
            if identity != edit.category_encoding:
                raise ValueError(
                    f"Inconsistent source-category identity in {col_name!r}."
                )
            if identity in actions:
                raise ValueError(f"Duplicate survival-event edit in {col_name!r}.")

            event_observed = parse_optional_event_observed(
                edit.event_observed,
                col_name=col_name,
                category=edit.category,
            )
            remove = parse_remove_instruction(
                edit.remove,
                col_name=col_name,
                category=edit.category,
            )

            if remove and event_observed is not None:
                raise ValueError(
                    f"Category {edit.category!r} in {col_name!r} cannot "
                    "be both mapped and removed."
                )
            if not remove and event_observed is None:
                raise ValueError(
                    f"Category {edit.category!r} in {col_name!r} has no action."
                )

            actions[identity] = (event_observed, remove)

        has_explicit_mapping = any(
            event_observed is not None for event_observed, _ in actions.values()
        )

        source_series = curated_df[col_name]
        observed_identities: set[str] = set()
        mapped_values: list[object] = []

        for value in source_series.array:
            if is_missing_scalar(value):
                mapped_values.append(pd.NA)
                continue

            identity = encode_category_scalar(value)
            observed_identities.add(identity)
            action = actions.get(identity)

            if action is not None:
                event_observed, remove = action
                mapped_values.append(pd.NA if remove else event_observed)
            elif has_explicit_mapping:
                raise ValueError(
                    f"Survival event column {col_name!r} has an unmapped "
                    f"category {value!r}. Complete the event mapping."
                )
            else:
                # A removal-only schema may leave canonical 0/1 values.
                mapped_values.append(value)

        absent_identities = set(actions) - observed_identities
        if require_all_categories and absent_identities:
            absent_categories = [
                decode_category_scalar(identity)
                for identity in sorted(absent_identities)
            ]
            raise ValueError(
                f"Survival-event schema categories are absent from "
                f"{col_name!r}: {absent_categories!r}. The schema may be stale."
            )

        mapped_series = pd.Series(
            mapped_values,
            index=source_series.index,
            name=source_series.name,
            dtype="object",
        )
        curated_df[col_name] = normalize_survival_event_series(
            mapped_series,
            column_name=col_name,
        )

    return curated_df


def normalize_survival_event_columns(
    *,
    df: pd.DataFrame,
    event_columns: Collection[str],
) -> pd.DataFrame:
    """Normalize every declared event column before curated-data persistence."""

    if isinstance(event_columns, (str, bytes)):
        raise TypeError("event_columns must be a collection, not one string.")
    if not df.columns.is_unique:
        raise ValueError("Survival-event curation requires unique column names.")

    ordered_columns = tuple(dict.fromkeys(event_columns))
    if any(not isinstance(name, str) or not name.strip() for name in ordered_columns):
        raise ValueError("Event column names must be non-empty strings.")

    missing_columns = [name for name in ordered_columns if name not in df.columns]
    if missing_columns:
        raise KeyError(
            f"Declared survival-event columns are missing: {missing_columns!r}."
        )

    curated_df = df.copy()
    for column_name in ordered_columns:
        curated_df[column_name] = normalize_survival_event_series(
            curated_df[column_name],
            column_name=column_name,
        )

    return curated_df
