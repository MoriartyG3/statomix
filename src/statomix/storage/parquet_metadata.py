"""Parquet persistence for Statomix categorical-rank metadata."""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RANK_METADATA_KEY = b"statomix.categorical_ranks"
RANK_METADATA_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class RankedReference:
    """Reference selected from ranks and observed categories."""

    column_name: str
    category: object
    rank: int
    absent_lower_ranked_categories: tuple[object, ...]


def _python_scalar(value: object) -> object:
    """Convert a NumPy scalar into its corresponding Python scalar."""

    if isinstance(value, np.generic):
        return value.item()

    return value


def _encode_category(
    value: object,
) -> dict[str, object]:
    """Encode a supported category while preserving its scalar type."""

    value = _python_scalar(value)

    if isinstance(value, bool):
        return {
            "type": "boolean",
            "value": value,
        }

    if isinstance(value, int):
        return {
            "type": "integer",
            "value": value,
        }

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Ranked category values must not be infinite " "or NaN.")

        return {
            "type": "number",
            "value": value,
        }

    if isinstance(value, str):
        return {
            "type": "string",
            "value": value,
        }

    raise TypeError(
        "Ranked categorical values must be strings, integers, "
        "finite floating-point numbers, or Booleans; received "
        f"{type(value).__name__}."
    )


def _decode_category(
    encoded: Mapping[str, object],
) -> object:
    """Decode a typed category stored in Parquet metadata."""

    category_type = encoded.get("type")
    value = encoded.get("value")

    if category_type == "boolean" and isinstance(value, bool):
        return value

    if (
        category_type == "integer"
        and isinstance(value, int)
        and not isinstance(value, bool)
    ):
        return value

    if (
        category_type == "number"
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        return float(value)

    if category_type == "string" and isinstance(value, str):
        return value

    raise ValueError("Invalid encoded categorical value: " f"{dict(encoded)!r}.")


def _category_identity(value: object) -> str:
    """Return a type-sensitive identity for a category scalar."""

    return json.dumps(
        _encode_category(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _build_metadata_payload(
    *,
    df: pd.DataFrame,
    category_ranks: Mapping[
        str,
        Mapping[str, int],
    ],
) -> dict[str, Any]:
    """Resolve report labels against actual curated values."""

    columns_payload: dict[
        str,
        list[dict[str, object]],
    ] = {}

    for col_name, declared_ranks in category_ranks.items():
        if col_name not in df.columns:
            raise KeyError(
                "Rank metadata refers to missing curated column " f"{col_name!r}."
            )

        observed_values = pd.unique(df[col_name].dropna())

        observed_by_label: dict[str, object] = {}
        observed_identity_by_label: dict[str, str] = {}

        for value in observed_values:
            python_value = _python_scalar(value)
            label = str(python_value)
            identity = _category_identity(python_value)

            previous_identity = observed_identity_by_label.get(label)

            if previous_identity is not None and previous_identity != identity:
                raise ValueError(
                    f"Column {col_name!r} contains distinct "
                    "category values that both render as "
                    f"{label!r}. Statomix cannot safely match "
                    "the Excel rank metadata to these values."
                )

            observed_by_label[label] = python_value
            observed_identity_by_label[label] = identity

        declared_labels = set(declared_ranks)
        observed_labels = set(observed_by_label)

        missing_ranks = sorted(observed_labels - declared_labels)

        absent_categories = sorted(declared_labels - observed_labels)

        if missing_ranks:
            raise ValueError(
                f"Ranked categorical column {col_name!r} "
                "has observed categories without ranks: "
                f"{missing_ranks!r}."
            )

        if absent_categories:
            raise ValueError(
                f"Rank metadata for {col_name!r} contains "
                "categories absent from the curated DataFrame: "
                f"{absent_categories!r}."
            )

        records = [
            {
                "category": _encode_category(observed_by_label[label]),
                "rank": int(rank),
            }
            for label, rank in sorted(
                declared_ranks.items(),
                key=lambda item: item[1],
            )
        ]

        columns_payload[col_name] = records

    return {
        "schema_version": RANK_METADATA_SCHEMA_VERSION,
        "columns": columns_payload,
    }


def write_dataframe_with_category_ranks(
    *,
    df: pd.DataFrame,
    path: Path,
    category_ranks: Mapping[
        str,
        Mapping[str, int],
    ],
) -> dict[str, Any]:
    """Write a DataFrame with typed ranks in Parquet metadata."""

    payload = _build_metadata_payload(
        df=df,
        category_ranks=category_ranks,
    )

    table = pa.Table.from_pandas(
        df=df,
        preserve_index=None,
    )

    metadata = dict(table.schema.metadata or {})

    metadata[RANK_METADATA_KEY] = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    table = table.replace_schema_metadata(metadata)

    pq.write_table(
        table=table,
        where=path,
    )

    return payload


def load_category_rank_metadata(
    path: Path,
) -> dict[str, Any]:
    """Load Statomix category ranks from a Parquet footer."""

    metadata = (
        pq.read_schema(
            where=path,
        ).metadata
        or {}
    )

    encoded_payload = metadata.get(RANK_METADATA_KEY)

    if encoded_payload is None:
        return {
            "schema_version": (RANK_METADATA_SCHEMA_VERSION),
            "columns": {},
        }

    try:
        payload = json.loads(encoded_payload.decode("utf-8"))
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(
            "Invalid Statomix categorical-rank metadata " f"in {path}."
        ) from exc

    if payload.get("schema_version") != RANK_METADATA_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported categorical-rank metadata schema "
            "version: "
            f"{payload.get('schema_version')!r}."
        )

    if not isinstance(payload.get("columns"), dict):
        raise ValueError("Categorical-rank metadata must contain a " "columns mapping.")

    return payload


def select_lowest_rank_reference(
    *,
    df: pd.DataFrame,
    column_name: str,
    rank_metadata: Mapping[str, Any],
) -> RankedReference:
    """Select the lowest-ranked category observed in a DataFrame."""

    if column_name not in df.columns:
        raise KeyError("Analysis DataFrame does not contain " f"{column_name!r}.")

    column_records = rank_metadata.get(
        "columns",
        {},
    ).get(column_name)

    if not column_records:
        raise ValueError("No categorical-rank metadata exists for " f"{column_name!r}.")

    observed_by_identity = {
        _category_identity(value): _python_scalar(value)
        for value in pd.unique(df[column_name].dropna())
    }

    ranked_observed: list[tuple[int, object]] = []

    absent: list[tuple[int, object]] = []

    for record in column_records:
        category = _decode_category(record["category"])

        rank = int(record["rank"])
        identity = _category_identity(category)

        if identity in observed_by_identity:
            ranked_observed.append(
                (
                    rank,
                    observed_by_identity[identity],
                )
            )
        else:
            absent.append(
                (
                    rank,
                    category,
                )
            )

    if not ranked_observed:
        raise ValueError(
            "None of the ranked categories for " f"{column_name!r} are observed."
        )

    selected_rank, selected_category = min(
        ranked_observed,
        key=lambda item: item[0],
    )

    absent_lower = tuple(category for rank, category in absent if rank < selected_rank)

    if absent_lower:
        warnings.warn(
            f"Lower-ranked categories {absent_lower!r} "
            f"are absent from {column_name!r}; selected "
            f"{selected_category!r} as the lowest observed "
            "reference category.",
            UserWarning,
            stacklevel=2,
        )

    return RankedReference(
        column_name=column_name,
        category=selected_category,
        rank=selected_rank,
        absent_lower_ranked_categories=absent_lower,
    )
