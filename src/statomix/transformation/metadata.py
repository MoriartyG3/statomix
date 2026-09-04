"""Validated data and semantic state shared by transformation operations."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from statomix.core.artifacts import canonical_json
from statomix.curation.columns import ColProfiler, DataTypes
from statomix.curation.survival.report import SurvPairs
from statomix.storage.parquet_metadata import (
    _category_identity,
    _decode_category,
    _encode_category,
)

from .specifications import Unit


@dataclass
class ArtifactData:
    df: pd.DataFrame
    profiles: dict
    pairs: SurvPairs
    metadata: dict
    ranks: dict
    lineage: pd.DataFrame

    def copy(self):
        return ArtifactData(
            self.df.copy(deep=True),
            dict(self.profiles),
            self.pairs,
            json.loads(canonical_json(self.metadata)),
            json.loads(canonical_json(self.ranks)),
            self.lineage.copy(deep=True),
        )


def refresh_profiles(state: ArtifactData, types: dict | None = None) -> None:
    types = types or {}
    profiler = ColProfiler()
    state.profiles = {
        name: replace(
            profiler.get_col_profile(col_name=name, col_series=state.df[name]),
            col_type=types.get(
                name,
                (
                    state.profiles[name].col_type
                    if name in state.profiles
                    else DataTypes.NUMERICAL
                ),
            ),
        )
        for name in state.df.columns
    }


def validate_state(state: ArtifactData) -> None:
    if not state.df.columns.is_unique or any(
        not isinstance(c, str) or not c for c in state.df.columns
    ):
        raise ValueError("Data requires unique, nonempty string column names.")
    if set(state.df.columns) != set(state.profiles):
        raise ValueError("Column profiles must cover the exact output schema.")
    if state.metadata.get("schema_version") != 1:
        raise ValueError("Unsupported semantic metadata schema.")
    columns = state.metadata["columns"]
    if set(columns) != set(state.df.columns):
        raise ValueError("Semantic metadata must cover the exact output schema.")
    for record in columns.values():
        if record["unit"] is not None:
            Unit(**record["unit"])
    if state.ranks.get("schema_version") != 1:
        raise ValueError("Unsupported category-rank metadata schema.")
    for name, records in state.ranks["columns"].items():
        if (
            name not in columns
            or state.profiles[name].col_type != DataTypes.CATEGORICAL
        ):
            raise ValueError(f"Ranks refer to a noncategorical column: {name!r}.")
        identities = {}
        for record in records:
            identity = _category_identity(_decode_category(record["category"]))
            rank = record["rank"]
            if type(rank) is not int or rank < 0 or identity in identities:
                raise ValueError(f"Invalid rank metadata for {name!r}.")
            identities[identity] = rank
        if len(set(identities.values())) != len(identities):
            raise ValueError(f"Different categories share a rank in {name!r}.")
        observed = {_category_identity(x) for x in state.df[name].dropna().unique()}
        if not observed.issubset(identities):
            raise ValueError(f"Observed unranked categories in {name!r}.")
    for label, pair in state.pairs.pairs.items():
        event = pair.event_profile.col_name
        duration = pair.time_profile.col_name
        if event not in columns or duration not in columns:
            raise ValueError(f"Endpoint {label!r} refers to missing columns.")
        if (
            state.profiles[event].col_type != DataTypes.SURVIVAL
            or state.profiles[duration].col_type != DataTypes.SURVIVAL
        ):
            raise ValueError(f"Endpoint {label!r} requires Survival column profiles.")
        unit = columns[duration]["unit"]
        if unit is not None and unit["dimension"] != "time":
            raise ValueError(f"Endpoint {label!r} requires time units.")
        values = state.df[duration]
        if not state.df[event].dropna().isin([False, True, 0, 1]).all():
            raise ValueError(f"Endpoint {label!r} contains uncurated event values.")
        if (
            not pd.api.types.is_numeric_dtype(values.dtype)
            or pd.api.types.is_bool_dtype(values.dtype)
            or pd.api.types.is_complex_dtype(values.dtype)
        ):
            raise TypeError(f"Endpoint duration {duration!r} is not numerical.")
        observed = values.dropna().to_numpy(dtype=float)
        if not np.isfinite(observed).all() or (observed < 0).any():
            raise ValueError(f"Endpoint duration {duration!r} contains invalid times.")
    if len(state.lineage) != len(state.df):
        raise ValueError("Row lineage must cover every output row.")


def initial_metadata(df, profiles, pairs, *, units=None, endpoint_definitions=None):
    units = units or {}
    endpoint_definitions = endpoint_definitions or {}
    if set(units) - set(df.columns) or set(endpoint_definitions) - set(pairs.pairs):
        raise ValueError("Metadata declarations refer to unknown columns/endpoints.")
    columns = {}
    for name in df.columns:
        domain = []
        if profiles[name].col_type == DataTypes.CATEGORICAL:
            domain = [_encode_category(x) for x in df[name].dropna().unique()]
            domain.sort(key=canonical_json)
        unit = units.get(name)
        columns[name] = {
            "unit": unit.to_dict() if unit else None,
            "category_domain": domain,
        }
    return {
        "schema_version": 1,
        "columns": columns,
        "endpoint_definitions": dict(endpoint_definitions),
    }
