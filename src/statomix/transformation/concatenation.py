"""Strict multi-parent concatenation; no implicit deduplication or schema loss."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from statomix.core.artifacts import canonical_json
from statomix.curation.columns import DataTypes
from statomix.curation.survival.report import SurvPairs

from .metadata import ArtifactData, refresh_profiles, validate_state


def _mapped(parent, mapping):
    if set(mapping) != set(parent.df.columns):
        raise ValueError(
            "Each mapping must cover every source column; implicit column loss is forbidden."
        )
    if any(not isinstance(v, str) or not v for v in mapping.values()) or len(
        set(mapping.values())
    ) != len(mapping):
        raise ValueError("Mapped output names must be unique, nonempty strings.")
    state = parent.copy()
    state.df = state.df.rename(columns=mapping)
    state.profiles = {
        mapping[n]: replace(p, col_name=mapping[n]) for n, p in state.profiles.items()
    }
    state.metadata["columns"] = {
        mapping[n]: v for n, v in state.metadata["columns"].items()
    }
    state.ranks["columns"] = {mapping[n]: v for n, v in state.ranks["columns"].items()}
    state.pairs = SurvPairs(
        pairs={
            label: replace(
                pair,
                event_profile=replace(
                    pair.event_profile, col_name=mapping[pair.event_profile.col_name]
                ),
                time_profile=replace(
                    pair.time_profile, col_name=mapping[pair.time_profile.col_name]
                ),
            )
            for label, pair in state.pairs.pairs.items()
        }
    )
    return state


def _endpoints(state):
    return {
        label: {
            "event": pair.event_profile.col_name,
            "time": pair.time_profile.col_name,
            "event_structure": pair.event_structure,
            "observation_scheme": pair.observation_scheme,
        }
        for label, pair in state.pairs.pairs.items()
    }


def concatenate_states(parents, *, mappings, identity_columns, cohort_column):
    if len(parents) < 2 or len(parents) != len(mappings):
        raise ValueError(
            "Concatenation needs at least two parents and one mapping per parent."
        )
    if not identity_columns or isinstance(identity_columns, str):
        raise ValueError("Provide a nonempty sequence of patient identity columns.")
    states = [
        _mapped(parent, mapping)
        for parent, mapping in zip(parents, mappings, strict=True)
    ]
    first = states[0]
    order = list(first.df.columns)
    if (
        cohort_column in order
        or not isinstance(cohort_column, str)
        or not cohort_column
    ):
        raise ValueError("Cohort provenance column must be a new nonempty name.")
    if not set(identity_columns).issubset(order):
        raise ValueError("Patient identity columns are missing from the output schema.")
    definitions = first.metadata["endpoint_definitions"]
    if any(
        not isinstance(definitions.get(label), str) or not definitions[label].strip()
        for label in first.pairs.pairs
    ):
        raise ValueError(
            "Each survival endpoint needs a reviewed definition/time-origin declaration before concatenation."
        )
    for other in states[1:]:
        if set(other.df.columns) != set(order):
            raise ValueError(
                "Mapped schemas differ; explicit harmonization is required."
            )
        if (
            _endpoints(first) != _endpoints(other)
            or definitions != other.metadata["endpoint_definitions"]
        ):
            raise ValueError("Endpoint bindings or definitions differ.")
        for name in order:
            if first.profiles[name].col_type != other.profiles[name].col_type:
                raise ValueError(f"Semantic datatype conflict: {name!r}.")
            if first.df[name].dtype != other.df[name].dtype:
                raise ValueError(
                    f"Physical datatype conflict: {name!r}; harmonize explicitly."
                )
            left, right = (
                first.metadata["columns"][name],
                other.metadata["columns"][name],
            )
            if left["unit"] != right["unit"]:
                raise ValueError(f"Unit conflict: {name!r}.")
            if (
                any(p.time_profile.col_name == name for p in first.pairs.pairs.values())
                and left["unit"] is None
            ):
                raise ValueError(f"Unknown survival duration unit: {name!r}.")
            domains = {
                canonical_json(v): v
                for v in left["category_domain"] + right["category_domain"]
            }
            left["category_domain"] = [domains[k] for k in sorted(domains)]
            lr, rr = first.ranks["columns"].get(name), other.ranks["columns"].get(name)
            if (lr is None) != (rr is None):
                raise ValueError(f"Ranked/unranked metadata conflict: {name!r}.")
            if lr is not None:
                merged = {canonical_json(r["category"]): r for r in lr}
                for record in rr:
                    key = canonical_json(record["category"])
                    if key in merged and merged[key]["rank"] != record["rank"]:
                        raise ValueError(f"Conflicting category rank in {name!r}.")
                    merged[key] = record
                first.ranks["columns"][name] = sorted(
                    merged.values(), key=lambda r: r["rank"]
                )
    frames, lineage = [], []
    for state in states:
        frame = state.df.loc[:, order].copy()
        if frame[list(identity_columns)].isna().any().any():
            raise ValueError("Patient identity contains missing values.")
        frame[cohort_column] = state.lineage["source_dataset"].to_numpy()
        frames.append(frame)
        lineage.append(state.lineage)
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(subset=list(identity_columns)).any():
        raise ValueError(
            "Overlapping patient identities: resolve duplicates explicitly before concatenation."
        )
    output_lineage = pd.concat(lineage, ignore_index=True)
    output_lineage["output_row"] = range(len(result))
    metadata = first.metadata
    metadata["columns"][cohort_column] = {
        "unit": None,
        "category_domain": [
            {"type": "string", "value": n}
            for n in sorted(result[cohort_column].unique())
        ],
    }
    output = ArtifactData(
        result, first.profiles, first.pairs, metadata, first.ranks, output_lineage
    )
    refresh_profiles(output, {cohort_column: DataTypes.CATEGORICAL})
    validate_state(output)
    audit = [
        {
            "step": i + 1,
            "operation": "concatenate",
            "input_rows": len(s.df),
            "output_rows": len(result),
            "overlap_policy": "reject",
            "parent_order": i,
        }
        for i, s in enumerate(states)
    ]
    return output, audit
