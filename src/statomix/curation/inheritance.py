"""Pure validation and reconstruction for curated-parent inheritance."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from statomix.core.errors import CuratedStateInheritanceError
from statomix.curation.categorical import CatMetaEditSchema
from statomix.curation.columns import ColProfile, ColProfiler, DataTypes
from statomix.curation.survival import (
    SurvCatMetaEditSchema,
    SurvivalDataTypes,
    SurvivalSemanticProfile,
    SurvPair,
    SurvPairs,
    get_survival_semantic_col_profile,
)
from statomix.curation.survival.events import apply_survival_event_edits


def apply_inherited_category_edits(
    *,
    target_df: pd.DataFrame,
    source_cat_meta_edit_schema: CatMetaEditSchema,
    source_surv_cat_meta_edit_schema: SurvCatMetaEditSchema,
    column_mapping: Mapping[str, str] | None,
    changed_columns: Collection[str],
) -> pd.DataFrame:
    """Apply parent decisions only to explicitly changed target columns."""

    if isinstance(changed_columns, (str, bytes)):
        raise CuratedStateInheritanceError(
            "changed_columns must be a collection of complete column names, "
            "not one string."
        )

    provided_changes = tuple(changed_columns)
    invalid = [
        name for name in provided_changes if not isinstance(name, str) or not name
    ]
    if invalid:
        raise CuratedStateInheritanceError(
            f"changed_columns must contain non-empty strings: {invalid!r}."
        )

    changed = set(provided_changes)
    mapping = dict(column_mapping or {})
    curated_df = target_df.copy(deep=True)

    for source_name, category_edits in source_cat_meta_edit_schema.cat_edits.items():
        target_name = mapping.get(source_name, source_name)
        if target_name not in changed:
            continue

        if target_name not in curated_df.columns:
            raise CuratedStateInheritanceError(
                "A parent category edit maps to a missing changed column: "
                f"{source_name!r} -> {target_name!r}."
            )

        category_mapping = {}
        for edit in category_edits.values():
            if edit.rename_to is not None:
                category_mapping[edit.category] = edit.rename_to
            elif edit.remove:
                category_mapping[edit.category] = pd.NA
            elif getattr(edit, "rank", None) is not None:
                continue
            else:
                raise CuratedStateInheritanceError(
                    f"Parent category edit for {source_name!r} has no action."
                )

        curated_df[target_name] = curated_df[target_name].replace(category_mapping)

    survival_edits = {}
    for (
        source_name,
        category_edits,
    ) in source_surv_cat_meta_edit_schema.cat_edits.items():
        target_name = mapping.get(source_name, source_name)
        if target_name not in changed:
            continue

        if target_name not in curated_df.columns:
            raise CuratedStateInheritanceError(
                "A parent event edit maps to a missing changed column: "
                f"{source_name!r} -> {target_name!r}."
            )

        survival_edits[target_name] = category_edits

    try:
        curated_df = apply_survival_event_edits(
            df=curated_df,
            category_edits=survival_edits,
            # Changed target data need not contain every parent category.
            require_all_categories=False,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CuratedStateInheritanceError(
            f"Inherited survival-event curation failed: {exc}"
        ) from exc

    return curated_df


@dataclass(frozen=True, slots=True, kw_only=True)
class InheritedCuratedState:
    """Target-specific semantic artifacts reconstructed from a parent state."""

    col_profiles: Mapping[str, ColProfile]
    survival_profiles: Mapping[str, SurvivalSemanticProfile]
    survival_pairs: SurvPairs
    column_mapping: Mapping[str, str]
    changed_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "col_profiles",
            MappingProxyType(dict(self.col_profiles)),
        )
        object.__setattr__(
            self,
            "survival_profiles",
            MappingProxyType(dict(self.survival_profiles)),
        )
        object.__setattr__(
            self,
            "column_mapping",
            MappingProxyType(dict(self.column_mapping)),
        )
        object.__setattr__(self, "changed_columns", tuple(self.changed_columns))

    @property
    def survival_labels(self) -> Mapping[str, str]:
        """Map each paired target column to its inherited endpoint label."""

        labels: dict[str, str] = {}
        for label, pair in self.survival_pairs.pairs.items():
            labels[pair.event_profile.col_name] = label
            labels[pair.time_profile.col_name] = label
        return MappingProxyType(labels)


def _validate_column_names(*, df: pd.DataFrame, role: str) -> tuple[str, ...]:
    columns = tuple(df.columns)
    non_string = [column for column in columns if not isinstance(column, str)]
    if non_string:
        raise CuratedStateInheritanceError(
            f"{role} dataframe contains non-string column names: {non_string!r}."
        )
    if len(set(columns)) != len(columns):
        raise CuratedStateInheritanceError(
            f"{role} dataframe contains duplicate column names."
        )
    return columns


def _validate_column_mapping(
    *,
    source_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    column_mapping: Mapping[str, str],
    strict: bool,
) -> dict[str, str]:
    mapping = dict(column_mapping)

    invalid_items = [
        (source, target)
        for source, target in mapping.items()
        if not isinstance(source, str)
        or not source.strip()
        or not isinstance(target, str)
        or not target.strip()
    ]
    if invalid_items:
        raise CuratedStateInheritanceError(
            "column_mapping keys and values must be non-empty strings; "
            f"invalid entries: {invalid_items!r}."
        )

    unknown_sources = sorted(set(mapping) - set(source_columns))
    if unknown_sources:
        raise CuratedStateInheritanceError(
            "column_mapping refers to columns absent from the curated parent: "
            f"{unknown_sources}."
        )

    mapped_targets = list(mapping.values())
    duplicate_targets = sorted(
        name for name, count in Counter(mapped_targets).items() if count > 1
    )
    if duplicate_targets:
        raise CuratedStateInheritanceError(
            "Multiple parent columns map to the same target column: "
            f"{duplicate_targets}."
        )

    complete_mapping = {
        source: mapping.get(source, source) for source in source_columns
    }
    complete_targets = list(complete_mapping.values())
    collisions = sorted(
        name for name, count in Counter(complete_targets).items() if count > 1
    )
    if collisions:
        raise CuratedStateInheritanceError(
            "The complete parent-to-target mapping contains collisions: "
            f"{collisions}."
        )

    expected = set(complete_targets)
    observed = set(target_columns)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)

    if missing:
        raise CuratedStateInheritanceError(
            "Target dataframe is missing inherited columns after mapping: "
            f"{missing}."
        )
    if strict and extra:
        raise CuratedStateInheritanceError(
            "Strict inheritance does not allow target-only columns: " f"{extra}."
        )

    return complete_mapping


def _align_parent_rows(
    *,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    complete_mapping: Mapping[str, str],
    row_key: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if row_key is None:
        if len(source_df) != len(target_df):
            raise CuratedStateInheritanceError(
                "Parent and target row counts differ. Supply a stable row_key "
                "only when both datasets contain the same observations."
            )
        return (
            source_df.reset_index(drop=True),
            target_df.reset_index(drop=True),
        )

    if row_key not in target_df.columns:
        raise CuratedStateInheritanceError(
            f"row_key {row_key!r} is absent from the target dataframe."
        )

    target_to_source = {target: source for source, target in complete_mapping.items()}
    source_row_key = target_to_source.get(row_key)
    if source_row_key is None:
        raise CuratedStateInheritanceError(
            f"row_key {row_key!r} does not correspond to a parent column."
        )

    source_keys = source_df[source_row_key]
    target_keys = target_df[row_key]
    for role, keys in (("parent", source_keys), ("target", target_keys)):
        if keys.isna().any():
            raise CuratedStateInheritanceError(
                f"{role} row_key {row_key!r} contains missing values."
            )
        if keys.duplicated().any():
            duplicates = keys[keys.duplicated(keep=False)].unique().tolist()
            raise CuratedStateInheritanceError(
                f"{role} row_key {row_key!r} is not unique; "
                f"duplicates include {duplicates[:5]!r}."
            )

    source_key_set = set(source_keys.tolist())
    target_key_set = set(target_keys.tolist())
    if source_key_set != target_key_set:
        missing = sorted(source_key_set - target_key_set, key=str)
        extra = sorted(target_key_set - source_key_set, key=str)
        raise CuratedStateInheritanceError(
            "Parent and target row identities differ: "
            f"missing target keys={missing[:5]!r}, "
            f"target-only keys={extra[:5]!r}."
        )

    aligned_source = source_df.set_index(source_row_key, drop=False).loc[
        target_keys.tolist()
    ]
    return aligned_source.reset_index(drop=True), target_df.reset_index(drop=True)


def _validate_declared_changes(
    *,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    complete_mapping: Mapping[str, str],
    changed_columns: Collection[str],
    row_key: str | None,
    strict: bool,
) -> tuple[str, ...]:
    if isinstance(changed_columns, (str, bytes)):
        raise CuratedStateInheritanceError(
            "changed_columns must be a collection of complete column names, "
            "not one string."
        )
    provided_changes = tuple(changed_columns)
    invalid = [
        name for name in provided_changes if not isinstance(name, str) or not name
    ]
    if invalid:
        raise CuratedStateInheritanceError(
            f"changed_columns must contain non-empty strings: {invalid!r}."
        )

    changed_set = set(provided_changes)
    unknown = sorted(changed_set - set(target_df.columns))
    if unknown:
        raise CuratedStateInheritanceError(
            f"changed_columns are absent from the target dataframe: {unknown}."
        )
    if row_key is not None and row_key in changed_set:
        raise CuratedStateInheritanceError(
            "row_key cannot also be declared as a changed column."
        )

    expected_targets = set(complete_mapping.values())
    target_only = set(target_df.columns) - expected_targets
    undeclared_target_only = sorted(target_only - changed_set)
    if not strict and undeclared_target_only:
        raise CuratedStateInheritanceError(
            "Target-only columns must be declared in changed_columns: "
            f"{undeclared_target_only}."
        )

    aligned_source, aligned_target = _align_parent_rows(
        source_df=source_df,
        target_df=target_df,
        complete_mapping=complete_mapping,
        row_key=row_key,
    )

    for source_name, target_name in complete_mapping.items():
        if target_name in changed_set:
            continue
        try:
            assert_series_equal(
                left=aligned_source[source_name],
                right=aligned_target[target_name],
                check_names=False,
                check_dtype=True,
                check_exact=True,
            )
        except AssertionError as error:
            raise CuratedStateInheritanceError(
                f"Target column {target_name!r} differs from parent column "
                f"{source_name!r} but was not declared in changed_columns."
            ) from error

    return tuple(name for name in target_df.columns if name in changed_set)


def _rebuild_column_profiles(
    *,
    target_df: pd.DataFrame,
    source_profiles: Mapping[str, ColProfile],
    complete_mapping: Mapping[str, str],
) -> dict[str, ColProfile]:
    missing_profiles = sorted(set(complete_mapping) - set(source_profiles))
    if missing_profiles:
        raise CuratedStateInheritanceError(
            "Parent curated column profiles are incomplete: " f"{missing_profiles}."
        )

    target_to_source = {target: source for source, target in complete_mapping.items()}
    profiler = ColProfiler()
    profiles: dict[str, ColProfile] = {}
    for target_name in target_df.columns:
        fresh_profile = profiler.get_col_profile(
            col_name=target_name,
            col_series=target_df[target_name],
        )
        source_name = target_to_source.get(target_name)
        if source_name is not None:
            fresh_profile = replace(
                fresh_profile,
                col_type=source_profiles[source_name].col_type,
            )
        profiles[target_name] = fresh_profile
    return profiles


def _rebuild_survival_profiles(
    *,
    target_df: pd.DataFrame,
    col_profiles: Mapping[str, ColProfile],
    source_survival_profiles: Mapping[str, SurvivalSemanticProfile],
    source_survival_pairs: SurvPairs,
    complete_mapping: Mapping[str, str],
) -> dict[str, SurvivalSemanticProfile]:
    target_to_source = {target: source for source, target in complete_mapping.items()}
    source_survival_columns = {
        source_name
        for source_name, target_name in complete_mapping.items()
        if col_profiles[target_name].col_type == DataTypes.SURVIVAL
    }
    paired_source_columns = {
        profile.col_name
        for pair in source_survival_pairs.pairs.values()
        for profile in (pair.event_profile, pair.time_profile)
    }
    missing = sorted(paired_source_columns - set(source_survival_profiles))
    if missing:
        raise CuratedStateInheritanceError(
            "Parent curated survival profiles are incomplete: " f"{missing}."
        )
    unexpected = sorted(set(source_survival_profiles) - source_survival_columns)
    if unexpected:
        raise CuratedStateInheritanceError(
            "Parent curated survival profiles contain stale columns: " f"{unexpected}."
        )

    profiles: dict[str, SurvivalSemanticProfile] = {}
    for source_name, source_profile in source_survival_profiles.items():
        if source_profile.col_type is None:
            raise CuratedStateInheritanceError(
                "Parent curated survival profile has no resolved role: "
                f"{source_name!r}."
            )
        target_name = complete_mapping.get(source_name)
        if target_name is None or target_name not in target_df.columns:
            raise CuratedStateInheritanceError(
                "Parent survival profile refers to a column outside the "
                f"curated parent dataframe: {source_name!r}."
            )
        fresh_profile = get_survival_semantic_col_profile(col_name=target_name)
        profiles[target_name] = replace(
            fresh_profile,
            col_type=source_profile.col_type,
        )

    for target_name, col_profile in col_profiles.items():
        if col_profile.col_type != DataTypes.SURVIVAL:
            continue
        if target_name not in profiles:
            source_name = target_to_source.get(target_name)
            if source_name is not None:
                # The parent intentionally has no semantic role for this
                # inherited survival-typed column. Preserve its datatype but
                # do not fabricate an endpoint profile.
                continue
            profile = get_survival_semantic_col_profile(col_name=target_name)
            if profile.col_type is None:
                raise CuratedStateInheritanceError(
                    "Target-only survival column has no inferred role: "
                    f"{target_name!r}."
                )
            profiles[target_name] = profile
    return profiles


def _validate_survival_values(
    *,
    target_df: pd.DataFrame,
    label: str,
    event_column: str,
    time_column: str,
) -> None:
    event_values = pd.to_numeric(target_df[event_column], errors="coerce")
    invalid_event = target_df[event_column].notna() & event_values.isna()
    if invalid_event.any():
        raise CuratedStateInheritanceError(
            f"Survival endpoint {label!r} has non-numeric event values in "
            f"{event_column!r}."
        )

    observed_events = set(event_values.dropna().astype(float).unique().tolist())
    if not observed_events <= {0.0, 1.0}:
        raise CuratedStateInheritanceError(
            f"Survival endpoint {label!r} requires binary 0/1 events in "
            f"{event_column!r}; found {sorted(observed_events)!r}."
        )

    time_values = pd.to_numeric(target_df[time_column], errors="coerce")
    invalid_time = target_df[time_column].notna() & time_values.isna()
    if invalid_time.any():
        raise CuratedStateInheritanceError(
            f"Survival endpoint {label!r} has non-numeric durations in "
            f"{time_column!r}."
        )
    non_missing_times = time_values.dropna()
    if not np.isfinite(non_missing_times.astype(float)).all():
        raise CuratedStateInheritanceError(
            f"Survival endpoint {label!r} has non-finite durations in "
            f"{time_column!r}."
        )
    if (non_missing_times < 0).any():
        raise CuratedStateInheritanceError(
            f"Survival endpoint {label!r} has negative durations in "
            f"{time_column!r}."
        )

    complete_cases = event_values.notna() & time_values.notna()
    if not complete_cases.any():
        raise CuratedStateInheritanceError(
            f"Survival endpoint {label!r} has no complete event/time rows."
        )


def _rebuild_survival_pairs(
    *,
    target_df: pd.DataFrame,
    source_pairs: SurvPairs,
    survival_profiles: Mapping[str, SurvivalSemanticProfile],
    complete_mapping: Mapping[str, str],
) -> SurvPairs:
    source_pairs.require_supported(operation="Curated-state inheritance")
    pairs: dict[str, SurvPair] = {}
    paired_columns: set[str] = set()
    for label, source_pair in source_pairs.pairs.items():
        if not str(label).strip():
            raise CuratedStateInheritanceError(
                "Parent survival pairs contain an empty endpoint label."
            )
        event_column = complete_mapping.get(source_pair.event_profile.col_name)
        time_column = complete_mapping.get(source_pair.time_profile.col_name)
        if event_column is None or time_column is None:
            raise CuratedStateInheritanceError(
                f"Survival endpoint {label!r} refers to an unmapped column."
            )
        if event_column == time_column:
            raise CuratedStateInheritanceError(
                f"Survival endpoint {label!r} maps event and time to the same "
                f"column {event_column!r}."
            )
        reused = paired_columns.intersection({event_column, time_column})
        if reused:
            raise CuratedStateInheritanceError(
                "A survival column cannot belong to multiple endpoints; "
                f"reused columns: {sorted(reused)}."
            )

        missing_profiles = [
            name
            for name in (event_column, time_column)
            if name not in survival_profiles
        ]
        if missing_profiles:
            raise CuratedStateInheritanceError(
                f"Survival endpoint {label!r} has missing target profiles: "
                f"{missing_profiles}."
            )

        event_profile = survival_profiles[event_column]
        time_profile = survival_profiles[time_column]
        if event_profile.col_type != SurvivalDataTypes.EVENT:
            raise CuratedStateInheritanceError(
                f"Inherited event column {event_column!r} is not typed as Event."
            )
        if time_profile.col_type != SurvivalDataTypes.TIME:
            raise CuratedStateInheritanceError(
                f"Inherited time column {time_column!r} is not typed as Time."
            )

        _validate_survival_values(
            target_df=target_df,
            label=label,
            event_column=event_column,
            time_column=time_column,
        )
        pairs[label] = replace(
            source_pair,
            surv_label=label,
            event_profile=event_profile,
            time_profile=time_profile,
        )
        paired_columns.update({event_column, time_column})
    return SurvPairs(pairs=pairs)


def build_inherited_curated_state(
    *,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_col_profiles: Mapping[str, ColProfile],
    source_survival_profiles: Mapping[str, SurvivalSemanticProfile],
    source_survival_pairs: SurvPairs,
    column_mapping: Mapping[str, str] | None,
    changed_columns: Collection[str],
    row_key: str | None,
    strict: bool,
) -> InheritedCuratedState:
    """Validate a derivation and rebuild target-specific semantic artifacts."""

    if not isinstance(strict, bool):
        raise CuratedStateInheritanceError("strict must be a boolean value.")
    if row_key is not None and (not isinstance(row_key, str) or not row_key):
        raise CuratedStateInheritanceError(
            "row_key must be None or a non-empty string."
        )

    source_columns = _validate_column_names(df=source_df, role="Parent")
    target_columns = _validate_column_names(df=target_df, role="Target")
    complete_mapping = _validate_column_mapping(
        source_columns=source_columns,
        target_columns=target_columns,
        column_mapping=column_mapping or {},
        strict=strict,
    )
    changed = _validate_declared_changes(
        source_df=source_df,
        target_df=target_df,
        complete_mapping=complete_mapping,
        changed_columns=changed_columns,
        row_key=row_key,
        strict=strict,
    )
    col_profiles = _rebuild_column_profiles(
        target_df=target_df,
        source_profiles=source_col_profiles,
        complete_mapping=complete_mapping,
    )
    survival_profiles = _rebuild_survival_profiles(
        target_df=target_df,
        col_profiles=col_profiles,
        source_survival_profiles=source_survival_profiles,
        source_survival_pairs=source_survival_pairs,
        complete_mapping=complete_mapping,
    )
    survival_pairs = _rebuild_survival_pairs(
        target_df=target_df,
        source_pairs=source_survival_pairs,
        survival_profiles=survival_profiles,
        complete_mapping=complete_mapping,
    )
    explicit_mapping = {
        source: target
        for source, target in complete_mapping.items()
        if source != target
    }
    return InheritedCuratedState(
        col_profiles=col_profiles,
        survival_profiles=survival_profiles,
        survival_pairs=survival_pairs,
        column_mapping=explicit_mapping,
        changed_columns=changed,
    )
