"""Stable, typed contracts shared by Statomix modules.

The public workflow still exposes legacy dictionaries where it did before.
Internally, these immutable objects make version/configuration selection and
procedure state explicit and prevent misspelled string keys from silently
crossing module boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True, kw_only=True)
class VersionRef:
    """Resolved pipeline version identifier."""

    version: int

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("version must be at least 1")


@dataclass(frozen=True, slots=True, kw_only=True)
class ConfigRef:
    """Resolved configuration identifier within a pipeline version."""

    version: int

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("config version must be at least 1")


@dataclass(frozen=True, slots=True, kw_only=True)
class GroupInfo:
    """One resolved Zarr group and its filesystem representation."""

    group: Any
    path: Path
    meta: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return the historical dictionary shape used by public workflows."""

        return {
            "group": self.group,
            "path": self.path,
            "meta": dict(self.meta),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class GroupBundle:
    """Resolved version and configuration groups."""

    version: GroupInfo
    config: GroupInfo

    def to_legacy_dict(self) -> dict[str, dict[str, Any]]:
        """Return the exact historical nested-dictionary contract."""

        return {
            "version": self.version.to_legacy_dict(),
            "config": self.config.to_legacy_dict(),
        }


class ProcedureState(StrEnum):
    """Lifecycle states persisted for a Cleaner/Analyzer procedure."""

    PENDING = "pending"
    COMPLETED = "completed"
    NOT_APPLICABLE = "not_applicable"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, kw_only=True)
class ProcedureStatus:
    """Auditable result of one workflow procedure."""

    status: ProcedureState
    reason: str
    input_count: int
    output_count: int

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("reason must not be empty")
        if self.input_count < 0 or self.output_count < 0:
            raise ValueError("procedure counts must be non-negative")

    def to_dict(self) -> dict[str, str | int]:
        """Serialize to the existing Zarr metadata representation."""

        return {
            "status": self.status.value,
            "reason": self.reason,
            "input_count": self.input_count,
            "output_count": self.output_count,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalyzerInputPaths:
    """Validated paths consumed by a dataset analyzer."""

    df: Path
    surv_pairs: Path
    col_profiles: Path

    def __post_init__(self) -> None:
        for field_name in ("df", "surv_pairs", "col_profiles"):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)))

    @classmethod
    def from_mapping(cls, paths: Mapping[str, str | Path]) -> AnalyzerInputPaths:
        missing = {"df", "surv_pairs", "col_profiles"} - set(paths)
        if missing:
            raise ValueError(f"Analyzer paths are missing keys: {sorted(missing)}")
        return cls(
            df=Path(paths["df"]),
            surv_pairs=Path(paths["surv_pairs"]),
            col_profiles=Path(paths["col_profiles"]),
        )

    def as_dict(self, *, stringify: bool = False) -> dict[str, Path | str]:
        values: dict[str, Path] = {
            "df": self.df,
            "surv_pairs": self.surv_pairs,
            "col_profiles": self.col_profiles,
        }
        if stringify:
            return {key: str(value) for key, value in values.items()}
        return values


@dataclass(frozen=True, slots=True, kw_only=True)
class CuratedStateLineage:
    """Immutable provenance for a dataset derived from curated parent data."""

    source_project: str
    source_dataset: str
    source_version: int
    source_config_version: int
    target_dataset: str
    target_version: int
    target_config_version: int
    column_mapping: Mapping[str, str]
    changed_columns: tuple[str, ...]
    row_key: str | None
    strict: bool
    applied_parent_category_edits: bool
    source_artifact_sha256: Mapping[str, str]
    target_source_df_sha256: str

    def __post_init__(self) -> None:
        for field_name in ("source_project", "source_dataset", "target_dataset"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must not be empty")

        for field_name in (
            "source_version",
            "source_config_version",
            "target_version",
            "target_config_version",
        ):
            if int(getattr(self, field_name)) < 1:
                raise ValueError(f"{field_name} must be at least 1")

        mapping = dict(self.column_mapping)
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("column_mapping target names must be unique")
        if not isinstance(self.strict, bool):
            raise ValueError("strict must be a boolean value")
        if not isinstance(self.applied_parent_category_edits, bool):
            raise ValueError("applied_parent_category_edits must be a boolean value")

        object.__setattr__(self, "column_mapping", MappingProxyType(mapping))
        object.__setattr__(self, "changed_columns", tuple(self.changed_columns))
        object.__setattr__(
            self,
            "source_artifact_sha256",
            MappingProxyType(dict(self.source_artifact_sha256)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a Zarr-attribute-compatible representation."""

        return {
            "kind": "curated_state_inheritance",
            "source": {
                "project": self.source_project,
                "dataset": self.source_dataset,
                "version": self.source_version,
                "config_version": self.source_config_version,
                "artifact_sha256": dict(self.source_artifact_sha256),
            },
            "target": {
                "dataset": self.target_dataset,
                "version": self.target_version,
                "config_version": self.target_config_version,
                "source_df_sha256": self.target_source_df_sha256,
            },
            "column_mapping": dict(self.column_mapping),
            "changed_columns": list(self.changed_columns),
            "row_key": self.row_key,
            "strict": self.strict,
            "applied_parent_category_edits": (self.applied_parent_category_edits),
        }
