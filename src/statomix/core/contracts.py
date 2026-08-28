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
