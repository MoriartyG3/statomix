from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .col_profiler import ColProfile
from .col_semantic_rules import DataTypes


@dataclass(frozen=True)
class DatatypeInventory:
    """Total mapping from every supported datatype to its curated columns."""

    columns_by_type: dict[DataTypes, tuple[str, ...]]

    @classmethod
    def from_profiles(
        cls,
        profiles: Mapping[str, ColProfile],
    ) -> DatatypeInventory:
        columns: dict[DataTypes, list[str]] = {datatype: [] for datatype in DataTypes}

        for profile in profiles.values():
            if profile.col_type is None:
                continue
            columns[profile.col_type].append(profile.col_name)

        return cls(
            columns_by_type={
                datatype: tuple(column_names)
                for datatype, column_names in columns.items()
            }
        )

    def columns(self, datatype: DataTypes) -> tuple[str, ...]:
        return self.columns_by_type[datatype]

    def count(self, datatype: DataTypes) -> int:
        return len(self.columns(datatype=datatype))

    def has(self, datatype: DataTypes) -> bool:
        return self.count(datatype=datatype) > 0

    def as_lists(self) -> dict[DataTypes, list[str]]:
        return {
            datatype: list(column_names)
            for datatype, column_names in self.columns_by_type.items()
        }

    def counts_by_name(self) -> dict[str, int]:
        return {
            datatype.value: len(column_names)
            for datatype, column_names in self.columns_by_type.items()
        }
