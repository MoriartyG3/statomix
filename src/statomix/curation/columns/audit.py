"""Immutable column-audit artifacts used by the Cleaner report."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

import pandas as pd

from statomix.core.tabular import frame_from_rows

from .profiler import ColProfile
from .semantic_rules import DataTypes


def _optional_float(value: object) -> float | None:
    """Convert a persisted optional numerical value."""

    if pd.isna(value):
        return None

    return float(value)


def _optional_int(value: object) -> int | None:
    """Convert a persisted optional integer value."""

    if pd.isna(value):
        return None

    return int(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class ColumnAuditProfile:
    """Read-only diagnostics for one source column."""

    PARQUET_SCHEMA: ClassVar[dict[str, str]] = {
        "col_name": "object",
        "inferred_datatype": "object",
        "source_dtype": "object",
        "missing_n": "int64",
        "missing_pct": "float64",
        "unique_n": "Int64",
        "num_conversion_pct": "float64",
        "numeric_n": "int64",
        "nonnumeric_n": "int64",
        "minimum": "float64",
        "q1": "float64",
        "median": "float64",
        "q3": "float64",
        "maximum": "float64",
        "exact_value_counts_included": "boolean",
    }

    col_name: str
    inferred_datatype: DataTypes | None
    source_dtype: str
    missing_n: int
    missing_pct: float
    unique_n: int | None
    num_conversion_pct: float | None
    numeric_n: int
    nonnumeric_n: int
    minimum: float | None
    q1: float | None
    median: float | None
    q3: float | None
    maximum: float | None
    exact_value_counts_included: bool

    def to_dict(self) -> dict[str, object]:
        """Convert the profile into a stable tabular record."""

        return {
            "col_name": self.col_name,
            "inferred_datatype": (
                self.inferred_datatype.value
                if self.inferred_datatype is not None
                else None
            ),
            "source_dtype": self.source_dtype,
            "missing_n": self.missing_n,
            "missing_pct": self.missing_pct,
            "unique_n": self.unique_n,
            "num_conversion_pct": self.num_conversion_pct,
            "numeric_n": self.numeric_n,
            "nonnumeric_n": self.nonnumeric_n,
            "minimum": self.minimum,
            "q1": self.q1,
            "median": self.median,
            "q3": self.q3,
            "maximum": self.maximum,
            "exact_value_counts_included": (self.exact_value_counts_included),
        }

    @classmethod
    def from_dict(
        cls,
        row: Mapping[str, object],
    ) -> ColumnAuditProfile:
        """Reconstruct a profile from a persisted record."""

        inferred_datatype = row["inferred_datatype"]

        return cls(
            col_name=str(row["col_name"]),
            inferred_datatype=(
                DataTypes(inferred_datatype) if pd.notna(inferred_datatype) else None
            ),
            source_dtype=str(row["source_dtype"]),
            missing_n=int(row["missing_n"]),
            missing_pct=float(row["missing_pct"]),
            unique_n=_optional_int(row["unique_n"]),
            num_conversion_pct=_optional_float(row["num_conversion_pct"]),
            numeric_n=int(row["numeric_n"]),
            nonnumeric_n=int(row["nonnumeric_n"]),
            minimum=_optional_float(row["minimum"]),
            q1=_optional_float(row["q1"]),
            median=_optional_float(row["median"]),
            q3=_optional_float(row["q3"]),
            maximum=_optional_float(row["maximum"]),
            exact_value_counts_included=bool(row["exact_value_counts_included"]),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ColumnValueFrequency:
    """Frequency of one observed categorical value."""

    PARQUET_SCHEMA: ClassVar[dict[str, str]] = {
        "col_name": "object",
        "inferred_datatype": "object",
        "value_display": "object",
        "value_type": "object",
        "is_missing": "boolean",
        "count": "int64",
        "percentage": "float64",
    }

    col_name: str
    inferred_datatype: DataTypes | None
    value_display: str
    value_type: str
    is_missing: bool
    count: int
    percentage: float

    def to_dict(self) -> dict[str, object]:
        """Convert the frequency into a stable tabular record."""

        return {
            "col_name": self.col_name,
            "inferred_datatype": (
                self.inferred_datatype.value
                if self.inferred_datatype is not None
                else None
            ),
            "value_display": self.value_display,
            "value_type": self.value_type,
            "is_missing": self.is_missing,
            "count": self.count,
            "percentage": self.percentage,
        }

    @classmethod
    def from_dict(
        cls,
        row: Mapping[str, object],
    ) -> ColumnValueFrequency:
        """Reconstruct a frequency from a persisted record."""

        inferred_datatype = row["inferred_datatype"]

        return cls(
            col_name=str(row["col_name"]),
            inferred_datatype=(
                DataTypes(inferred_datatype) if pd.notna(inferred_datatype) else None
            ),
            value_display=str(row["value_display"]),
            value_type=str(row["value_type"]),
            is_missing=bool(row["is_missing"]),
            count=int(row["count"]),
            percentage=float(row["percentage"]),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ColumnAudit:
    """Complete immutable audit for a source DataFrame."""

    profiles: Mapping[str, ColumnAuditProfile]
    value_frequencies: tuple[ColumnValueFrequency, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "profiles",
            MappingProxyType(dict(self.profiles)),
        )
        object.__setattr__(
            self,
            "value_frequencies",
            tuple(self.value_frequencies),
        )

    @classmethod
    def from_dataframe(
        cls,
        *,
        df: pd.DataFrame,
        col_profiles: Mapping[str, ColProfile],
    ) -> ColumnAudit:
        """Audit every column without modifying the DataFrame."""

        if not df.columns.is_unique:
            duplicate_columns = list(df.columns[df.columns.duplicated()])
            raise ValueError(
                "Column auditing requires unique column names. "
                f"Duplicates: {duplicate_columns}"
            )

        if list(df.columns) != list(col_profiles):
            raise ValueError(
                "DataFrame columns and persisted column profiles "
                "do not have identical names and order."
            )

        audit_profiles: dict[
            str,
            ColumnAuditProfile,
        ] = {}
        value_frequencies: list[ColumnValueFrequency] = []

        for col_name, profile in col_profiles.items():
            series = df[col_name]
            nonmissing_series = series.dropna()

            observed_missing_n = int(series.isna().sum())
            observed_unique_n = int(nonmissing_series.nunique())

            if observed_missing_n != profile.missing_n:
                raise ValueError(
                    f"Missing-count mismatch for {col_name!r}: "
                    f"profile={profile.missing_n}, "
                    f"source={observed_missing_n}"
                )

            if profile.unique_n is not None and observed_unique_n != profile.unique_n:
                raise ValueError(
                    f"Unique-count mismatch for {col_name!r}: "
                    f"profile={profile.unique_n}, "
                    f"source={observed_unique_n}"
                )

            if profile.num_conversion_pct is None:
                numeric_series = pd.Series(dtype="float64")
            else:
                numeric_series = pd.to_numeric(
                    nonmissing_series,
                    errors="coerce",
                )

            valid_numeric_series = numeric_series.dropna()
            numeric_n = int(valid_numeric_series.size)
            nonnumeric_n = int(nonmissing_series.size - numeric_n)

            if numeric_n == 0:
                minimum = None
                q1 = None
                median = None
                q3 = None
                maximum = None
            else:
                quantiles = valid_numeric_series.quantile(
                    [
                        0.00,
                        0.25,
                        0.50,
                        0.75,
                        1.00,
                    ]
                )

                minimum = float(quantiles.loc[0.00])
                q1 = float(quantiles.loc[0.25])
                median = float(quantiles.loc[0.50])
                q3 = float(quantiles.loc[0.75])
                maximum = float(quantiles.loc[1.00])

            include_exact_counts = profile.col_type == DataTypes.CATEGORICAL

            audit_profiles[col_name] = ColumnAuditProfile(
                col_name=col_name,
                inferred_datatype=(profile.col_type),
                source_dtype=str(series.dtype),
                missing_n=observed_missing_n,
                missing_pct=profile.missing_pct,
                unique_n=profile.unique_n,
                num_conversion_pct=(profile.num_conversion_pct),
                numeric_n=numeric_n,
                nonnumeric_n=nonnumeric_n,
                minimum=minimum,
                q1=q1,
                median=median,
                q3=q3,
                maximum=maximum,
                exact_value_counts_included=(include_exact_counts),
            )

            if not include_exact_counts:
                continue

            nonmissing_counts = nonmissing_series.value_counts(
                dropna=False,
                sort=True,
            )

            for observed_value, count in nonmissing_counts.items():
                if int(count) == 0:
                    continue

                value_frequencies.append(
                    ColumnValueFrequency(
                        col_name=col_name,
                        inferred_datatype=(profile.col_type),
                        value_display=repr(observed_value),
                        value_type=(
                            f"{type(observed_value).__module__}."
                            f"{type(observed_value).__qualname__}"
                        ),
                        is_missing=False,
                        count=int(count),
                        percentage=round(
                            (int(count) / len(series)) * 100,
                            6,
                        ),
                    )
                )

            if observed_missing_n > 0:
                value_frequencies.append(
                    ColumnValueFrequency(
                        col_name=col_name,
                        inferred_datatype=(profile.col_type),
                        value_display="<MISSING>",
                        value_type="missing",
                        is_missing=True,
                        count=observed_missing_n,
                        percentage=round(
                            (observed_missing_n / len(series)) * 100,
                            6,
                        ),
                    )
                )

        return cls(
            profiles=audit_profiles,
            value_frequencies=tuple(value_frequencies),
        )

    def save(
        self,
        *,
        profiles_path: Path,
        value_frequencies_path: Path,
    ) -> None:
        """Persist both machine-readable audit artifacts."""

        profile_rows = [profile.to_dict() for profile in self.profiles.values()]
        frequency_rows = [frequency.to_dict() for frequency in self.value_frequencies]

        profiles_df = frame_from_rows(
            rows=profile_rows,
            schema=ColumnAuditProfile.PARQUET_SCHEMA,
        )
        frequencies_df = frame_from_rows(
            rows=frequency_rows,
            schema=(ColumnValueFrequency.PARQUET_SCHEMA),
        )

        profiles_df.to_parquet(
            path=profiles_path,
            index=False,
        )
        frequencies_df.to_parquet(
            path=value_frequencies_path,
            index=False,
        )

    @classmethod
    def load(
        cls,
        *,
        profiles_path: Path,
        value_frequencies_path: Path,
    ) -> ColumnAudit:
        """Load a complete persisted audit."""

        profiles_df = pd.read_parquet(profiles_path)
        frequencies_df = pd.read_parquet(value_frequencies_path)

        profiles = {
            profile.col_name: profile
            for profile in (
                ColumnAuditProfile.from_dict(row)
                for row in profiles_df.to_dict(orient="records")
            )
        }

        value_frequencies = tuple(
            ColumnValueFrequency.from_dict(row)
            for row in frequencies_df.to_dict(orient="records")
        )

        return cls(
            profiles=profiles,
            value_frequencies=value_frequencies,
        )
