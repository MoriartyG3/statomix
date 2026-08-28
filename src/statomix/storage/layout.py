"""Canonical names and path construction for the persisted Statomix layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True, kw_only=True)
class StatomixLayout:
    """Construct existing paths without owning or mutating storage."""

    root: Path

    SOURCE_DF = "source_df.parquet"
    GROUP_ANALYZER_PATHS = "group_analyzer_path.yaml"
    SUMMARY_REPORT = "summary.xlsx"
    ANALYSIS_CONFIG_TEMPLATE = "analysis_config_version{version}.xlsx"

    CURATED_GROUP = "curated_data"
    CURATED_DF = "df.parquet"
    CURATED_SURV_PAIRS = "surv_pairs.parquet"
    CURATED_COL_PROFILES = "col_profiles.parquet"

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @staticmethod
    def version_group(*, version: int) -> str:
        if version < 1:
            raise ValueError("version must be at least 1")
        return f"version{version}"

    @staticmethod
    def config_group(*, config_version: int) -> str:
        if config_version < 1:
            raise ValueError("config_version must be at least 1")
        return f"config{config_version}"

    def source_df(self) -> Path:
        return self.root / self.SOURCE_DF

    def group_analyzer_paths(self) -> Path:
        return self.root / self.GROUP_ANALYZER_PATHS

    def summary_report(self) -> Path:
        return self.root / self.SUMMARY_REPORT

    def analysis_config(self, *, version: int) -> Path:
        if version < 1:
            raise ValueError("analysis configuration version must be at least 1")
        return self.root / self.ANALYSIS_CONFIG_TEMPLATE.format(version=version)

    def curated_artifacts(self) -> dict[str, Path]:
        curated_root = self.root / self.CURATED_GROUP
        return {
            "df": curated_root / self.CURATED_DF,
            "surv_pairs": curated_root / self.CURATED_SURV_PAIRS,
            "col_profiles": curated_root / self.CURATED_COL_PROFILES,
        }
