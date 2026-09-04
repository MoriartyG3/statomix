"""Dataset analysis service over one resolved set of curated artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from statomix.analytics.datatypes.categorical import BaseCategorical
from statomix.analytics.datatypes.numerical import BaseNumerical, Normality
from statomix.core.analysis_results import (
    empty_categorical_summary,
    empty_normality_diagnostics,
    empty_numerical_summary,
)
from statomix.core.contracts import AnalyzerInputPaths
from statomix.curation.columns import (
    ColReport,
    DatatypeInventory,
    DataTypes,
)
from statomix.curation.survival import SurvPairs
from statomix.reporting.excel.summary import (
    SummaryWorkbook,
    SummaryWorkbookRenderer,
)


class GroupAnalyzer:
    """Compute summaries without owning the Analyzer's Zarr hierarchy."""

    def __init__(self, paths: dict[str, Path] | AnalyzerInputPaths) -> None:
        self.input_paths = (
            paths
            if isinstance(paths, AnalyzerInputPaths)
            else AnalyzerInputPaths.from_mapping(paths)
        )
        self.paths = self.input_paths.as_dict()
        self.cache: dict[str, Any] = {}

    def _get_df(self) -> pd.DataFrame:
        if "df" not in self.cache:
            self.cache["df"] = pd.read_parquet(path=self.input_paths.df)
        return self.cache["df"]

    def _get_col_profiles(self) -> dict:
        if "col_profiles" not in self.cache:
            self.cache["col_profiles"] = ColReport.load_col_profiles(
                path=self.input_paths.col_profiles
            )
        return self.cache["col_profiles"]

    def _get_surv_pairs(self) -> SurvPairs:
        if not self.input_paths.surv_pairs.exists():
            raise FileNotFoundError(
                "Survival pairs do not exist for the given dataset at "
                f"{self.input_paths.surv_pairs}."
            )

        if "surv_pairs" not in self.cache:
            self.cache["surv_pairs"] = SurvPairs.load(path=self.input_paths.surv_pairs)

        pairs = self.cache["surv_pairs"]
        pairs.require_supported(operation="GroupAnalyzer")
        return pairs

    def _get_datatype_map(self) -> dict[DataTypes, list[str]]:
        if "datatype_map" not in self.cache:
            inventory = DatatypeInventory.from_profiles(
                profiles=self._get_col_profiles()
            )
            self.cache["datatype_map"] = inventory.as_lists()
        return self.cache["datatype_map"]

    def _get_datatype_map_df(self) -> pd.DataFrame:
        datatype_map = self._get_datatype_map()
        datatype_df = pd.DataFrame(
            {
                datatype.value: pd.Series(
                    datatype_map[datatype],
                    dtype="object",
                )
                for datatype in DataTypes
            }
        )
        datatype_df["Survival Labels"] = pd.Series(
            list(self._get_surv_pairs().pairs),
            dtype="object",
        )
        return datatype_df

    def get_cat_summary_df(self) -> pd.DataFrame:
        df = self._get_df()
        col_names = self._get_datatype_map()[DataTypes.CATEGORICAL]
        distributions: list[pd.DataFrame] = []
        for col_name in col_names:
            distribution = BaseCategorical.get_distribution_df(series=df[col_name])
            distribution["col_name"] = col_name
            distributions.append(distribution)

        if not distributions:
            return empty_categorical_summary()
        return pd.concat(distributions, ignore_index=True).set_index(
            ["col_name", "category"]
        )

    def get_num_summary_df(self) -> pd.DataFrame:
        df = self._get_df()
        col_names = self._get_datatype_map()[DataTypes.NUMERICAL]
        summaries: list[dict[str, Any]] = []
        for col_name in col_names:
            series = df[col_name]
            summary = BaseNumerical.get_summary(series=series)
            summary["name"] = series.name
            summaries.append(summary)

        if not summaries:
            return empty_numerical_summary()
        return pd.DataFrame(data=summaries).set_index("name")

    def get_normality_diagnostics_df(
        self,
        progress_bar: bool = False,
        alpha: float = 0.05,
        ddof: int = 1,
    ) -> pd.DataFrame:
        df = self._get_df()
        col_names = self._get_datatype_map()[DataTypes.NUMERICAL]
        iterator = tqdm(col_names) if progress_bar else col_names
        diagnostics: list[dict[str, Any]] = []

        for col_name in iterator:
            series = df[col_name]
            normality = Normality(series=series, alpha=alpha, ddof=ddof)
            flattened = (
                pd.json_normalize(data=normality.get_full_diagnostics())
                .iloc[0]
                .to_dict()
            )
            recommended = normality.recommend_test_for_purpose(
                purpose="parametric_test"
            )
            report = normality.get_normality_report(
                test_type=recommended["recommended_test"]
            )
            report.update(flattened)
            report["name"] = series.name
            diagnostics.append(report)

        if not diagnostics:
            return empty_normality_diagnostics()

        frame = pd.DataFrame(diagnostics).set_index(["name"])
        trailing = ["power.note", "outliers.outlier_values"]
        return frame[
            [column for column in frame.columns if column not in trailing] + trailing
        ]

    def create_summary_report(self, path: Path) -> None:
        """Compatibility method delegating presentation to a renderer."""

        SummaryWorkbookRenderer.render(
            workbook=SummaryWorkbook(
                numerical=self.get_num_summary_df(),
                normality=self.get_normality_diagnostics_df(),
                categorical=self.get_cat_summary_df(),
            ),
            path=Path(path),
        )
