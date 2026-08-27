from pathlib import Path

import pandas as pd
from fileverse.formats.excel import BaseExcel
from tqdm.auto import tqdm

from statomix.analytics.datatypes.categorical.base import BaseCategorical
from statomix.analytics.datatypes.numerical.base import BaseNumerical
from statomix.analytics.datatypes.numerical.normality import Normality
from statomix.pipelines.cleaner.col.col_report import ColReport
from statomix.pipelines.cleaner.col.col_semantic_rules import DataTypes
from statomix.pipelines.cleaner.col.datatype_inventory import (
    DatatypeInventory,
)
from statomix.pipelines.cleaner.surv.surv_report import SurvPairs

from .contracts import (
    empty_categorical_summary,
    empty_normality_diagnostics,
    empty_numerical_summary,
)


class GroupAnalyzer:
    def __init__(self, paths: dict[str, Path]):
        self.paths = paths
        self.cache = {}

    def _get_df(self):
        if "df" not in self.cache:
            self.cache["df"] = pd.read_parquet(path=self.paths["df"])

        return self.cache["df"]

    def _get_col_profiles(self):
        if "col_profiles" not in self.cache:
            self.cache["col_profiles"] = ColReport.load_col_profiles(
                path=self.paths["col_profiles"]
            )

        return self.cache["col_profiles"]

    def _get_surv_pairs(self):
        if not self.paths["surv_pairs"].exists():
            raise FileNotFoundError(
                "Survival pairs do not exist for the given dataset."
            )

        if "surv_pairs" not in self.cache:
            self.cache["surv_pairs"] = SurvPairs.load(path=self.paths["surv_pairs"])

        return self.cache["surv_pairs"]

    def _get_datatype_map(self):
        if "datatype_map" not in self.cache:
            col_profiles = self._get_col_profiles()
            inventory = DatatypeInventory.from_profiles(profiles=col_profiles)
            self.cache["datatype_map"] = inventory.as_lists()

        return self.cache["datatype_map"]

    def _get_datatype_map_df(self):
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

        surv_pairs = self._get_surv_pairs()
        datatype_df["Survival Labels"] = pd.Series(
            list(surv_pairs.pairs),
            dtype="object",
        )

        return datatype_df

    def get_cat_summary_df(self):
        df = self._get_df()
        datatype_map = self._get_datatype_map()
        col_names = datatype_map[DataTypes.CATEGORICAL]

        distribution_dfs = []
        for col_name in col_names:
            series = df[col_name]
            distribution_df = BaseCategorical.get_distribution_df(series=series)
            distribution_df["col_name"] = col_name
            distribution_dfs.append(distribution_df)

        if not distribution_dfs:
            return empty_categorical_summary()

        final_distribution_df = pd.concat(
            distribution_dfs,
            ignore_index=True,
        )
        return final_distribution_df.set_index(["col_name", "category"])

    def get_num_summary_df(self):
        df = self._get_df()
        datatype_map = self._get_datatype_map()
        col_names = datatype_map[DataTypes.NUMERICAL]

        num_dicts = []
        for col_name in col_names:
            series = df[col_name]
            num_dict = BaseNumerical.get_summary(series=series)
            num_dict["name"] = series.name
            num_dicts.append(num_dict)

        if not num_dicts:
            return empty_numerical_summary()

        return pd.DataFrame(data=num_dicts).set_index("name")

    def create_summary_report(self, path):
        cat_summary_df = self.get_cat_summary_df()
        num_summary_df = self.get_num_summary_df()
        normality_diagnostics_df = self.get_normality_diagnostics_df()

        with pd.ExcelWriter(path=path, engine="openpyxl") as writer:
            num_summary_df.to_excel(
                excel_writer=writer,
                index=True,
                sheet_name="Numerical",
            )
            normality_diagnostics_df.to_excel(
                excel_writer=writer,
                index=True,
                sheet_name="Normality Diagnostics",
            )
            cat_summary_df.to_excel(
                excel_writer=writer,
                index=True,
                sheet_name="Categorical",
            )

        BaseExcel.format_cell_length(path=path)

    def get_normality_diagnostics_df(
        self,
        progress_bar=False,
        alpha=0.05,
        ddof=1,
    ):
        df = self._get_df()
        datatype_map = self._get_datatype_map()
        col_names = datatype_map[DataTypes.NUMERICAL]

        iterator = tqdm(col_names) if progress_bar else col_names
        full_diagnostics_list = []

        for col_name in iterator:
            series = df[col_name]
            normality = Normality(
                series=series,
                alpha=alpha,
                ddof=ddof,
            )

            full_diagnostics = normality.get_full_diagnostics()
            full_diagnostics_dict = (
                pd.json_normalize(data=full_diagnostics).iloc[0].to_dict()
            )

            test_for_purpose = normality.recommend_test_for_purpose("parametric_test")
            normality_report = normality.get_normality_report(
                test_type=test_for_purpose["recommended_test"]
            )
            normality_report.update(full_diagnostics_dict)
            normality_report["name"] = series.name

            full_diagnostics_list.append(normality_report)

        if not full_diagnostics_list:
            return empty_normality_diagnostics()

        full_diagnostics_df = pd.DataFrame(full_diagnostics_list).set_index(["name"])

        cols_to_move = ["power.note", "outliers.outlier_values"]
        return full_diagnostics_df[
            [
                column
                for column in full_diagnostics_df.columns
                if column not in cols_to_move
            ]
            + cols_to_move
        ]
