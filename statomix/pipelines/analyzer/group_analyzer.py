import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict

#from fileverse.formats.zarr import BaseZARR
from fileverse.formats.excel import BaseExcel

from statomix.pipelines.cleaner.surv.surv_report import SurvPairs
from statomix.pipelines.cleaner.col.col_report import ColReport, DataTypes

from statomix.analytics.datatypes.numerical.base import BaseNumerical
from statomix.analytics.datatypes.numerical.normality import Normality
from statomix.analytics.datatypes.categorical.base import BaseCategorical

class GroupAnalyzer:
    def __init__(self, paths: dict[str, Path]):
        self.paths = paths
        self.cache = {}
        
    # def __init__(self, data_group):
    #     self.data_group = data_group
    #     self._create_paths()

    # def _create_paths(self):
    #     self.paths = {}

    #     base_path = BaseZARR.get_abs_path(self.data_group)

    #     self.paths["df"] = base_path / "df.parquet"
    #     self.paths["surv_pairs"] = base_path / "surv_pairs.parquet"
    #     self.paths["col_profiles"] = base_path / "col_profiles.parquet"
        
    def _get_df(self):
        if "df" not in self.cache:
            self.cache["df"] = pd.read_parquet(path=self.paths["df"])
            
        return self.cache["df"]
        
    def _get_col_profiles(self):
        if "col_profiles" not in self.cache:
            self.cache["col_profiles"] = ColReport.load_col_profiles(path=self.paths["col_profiles"])

        return self.cache["col_profiles"]

    def _get_surv_pairs(self):
        if not self.paths["surv_pairs"].exists():
            raise FileNotFoundError(
                "Survival pairs do not exist for the given dataset."
            )
    
        if "surv_pairs" not in self.cache:
            self.cache["surv_pairs"] = SurvPairs.load(
                path=self.paths["surv_pairs"]
            )
        return self.cache["surv_pairs"]
        
    def _get_datatype_map(self):
        if "datatype_map" not in self.cache:
            
            col_profiles = self._get_col_profiles()
        
            datatype_map = defaultdict(list)
            for profile in col_profiles.values():
                datatype_map[profile.col_type].append(profile.col_name)

            self.cache["datatype_map"] =  datatype_map
            
        return self.cache["datatype_map"]

    def _get_datatype_map_df(self):
        datatype_map = self._get_datatype_map()

        datatype_df = pd.DataFrame({
            k.value: pd.Series(v)
            for k, v in datatype_map.items()
        })

        surv_pairs = self._get_surv_pairs()
        datatype_df['Survival Labels'] = pd.Series(list(surv_pairs.pairs.keys()))

        return datatype_df

    def get_cat_summary_df(self):
        
        df = self._get_df()
        datatype_map = self._get_datatype_map()
        col_names=datatype_map[DataTypes.CATEGORICAL]
        
        distribution_dfs = []
        for col_name in col_names:
            series = df[col_name]
            distribution_df = BaseCategorical.get_distribution_df(series=series)
            distribution_df["col_name"] = col_name
            distribution_dfs.append(distribution_df)
        if not distribution_dfs:
            return (
                pd.DataFrame(
                    columns=["count", "percentage"]
                )
                .set_index(
                    pd.MultiIndex.from_arrays(
                        [[], []],
                        names=["col_name", "category"],
                    )
                )
            )
        
        final_distribution_df = pd.concat(
            distribution_dfs,
            ignore_index=True,
        )
        
        final_distribution_df = final_distribution_df.set_index(
            ["col_name", "category"]
        )
        return final_distribution_df
    
    def get_num_summary_df(self):
        
        df = self._get_df()
        datatype_map = self._get_datatype_map()
        col_names=datatype_map[DataTypes.NUMERICAL]
        
        num_dicts = []
        for col_name in col_names:
            series = df[col_name]
            num_dict = BaseNumerical.get_summary(series=series)
            num_dict['name'] = series.name
            num_dicts.append(num_dict)
            
        if not num_dicts:
            return (
                pd.DataFrame()
                .rename_axis("name")
            )
            
        num_summary_df = pd.DataFrame(data=num_dicts).set_index('name')
    
        return num_summary_df

    def create_summary_report(self, path):
    
        
        
        cat_summary_df = self.get_cat_summary_df()
        num_summary_df = self.get_num_summary_df()
        normality_diagnostics_df = self.get_normality_diagnostics_df()
    
        #writer = pd.ExcelWriter(path=path, engine="openpyxl")
        with pd.ExcelWriter(path=path, engine="openpyxl") as writer:
            num_summary_df.to_excel(excel_writer=writer, index=True, sheet_name="Numerical")
            normality_diagnostics_df.to_excel(excel_writer=writer, index=True, sheet_name="Normality Diagnostics")
            cat_summary_df.to_excel(excel_writer=writer, index=True, sheet_name="Categorical")
        
        BaseExcel.format_cell_length(path=path)

    def get_normality_diagnostics_df(self, progress_bar=False, alpha=0.05, ddof=1):

        df = self._get_df()
        datatype_map = self._get_datatype_map()
        col_names = datatype_map[DataTypes.NUMERICAL]
        
        full_diagnostics_list = []
        
        if progress_bar:
            iterator = tqdm(col_names)
        else:
            iterator = col_names
            
        for col_name in iterator:
            series = df[col_name]
            norm =  Normality(series=series, alpha=alpha, ddof=ddof)
            
            full_diagnostics = norm.get_full_diagnostics()
            full_diagnostics_dict = pd.json_normalize(data=full_diagnostics).iloc[0].to_dict()

            test_for_purpose = norm.recommend_test_for_purpose('parametric_test')
            normality_report  = norm.get_normality_report(test_type=test_for_purpose['recommended_test'])
            #normality_report = norm.get_normality_report_default()
            normality_report.update(full_diagnostics_dict)
            normality_report['name'] =  series.name
            
            full_diagnostics_list.append(normality_report)
        
        full_diagnostics_df = pd.DataFrame(full_diagnostics_list).set_index(["name"])

        #Moving Columns to end for better visuals
        cols_to_move = ["power.note", "outliers.outlier_values"]
        full_diagnostics_df = full_diagnostics_df[
            [c for c in full_diagnostics_df.columns if c not in cols_to_move]
            + cols_to_move
        ]
    
        return full_diagnostics_df