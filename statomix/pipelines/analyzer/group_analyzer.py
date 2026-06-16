import pandas as pd
from collections import defaultdict

from fileverse.formats.zarr import BaseZARR
from fileverse.formats.excel import BaseExcel

from statomix.pipelines.cleaner.surv.surv_report import SurvPairs
from statomix.pipelines.cleaner.col.col_report import ColReport, DataTypes

from statomix.analytics.datatypes.base.numerical import BaseNumerical
from statomix.analytics.datatypes.base.categorical import BaseCategorical

class GroupAnalyzer:
    def __init__(self, data_group):
        self.data_group = data_group
        self._create_paths()

    def _create_paths(self):
        self.paths = {}

        base_path = BaseZARR.get_abs_path(self.data_group)

        self.paths["df"] = base_path / "df.parquet"
        self.paths["surv_pairs"] = base_path / "surv_pairs.parquet"
        self.paths["col_profiles"] = base_path / "col_profiles.parquet"
        
    def _get_df(self):
        return pd.read_parquet(path=self.paths["df"])

    def _get_col_profiles(self):
        return ColReport.load_col_profiles(path=self.paths["col_profiles"])

    def _get_surv_pairs(self):
        if self.paths["surv_pairs"].exists():
            return SurvPairs.load(path=self.paths["surv_pairs"])
        else:
            error_msg = f"Survival Pairs do not exists for the given dataset"
            raise FileExistsError(error_msg)

    def _get_datatype_map(self):
        col_profiles = self._get_col_profiles()
    
        datatype_map = defaultdict(list)
        for profile in col_profiles.values():
            datatype_map[profile.col_type].append(profile.col_name)
    
        return datatype_map

    @staticmethod
    def get_cat_summary_df(df, col_names):
        distribution_dfs = []
        for col_name in col_names:
            series = df[col_name]
            distribution_df = BaseCategorical.get_distribution_df(series=series)
            distribution_df["col_name"] = col_name
            distribution_dfs.append(distribution_df)
        
        final_distribution_df = pd.concat(
            distribution_dfs,
            ignore_index=True,
        )
        
        final_distribution_df = final_distribution_df.set_index(
            ["col_name", "category"]
        )
        return final_distribution_df
    
    @staticmethod
    def get_num_summary_df(df, col_names):
        num_dicts = []
        for col_name in col_names:
            series = df[col_name]
            num_dicts.append(BaseNumerical.get_summary(series=series).to_dict())
        
        num_summary_df = pd.DataFrame(data=num_dicts)
    
        return num_summary_df

    def create_summary_report(self, path):
    
        df = self._get_df()
        datatype_map = self._get_datatype_map()
        
        cat_summary_df = self.get_cat_summary_df(df=df, col_names=datatype_map[DataTypes.CATEGORICAL])
        num_summary_df = self.get_num_summary_df(df=df, col_names=datatype_map[DataTypes.NUMERICAL])
    
        writer = pd.ExcelWriter(path=path, engine="openpyxl")
        num_summary_df.to_excel(excel_writer=writer, index=False, sheet_name="Numerical")
        cat_summary_df.to_excel(excel_writer=writer, index=True, sheet_name="Categorical")
        writer.close()
        
        BaseExcel.format_cell_length(path=path)