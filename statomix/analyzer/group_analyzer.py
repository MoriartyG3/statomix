import pandas as pd

from fileverse.formats.zarr import BaseZARR

from statomix.cleaner.col.col_report import ColReport
from statomix.cleaner.surv.surv_report import SurvPairs

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
        return ColReport.load_col_profiles(profiles_path=self.paths["col_profiles"])

    def _get_surv_pairs(self):
        if self.paths["surv_pairs"].exists():
            return SurvPairs.load(path=self.paths["surv_pairs"])
        else:
            error_msg = f"Survival Pairs do not exists for the given dataset"
            raise FileExistsError(error_msg)