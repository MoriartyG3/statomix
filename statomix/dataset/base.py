import pandas as pd
from pandas.testing import assert_frame_equal

from fileverse.logger import Logger
from fileverse.formats.zarr import BaseZARR

from statomix.reports.col_report import ColReport

logger = Logger(name="BaseDataset").get_logger()

class BaseDataset:
    def __init__(self, dataset_name:str, root_group, df:pd.DataFrame=None):
        
        self.dataset_name = dataset_name
        self.col_report = ColReport()

        self._create_groups(root_group=root_group)
        self._create_paths()
        
        self._create_source_df(df=df)

    def _create_groups(self, root_group):
        self.zarr_groups = {}
        self.zarr_groups['root'] = root_group.require_group(self.dataset_name)
        
        self.zarr_groups['df'] = self.zarr_groups['root'].require_group('df')
        if 'source_df_exists' not in self.zarr_groups['df'].attrs:
            self.zarr_groups['df'].attrs['source_df_exists'] =  False
        
        self.zarr_groups['col_report'] =  self.zarr_groups['root'].require_group('col_report')

        # col_report_meta = self.zarr_groups['col_report'].attrs.get("col_report", {})

        # if 'default' not in col_report_meta:
        #     col_report_meta['default'] =  {}
        #     col_report_meta['default']['exists'] = False
        #     col_report_meta['default']['version'] = 0

        #     self.zarr_groups['col_report'].attrs['col_report'] = col_report_meta

    def _create_paths(self):
        self.paths = {}
        self.paths['source_df'] = BaseZARR.get_abs_path(zarr_group=self.zarr_groups['df'])/"source_df.parquet"

        #Col Reports
        # col_report_meta = self.zarr_groups["col_report"].attrs["col_report"]
        # version = col_report_meta['default']['version']

        # self.paths["col_report"] = {}
        # self.paths["col_report"]["default"]=BaseZARR.get_abs_path(zarr_group=self.zarr_groups['col_report'])/ f"col_report_version{version}.xlsx"
    
        
    def get_source_df(self):
        source_df_path =  self.paths['source_df']

        if source_df_path.exists():
            return pd.read_parquet(path=source_df_path)
        else:
            logger.error(f'source_df does not exist at {source_df_path}')
            return
            
    def _create_source_df(self, df:pd.DataFrame|None):
        source_df_path =  self.paths['source_df']

        if source_df_path.exists():
            
            if df is not None:
                
                existing_df = pd.read_parquet(source_df_path)

                try:
                    assert_frame_equal(left=existing_df, right=df)
                    logger.warning(f"source_df already exists. The provided DataFrame was NOT saved to avoid overwriting.")
                except AssertionError as e:
                    logger.warning(f"source_df already exists. However, the provided DataFrame is NOT identical to the saved DataFrame.")
                    logger.debug(str(e))
                    
            return  

        if df is not None:
            df.to_parquet(path=source_df_path, index=False)
            self.zarr_groups['df'].attrs['source_df_exists'] = True
            logger.info(f"Successfully created and saved new source_df.")
            return

        error_msg = f"source_df doesn't exist at {source_df_path} and provided df is None."
        logger.error(msg=error_msg)
        raise ValueError(error_msg)