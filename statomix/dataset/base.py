import pandas as pd

from fileverse.logger import Logger
from fileverse.formats.zarr import BaseZARR

from statomix.reports.col_report import ColReport

logger = Logger(name="BaseDataset").get_logger()

class BaseDataset:
    def __init__(self, dataset_name:str, root_group, df:pd.DataFrame=None):
        
        self.dataset_name = dataset_name
        self.col_report = ColReport()

        self._create_groups(root_group=root_group)
        self._create_source_df(df=df)

    def _create_groups(self, root_group):
        self.zarr_groups = {}
        self.zarr_groups['root'] = root_group.require_group(self.dataset_name)
        
        self.zarr_groups['df'] = self.zarr_groups['root'].require_group('df')
        if 'source_df_exists' not in self.zarr_groups['df'].attrs:
            self.zarr_groups['df'].attrs['source_df_exists'] =  False
        
        self.zarr_groups['report'] =  self.zarr_groups['root'].require_group('report')

        if 'col_report_exists' not in self.zarr_groups['report'].attrs:
            self.zarr_groups['report'].attrs['col_report_exists'] =  False
            self.zarr_groups['report'].attrs['col_report_version'] = 0
        
    def get_source_df(self):
        source_df_path =  BaseZARR.get_abs_path(zarr_group=self.zarr_groups['df'])/"source_df.parquet"

        if source_df_path.exists():
            return pd.read_parquet(path=source_df_path)
        else:
            logger.error(f'source_df does not exist at {source_df_path}')
            return
            
    def _create_source_df(self, df):
        source_df_path =  BaseZARR.get_abs_path(zarr_group=self.zarr_groups['df'])/"source_df.parquet"

        if source_df_path.exists():
            if df is not None:
                logger.warning(f"source_df already exists. The provided DataFrame was NOT saved to avoid overwriting.")
            # else:
            #     logger.info(f"source_df already exists. Loaded from disk storage.")
            return  # Exit early since the file is safely there

        if df is not None:
            df.to_parquet(path=source_df_path, index=False)
            self.zarr_groups['df'].attrs['source_df_exists'] = True
            logger.info(f"Successfully created and saved new source_df.")
            return

        error_msg = f"source_df doesn't exist at {source_df_path} and provided df is None."
        logger.error(msg=error_msg)
        raise ValueError(error_msg)