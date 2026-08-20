import pandas as pd
from pandas.testing import assert_frame_equal

from fileverse.logger import Logger
from fileverse.formats.zarr import BaseZARR

logger = Logger(name="BaseDataset").get_logger()

class BaseDataset:
    def __init__(self, dataset_name, root_group, df:pd.DataFrame = None):
        self.dataset_name = dataset_name
        self._create_groups(root_group=root_group)
        self._create_paths()
        self._create_source_df(df=df)

    def _create_groups(self, root_group):
    
        self.groups = {}
        self.groups["root"] = root_group.require_group(self.dataset_name)
        self.groups["df"] = self.groups["root"].require_group("df")
        self.groups["cleaner"] = self.groups["root"].require_group("cleaner")
        self.groups["analyzer"] = self.groups["root"].require_group("analyzer")

    def _create_paths(self):
        self.paths = {}
        self.paths['df'] = {}
        self.paths['df']['source'] = BaseZARR.get_abs_path(group=self.groups["df"])/"source_df.parquet"

    def _get_source_df(self):
        source_df_path = self.paths['df']['source']
        if source_df_path.exists():
            return pd.read_parquet(source_df_path)
        else:
            error_msg = f"Source dataframe does not exists at:\n{source_df_path}"
            raise FileNotFoundError()
            
    def _create_source_df(self, df: pd.DataFrame | None):
        source_df_path = self.paths['df']['source']

        if source_df_path.exists():

            if df is not None:

                existing_df = pd.read_parquet(source_df_path)

                try:
                    assert_frame_equal(left=existing_df, right=df)
                    logger.warning(
                        f"Provided data already exists. The provided DataFrame was NOT saved."
                    )
                except AssertionError as e:
                    logger.warning(
                        f"Provided data already exists. However, the given data is NOT identical to the saved data."
                    )
                    logger.debug(str(e))

            return

        if df is not None:
            df.to_parquet(path=source_df_path, index=False)
            self.groups["df"].attrs["source_df_exists"] = True
            logger.info(f"Successfully created and saved new data.")
            return

        error_msg = (
            f"Source data doesn't exist at {source_df_path} and provided data is None."
        )
        logger.error(msg=error_msg)
        raise ValueError(error_msg)