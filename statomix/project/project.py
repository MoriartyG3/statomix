import pandas as pd
from pathlib import Path
from pandas.testing import assert_frame_equal

from fileverse.logger import Logger
from fileverse.formats.zarr import BaseZARR

from statomix.dataset.dataset import Dataset

from .analyzer.analyzer import Analyzer

ROOT = Path.cwd() / "multiomix/statomix"
logger = Logger(name="Project").get_logger()


class Project:
    def __init__(self, project_name: str):#, project_dir = None):
        self.project_name = project_name
        
        self._create_groups()
        
        self._discover_datasets()
        #self._init_meta()

        self.analyzer = Analyzer(root_group = self.groups["analyzer_root"])

    def _create_groups(self):
        
        path = ROOT / f"{self.project_name}"
        self._zarr_storage = BaseZARR(path=path)
        
        self.groups = {}
        self.groups["root"] = self._zarr_storage.root_group
        self.groups["datasets_root"] = self.groups["root"].require_group(
            "Datasets"
        )
        self.groups["analyzer_root"] = self.groups["root"].require_group(
            "Analyzer"
        ) 

    # def _init_meta(self):
    #     root_group = self.groups['root']
        
    #     if 'meta' in root_group.attrs:
    #         return
            
    #     root_group.attrs['meta'] = {}
    #     root_meta = root_group.attrs['meta']
        
    #     if 'project_level_analysis' not in root_meta:
    #         root_meta['project_level_analysis'] = {}
        
    #     if 'config' not in root_meta[ 'project_level_analysis']:
    #         root_meta['project_level_analysis']['config'] = {}
    #         root_meta['project_level_analysis']['config']["latest_version"] = 1
    #         root_meta['project_level_analysis']['config']["version_history"] = [1]

    #     root_group.attrs['meta'] = root_meta

    def add_dataset(self, df, dataset_name):
        project_datasets_meta = self.groups["root"].attrs.get("datasets", {})

        if (
            dataset_name in project_datasets_meta
            and project_datasets_meta[dataset_name]["created_successfully"]
        ):

            if df is not None:
                existing_df = pd.read_parquet(
                    self.datasets[dataset_name].paths["df"]["source"]
                )
                # Note: This check also exists in the BaseDataset._create_source_df() Method
                existing_df = existing_df.fillna(value=pd.NA)
                df = df.fillna(value=pd.NA)

                try:
                    assert_frame_equal(left=existing_df, right=df)
                    logger.warning(
                        f"Dataset '{dataset_name}' already exists in this project. Please choose a unique name or delete the existing dataset."
                    )
                except AssertionError as e:
                    logger.warning(
                        f"Dataset '{dataset_name}' already exists in this project. Please choose a unique name or delete the existing dataset.\nNote: the provided DataFrame is NOT identical to the saved DataFrame."
                    )
                    logger.debug(str(e))
            else:
                logger.warning(
                    f"Dataset '{dataset_name}' already exists in this project. Please choose a unique name or delete the existing dataset."
                )
            return

        project_datasets_meta[dataset_name] = {}
        project_datasets_meta[dataset_name]["created_successfully"] = False
        self.groups["root"].attrs["datasets"] = project_datasets_meta

        self.datasets[dataset_name] = Dataset(
            df=df,
            dataset_name=dataset_name,
            root_group=self.groups["datasets_root"],
        )

        project_datasets_meta[dataset_name]["created_successfully"] = True
        self.groups["root"].attrs["datasets"] = project_datasets_meta

        logger.info(
            f"Dataset '{dataset_name}' successfully initialized and registered."
        )

    def _discover_datasets(self):

        self.datasets = {}
        project_datasets_meta = self.groups["root"].attrs.get("datasets", {})

        if project_datasets_meta:
            for dataset_name, dataset_meta in project_datasets_meta.items():
                if dataset_meta["created_successfully"]:
                    self.datasets[dataset_name] = Dataset(
                        df=None,
                        dataset_name=dataset_name,
                        root_group=self.groups["datasets_root"],
                    )
                    logger.info(
                        msg=f"Discovered and loaded existing dataset: '{dataset_name}'"
                    )
    # def create_analysis_config(self):
        
    # def create_col_report(
    #     self,
    #     dataset_name,
    #     report_type="default",
    #     create_new=False,
    #     password="statomix",
    #     lock=False,
    # ):
    #     dataset = self.datasets[dataset_name]
        
    #     if report_type == "default":
    #         zarr_group = dataset.zarr_groups["col_report_default"]
    #         col_report_default_meta  = zarr_group.attrs.get("col_report_default", {})
            
    #         if 'default' not in col_report_default_meta:
    #             col_report_default_meta['default'] =  {}
    #             col_report_default_meta['default']['exists'] = False
    #             col_report_default_meta['default']['version'] = 0
                
    #         version = col_report_default_meta['default']['version']

    #         if not col_report_default_meta['default']['exists']:
    #             version = 1
    #         elif create_new:
    #             version += 1
    #         else:
    #             logger.info(
    #                 msg=f"{Logger.Emojis.WARN} Default column report version {version} already exists. Set create_new=True to create a new version."
    #             )
    #             return

    #         report_path = BaseZARR.get_abs_path(zarr_group=zarr_group)/ f"col_report_version{version}.xlsx"
    #         profiles_path = BaseZARR.get_abs_path(zarr_group=zarr_group)/ f"col_profile_version{version}.parquet"

    #         dataset.col_report.create_col_report_default(
    #             df=dataset.get_source_df(),
    #             report_path=report_path,
    #             profiles_path=profiles_path,
    #             password=password,
    #             lock=lock,
    #         )

    #         col_report_default_meta['default']['version'] = version
    #         col_report_default_meta['default']['exists'] = True

    #         dataset.zarr_groups["col_report_default"].attrs["col_report_default"] = col_report_default_meta 
            