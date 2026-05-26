from pathlib import Path

from fileverse.logger import Logger
from fileverse.formats.zarr import BaseZARR

from statomix.dataset.base import BaseDataset

ROOT = Path.cwd() / "multiomix/statomix"
logger = Logger(name="Project").get_logger()

class Project:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self._zarr_storage = BaseZARR(path=ROOT / f"{self.project_name}")
        self.zarr_groups = {}
        self.zarr_groups["root"] = self._zarr_storage.root_group

        # if 'datasets' not in self.zarr_groups['root'].attrs:
        #     self.zarr_groups['root'].attrs['datasets'] = {}
        self.zarr_groups["datasets_root"] = self.zarr_groups["root"].require_group(
            "Datasets"
        )

        self._discover_datasets()

    def add_dataset(self, df, dataset_name):
        project_datasets_meta = self.zarr_groups["root"].attrs.get("datasets", {})

        if (
            dataset_name in project_datasets_meta
            and project_datasets_meta[dataset_name]["created_successfully"]
        ):
            logger.warning(
                f"Dataset '{dataset_name}' already exists in this project. Please choose a unique name or delete the existing dataset."
            )
            return

        project_datasets_meta[dataset_name] = {}
        project_datasets_meta[dataset_name]["created_successfully"] = False
        self.zarr_groups["root"].attrs["datasets"] = project_datasets_meta

        self.datasets[dataset_name] = BaseDataset(
            df=df, dataset_name=dataset_name, root_group=self.zarr_groups["datasets_root"]
        )

        project_datasets_meta[dataset_name]["created_successfully"] = True
        self.zarr_groups["root"].attrs["datasets"] = project_datasets_meta

        logger.info(f"Dataset '{dataset_name}' successfully initialized and registered.")

    def _discover_datasets(self):

        self.datasets = {}
        project_datasets_meta = self.zarr_groups["root"].attrs.get("datasets", {})

        if project_datasets_meta:
            for dataset_name, dataset_meta in project_datasets_meta.items():
                if dataset_meta["created_successfully"]:
                    self.datasets[dataset_name] = BaseDataset(
                        df=None,
                        dataset_name=dataset_name,
                        root_group=self.zarr_groups["datasets_root"],
                    )
                    logger.info(
                        msg=f"Discovered and loaded existing dataset: '{dataset_name}'"
                    )

    def create_col_report(self, dataset_name, report_type='default', create_new=False):
    
        dataset = self.datasets[dataset_name]
    
        zarr_group = dataset.zarr_groups['report']
        version = dataset.zarr_groups['report'].attrs['col_report_version']
        
        if not zarr_group.attrs['col_report_exists']:
            version = 1
        elif create_new:
            version += 1
        else:
            logger.info(msg=f"{Logger.Emojis.WARN} col_report version already exists. Set create_new=True to create a new version.")
            return
        
        report_path = BaseZARR.get_abs_path(zarr_group=zarr_group)/f"col_report_version{version}.xlsx"
        
        dataset.col_report.create_col_report_default(df=dataset.get_source_df(), report_path=report_path)
    
        zarr_group.attrs['col_report_exists'] = True
        zarr_group.attrs['col_report_version'] = version

    # def add_dataset(self, df, df_name):
    #     if df_name not in self.df_dict:
    #         self.df_dict[df_name] = DatasetNode(df=df, df_name=df_name, df_root_group=self._df_root_group)
    #     else:
    #         logger.warning(f"Dataset {df_name} already exists in the project.")