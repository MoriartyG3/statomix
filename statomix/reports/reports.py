import pandas as pd

from statomix.reports.col_report import ColReport, ColEditSchema
from statomix.reports.meta_report import MetaReport, MetaEditSchema

from fileverse.logger import Logger
from fileverse.formats.zarr import BaseZARR
from fileverse.formats.yaml import BaseYAML

base_yaml = BaseYAML()
logger = Logger(name="BaseDataset").get_logger()


class Reports:
    def __init__(self, root_group):
        self.root_group = root_group
        self.meta = self.root_group.attrs.get("meta", {})

        if "latest_version" not in self.meta:
            self.meta["latest_version"] = 1
            self.meta["version_history"] = [1]
            self._save_meta()

        self.col_report = ColReport()
        self.meta_report = MetaReport()

    def _save_meta(self):
        self.root_group.attrs["meta"] = self.meta

    def get_version_group(self, version, create_new, version_name):
        if version is None:
            version = self.meta["latest_version"]

        if create_new:
            version += 1
            self.meta["latest_version"] = version
            self.meta["version_history"].append(version)
            self._save_meta()
            version_group = self.root_group.require_group(f"version{version}")
        else:
            if version == 1:
                version_group = self.root_group.require_group(f"version{version}")
            elif f"version{version}" in self.root_group:
                version_group = self.root_group.require_group(f"version{version}")
            else:
                error_msg = f"\nReport version {version} not found. Set create_new=True to create a new report.\nLatest version is {self.meta["latest_version"]}"
                raise FileNotFoundError(error_msg)

        version_meta = version_group.attrs.get("meta", {})
        if "version" not in version_meta:
            version_meta["version"] = version
            version_meta["name"] = version_name
            version_meta["col_report_exists"] = False
            version_meta["col_edit_schema_exists"] = False

            version_meta["config"] = {}
            version_meta["config"]["latest_version"] = 1
            version_meta["config"]["version_history"] = [1]

            version_group.attrs["meta"] = version_meta

        return version_group

    def get_config_version_group(
        self, config_version, version_group, config_name, create_new
    ):
        version_meta = version_group.attrs["meta"]

        if config_version is None:
            config_version = version_meta["config"]["latest_version"]

        if create_new:
            config_version += 1

            version_meta["config"]["latest_version"] = config_version
            version_meta["config"]["version_history"].append(config_version)
            version_group.attrs["meta"] = version_meta
            config_version_group = version_group.require_group(
                f"config_version{config_version}"
            )
        else:
            if config_version == 1:
                config_version_group = version_group.require_group(
                    f"config_version{config_version}"
                )
            elif f"config_version{config_version}" in version_group:
                config_version_group = version_group.require_group(
                    f"config_version{config_version}"
                )
            else:
                error_msg = f"\nReport version {version} not found. Set create_new=True to create a new report.\nLatest version is {self.meta["latest_version"]}"
                raise FileNotFoundError(error_msg)

        config_version_meta = config_version_group.attrs.get("meta", {})
        if "version" not in config_version_meta:
            config_version_meta["config_version"] = config_version
            config_version_meta["config_name"] = config_name
            config_version_meta["meta_report_exists"] = False

            config_version_group.attrs["meta"] = config_version_meta

        return config_version_group

    def create_col_report(self, df, version=None, create_new=False, version_name=None):
        version_group = self.get_version_group(
            version=version, version_name=version_name, create_new=create_new
        )
        version_meta = version_group.attrs["meta"]

        base_path = BaseZARR.get_abs_path(zarr_group=version_group)
        col_report_path = base_path / "col_report.xlsx"
        col_profiles_path = base_path / "col_profiles.parquet"
        if col_report_path.exists() and not create_new:
            logger.info(
                f"Column report version {version_meta['version']} already exists. Set create_new=True to create a new one."
            )
            return

        self.col_report.create_col_profiles(
            df=df, profiles_path=col_profiles_path, replace=create_new
        )

        self.col_report.create_col_report(
            df=df,
            report_path=col_report_path,
            profiles_path=col_profiles_path,
            password="statomix",
            lock=True,
            replace=create_new,
        )

        version_meta["col_report_exists"] = True
        version_group.attrs["meta"] = version_meta

    def create_col_edit_schema(self, version=None):

        version_group = self.get_version_group(
            version=version, create_new=False, version_name=None
        )
        version_meta = version_group.attrs["meta"]

        base_path = BaseZARR.get_abs_path(zarr_group=version_group)

        col_report_curated_path = base_path / "col_report_curated.xlsx"
        if not col_report_curated_path.exists():
            error_msg = (
                f"Curated column report does not exist at \n{col_report_curated_path}."
            )
            raise FileNotFoundError(error_msg)

        col_profiles_path = base_path / "col_profiles.parquet"
        rename_mapping_path = base_path / "rename_mapping.yaml"
        col_edit_schema_path = base_path / "col_edit_schema.parquet"
        col_profiles_curated_path = base_path / "col_profiles_curated.parquet"

        if (
            rename_mapping_path.exists()
            and col_edit_schema_path.exists()
            and col_profiles_curated_path.exists()
        ):
            logger.info(f"Column edit schema already exists.")
            return

        curated_col_report = pd.ExcelFile(col_report_curated_path)
        rename_mapping, col_edit_schema = self.col_report.get_col_edit_schema(
            curated_col_report=curated_col_report
        )

        base_yaml.save(data=rename_mapping, path=rename_mapping_path)
        col_edit_schema.save(path=col_edit_schema_path)

        col_profiles = self.col_report.load_col_profiles(
            profiles_path=col_profiles_path
        )
        col_profiles_curated = self.col_report.get_curated_col_profiles(
            col_profiles=col_profiles, col_edit_schema=col_edit_schema
        )
        self.col_report.save_col_profiles(
            col_profiles=col_profiles_curated, profiles_path=col_profiles_curated_path
        )

        version_meta["col_edit_schema_exists"] = True
        version_group.attrs["meta"] = version_meta

    def create_meta_report(
        self, df, version=None, config_version=None, config_name=None, create_new=False
    ):

        version_group = self.get_version_group(
            version=version, create_new=False, version_name=None
        )
        req_base_path = BaseZARR.get_abs_path(zarr_group=version_group)

        config_version_group = self.get_config_version_group(
            config_version=config_version,
            version_group=version_group,
            config_name=config_name,
            create_new=create_new,
        )
        config_version_meta = config_version_group.attrs["meta"]

        base_path = BaseZARR.get_abs_path(zarr_group=config_version_group)

        col_profiles_curated_path = req_base_path / "col_profiles_curated.parquet"
        col_profiles_curated = self.col_report.load_col_profiles(
            profiles_path=col_profiles_curated_path
        )

        rename_mapping_path = req_base_path / "rename_mapping.yaml"
        rename_mapping = base_yaml.load(path=rename_mapping_path)

        meta_report_path = base_path / "meta_report.xlsx"

        if meta_report_path.exists():
            logger.info(f"Metadata report already exists at \n{meta_report_path}")
            return

        self.meta_report.create_meta_report(
            df=df,
            col_profiles=col_profiles_curated,
            rename_mapping=rename_mapping,
            report_path=meta_report_path,
        )

        config_version_meta["meta_report_exists"] = True
        config_version_group.attrs["meta"] = config_version_meta

    def create_meta_edit_schema(self, version=None, config_version=None):
        version_group = self.get_version_group(
                version=version, create_new=False, version_name=None
            )
        
        config_version_group = self.get_config_version_group(
            config_version=config_version,
            version_group=version_group,
            config_name=None,
            create_new=False
        )
        
        config_version_meta = config_version_group.attrs["meta"]
        
        base_path = BaseZARR.get_abs_path(zarr_group=config_version_group)
        meta_edit_schema_path = base_path/"meta_schema.xlsx"
    
        if meta_edit_schema_path.exists():
            logger.info(f"Metadata edit schema already exists at \n{meta_edit_schema_path}.")
            return
        
        curated_meta_report_path = base_path / "meta_report_curated.xlsx"
        curated_meta_report = pd.ExcelFile(curated_meta_report_path)
        
        meta_edit_schema = self.meta_report.get_meta_edit_schema(curated_meta_report)
        meta_edit_schema.save(path = meta_edit_schema_path)


    def create_schema_df(self, df, version=None, config_version=None):
    
        version_group = self.get_version_group(
            version=version, create_new=False, version_name=None
        )
        req_base_path = BaseZARR.get_abs_path(version_group)
        
        config_version_group = self.get_config_version_group(
            config_version=config_version,
            version_group=version_group,
            config_name=None,
            create_new=False,
        )
        base_path = BaseZARR.get_abs_path(config_version_group)
        
        schema_df_path = base_path / "schema_df.parquet"
        meta_edit_schema_path = base_path / "meta_schema.xlsx"
        rename_mapping_path = req_base_path / "rename_mapping.yaml"
        col_edit_schema_path = req_base_path / "col_edit_schema.parquet"
        
        if schema_df_path.exists():
            logger.info(
                f"Schema df already exists for, \nversion: {version_group.attrs['meta']['version']}\nconfig_version:{config_version_group.attrs['meta']['config_version']}\n"
            )
            return
        
        rename_mapping = base_yaml.load(path=rename_mapping_path)
        col_edit_schema = ColEditSchema.load(path=col_edit_schema_path)
        meta_edit_schema = MetaEditSchema.load(path=meta_edit_schema_path)
        
        
        rename_mapping_swapped = {v: k for k, v in rename_mapping.items()}
        
        remove_cols = []
        for col_name, col_edit in col_edit_schema.edits.items():
            if col_edit.remove:
                remove_cols.append(col_name)
        
        df = df.drop(columns=remove_cols)
        df = df.rename(columns=rename_mapping_swapped)
        
        df.to_parquet(path=schema_df_path)