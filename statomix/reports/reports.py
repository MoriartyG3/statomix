import pandas as pd

from statomix.reports.col_report import ColReport
from statomix.reports.meta_report import MetaReport

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
        self.meta_report =  MetaReport()

        
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
        if 'version' not in version_meta:
            version_meta['version'] =  version
            version_meta['name'] = version_name
            version_meta['col_report_exists'] = False
            version_meta['col_edit_schema_exists'] = False

            version_meta['config'] = {}
            version_meta['config']['latest_version'] = 1
            version_meta['config']['version_history'] =  [1]
            
            version_group.attrs['meta'] = version_meta
        
        return version_group

    def create_col_report(self, df, version=None, create_new=False, version_name=None):
        version_group = self.get_version_group(version=version, version_name=version_name, create_new=create_new)
        version_meta = version_group.attrs.get("meta", {})
        
        base_path = BaseZARR.get_abs_path(zarr_group=version_group)
        col_report_path = base_path/"col_report.xlsx"
        col_profiles_path =  base_path/"col_profiles.parquet"
        if col_report_path.exists() and not create_new:
            logger.info(f"Column report version {version_meta['version']} already exists. Set create_new=True to create a new one.")
            return
        
        self.col_report.create_col_profiles(df=df, profiles_path=col_profiles_path, replace=create_new)
        
        self.col_report.create_col_report(
            df=df,  
            report_path=col_report_path,
            profiles_path=col_profiles_path,
            password="statomix",
            lock=True,
            replace=create_new
    
        )

        version_meta['col_report_exists'] = True
        version_group.attrs['meta'] = version_meta


    def create_col_edit_schema(self, version=None):
    
        version_group = self.get_version_group(version=version, create_new=False, version_name=None)
        version_meta = version_group.attrs["meta"]
        
        base_path = BaseZARR.get_abs_path(zarr_group=version_group)
        
        col_report_curated_path = base_path/"col_report_curated.xlsx"
        if not col_report_curated_path.exists():
            error_msg = f"Curated column report does not exist at \n{col_report_curated_path}."
            raise FileNotFoundError(error_msg)
        
        rename_mapping_path =  base_path/"rename_mapping.yaml"
        col_edit_schema_path = base_path/"col_edit_schema.parquet"
        
        if rename_mapping_path.exists() and col_edit_schema_path.exists():
            logger.info(f"Column edit schema already exists.")
            return
        
        curated_col_report = pd.ExcelFile(col_report_curated_path)
        rename_mapping, col_edit_schema = self.col_report.get_col_edit_schema(curated_col_report=curated_col_report)
        
        
        base_yaml.save(data=rename_mapping, path=rename_mapping_path)
        col_edit_schema.save(path=col_edit_schema_path)
        
        version_meta['col_edit_schema_exists'] = True
        version_group.attrs['meta'] = version_meta