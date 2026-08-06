import pandas as pd
#from pathlib import Path
from collections import defaultdict

from statomix.pipelines.base import BasePipeline

from fileverse.logger import Logger
from fileverse.formats.zarr import BaseZARR
from fileverse.formats.yaml import BaseYAML

from .col.col_semantic_rules import DataTypes
from .col.col_report import ColReport, ColEditSchema
from .cat_meta_report import CatMetaReport, CatMetaEditSchema
from .surv.surv_report import SurvMetaReport, SurvPairs, SurvCatMetaEditSchema

logger = Logger(name="Cleaner").get_logger()

class Cleaner(BasePipeline):
    def __init__(self, df_path, root_group):
        super().__init__(root_group=root_group)

        self.df_path = df_path

        self.col_report = ColReport()
        self.cat_meta_report = CatMetaReport()
        self.surv_meta_report = SurvMetaReport()

    def _get_default_version_meta(self):
        return {}
        
    def _get_default_config_meta(self):
        return {}

    def create_col_report(self, version=None, version_name=None, config_version=None, config_name=None, create_new=False):
    
        group_bundle = self._get_group_bundle(
            version=version, 
            version_name=version_name,
            version_create_new=create_new,
            
            config_version=config_version,
            config_name=config_name,
            config_version_create_new=create_new,
        )
        
        version_meta = group_bundle['version']['meta']
        
        base_path = group_bundle['version']['path']
        
        col_report_path = base_path / "col_report.xlsx"
        col_profiles_path = base_path / "col_profiles.parquet"
        if col_report_path.exists() and not create_new:
            version = version_meta['version']
            logger.info(
                f"Column report already exists for version {version}. Set create_new=True to create a new one."
            )
            return
        
        df = pd.read_parquet(self.df_path)
        
        self.col_report.create_col_profiles(
            df=df, path=col_profiles_path, replace=create_new
        )
        
        self.col_report.create_col_report(
            df=df,
            report_path=col_report_path,
            profiles_path=col_profiles_path,
            #password="statomix",
            #lock=True,
            replace=create_new,
        )
        
        version_meta["col_report_exists"] = True
        group_bundle['version']['group'].attrs['meta'] = version_meta

    def create_col_edit_schema(self, version = None):
    
        group_bundle = self._get_group_bundle(version=version, config_version=None)
        version_meta = group_bundle['version']['meta']
        base_path = group_bundle['version']['path']
        
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
            version = version_meta['version']
            logger.info(f"Column edit schema already exists for version:{version}.")
            return
        
        curated_col_report = pd.ExcelFile(col_report_curated_path)
        rename_mapping, col_edit_schema = self.col_report.get_col_edit_schema(
            curated_col_report=curated_col_report
        )
        
        BaseYAML.save(data=rename_mapping, path=rename_mapping_path)
        col_edit_schema.save(path=col_edit_schema_path)
        
        col_profiles = self.col_report.load_col_profiles(
            path=col_profiles_path
        )
        col_profiles_curated = self.col_report.get_curated_col_profiles(
            col_profiles=col_profiles, col_edit_schema=col_edit_schema
        )
        self.col_report.save_col_profiles(
            col_profiles=col_profiles_curated, path=col_profiles_curated_path
        )
        
        version_meta["col_edit_schema_exists"] = True
        group_bundle['version']['group'].attrs['meta'] = version_meta

    def create_cat_meta_report(self, version=None, config_version=None, config_name=None, create_new=False):
    
        group_bundle = self._get_group_bundle(
            version=version,
            version_name=None,
            version_create_new=False,
        
            config_name = config_name,
            config_version = config_version,
            config_version_create_new=create_new,
        )
        
        version_meta = group_bundle['version']['meta']
        config_meta = group_bundle['config']['meta']
        
        base_path =  group_bundle['config']['path']
        req_base_path = group_bundle['version']['path']
        
        col_profiles_curated_path = req_base_path / "col_profiles_curated.parquet"
        col_profiles_curated = self.col_report.load_col_profiles(
            path=col_profiles_curated_path
        )
        
        rename_mapping_path = req_base_path / "rename_mapping.yaml"
        rename_mapping = BaseYAML.load(path=rename_mapping_path)
        
        meta_report_path = base_path / "cat_meta_report.xlsx"
        
        if meta_report_path.exists():
            version = version_meta['version']
            config_version = config_meta['version']
            logger.info(f"Categorical metadata report already exists for version:{version} and config_version:{config_version}")
            return
        
        df = pd.read_parquet(self.df_path)
        
        self.cat_meta_report.create_meta_report(
            df=df,
            col_profiles=col_profiles_curated,
            rename_mapping=rename_mapping,
            report_path=meta_report_path,
        )
        
        config_meta["cat_meta_report_exists"] = True
        group_bundle['config']['group'].attrs['meta'] =  config_meta

    def create_cat_meta_edit_schema(self, version=None, config_version=None):
        group_bundle = self._get_group_bundle(version=version, config_version=config_version)
    
        version_meta = group_bundle['version']['meta']
        config_meta = group_bundle['config']['meta']
        
        base_path =  group_bundle['config']['path']
        
        meta_edit_schema_path = base_path/"cat_meta_edit_schema.parquet"
        
        if meta_edit_schema_path.exists():
            version = version_meta['version']
            config_version = config_meta['version']
            logger.info(f"Categorical metadata edit schema already exists for version: {version} and config_version:{config_version}")
            return
        
        curated_meta_report_path = base_path / "cat_meta_report_curated.xlsx"
        curated_meta_report = pd.ExcelFile(curated_meta_report_path)
        
        meta_edit_schema = self.cat_meta_report.get_meta_edit_schema(curated_meta_report)
        meta_edit_schema.save(path = meta_edit_schema_path)

    def create_surv_meta_report(self, version=None, config_version=None, config_name=None, create_new=False):
    
        group_bundle = self._get_group_bundle(
            version=version,
            version_name=None,
            version_create_new=False,
        
            config_name = config_name,
            config_version = config_version,
            config_version_create_new=create_new,
        )
        
        version_meta = group_bundle['version']['meta']
        config_meta = group_bundle['config']['meta']
        
        base_path =  group_bundle['config']['path']
        req_base_path = group_bundle['version']['path']
        
        surv_profiles_path = base_path/ "surv_profiles.parquet"
        meta_report_path = base_path / "surv_meta_report.xlsx"
        
        if surv_profiles_path.exists() and meta_report_path.exists():
            version = version_meta['version']
            config_version = config_meta['version']
            logger.info(f"Survival metadata report already exists for version: {version} and config_version:{config_version}")
            return
        
        col_profiles_path = req_base_path/"col_profiles_curated.parquet"
        col_profiles = self.col_report.load_col_profiles(path=col_profiles_path)
        
        datatype_map = defaultdict(list)
        for profile in col_profiles.values():
            datatype_map[profile.col_type].append(profile.col_name)
        col_names = datatype_map[DataTypes.SURVIVAL]
        
        self.surv_meta_report.create_surv_report(col_names=col_names, report_path=meta_report_path, profiles_path=surv_profiles_path)

    def create_surv_meta_edit_schema(self, version=None, config_version=None):
    
        group_bundle = self._get_group_bundle(version=version, config_version=config_version)
        
        version_meta = group_bundle['version']['meta']
        config_meta = group_bundle['config']['meta']
        
        base_path =  group_bundle['config']['path']
        
        surv_pairs_path = base_path/"surv_pairs.parquet"
        surv_profiles_path = base_path/ "surv_profiles.parquet"
        surv_profiles_curated_path = base_path/"surv_profiles_curated.parquet"
        
        meta_report_path = base_path / "surv_meta_report.xlsx"
        meta_edit_schema_path = base_path/"surv_meta_edit_schema.parquet"
        meta_report_curated_path = base_path / "surv_meta_report_curated.xlsx"
        
        if surv_pairs_path.exists() and surv_profiles_curated_path.exists() and meta_edit_schema_path.exists():
            version = version_meta['version']
            config_version = config_meta['version']
            logger.info(f"Surival meta data already exists for version:{version} and config_version:{config_version}")
            return
        
        surv_profiles = self.surv_meta_report.load_semantic_profiles(path=surv_profiles_path)
        curated_meta_report = pd.ExcelFile(meta_report_curated_path)
        
        meta_edit_schema = self.surv_meta_report.get_surv_edit_schema(curated_meta_report=curated_meta_report)
        meta_edit_schema.save(path=meta_edit_schema_path)
        
        surv_profiles_curated = self.surv_meta_report.get_curated_surv_profiles(meta_edit_schema=meta_edit_schema, surv_profiles=surv_profiles)
        self.surv_meta_report.save_semantic_profiles(semantic_profiles=surv_profiles_curated, path=surv_profiles_curated_path)
        
        surv_meta_df = curated_meta_report.parse(sheet_name="SurvMeta")
        surv_pairs = self.surv_meta_report.get_surv_pairs(surv_meta_df =surv_meta_df ,surv_profiles=surv_profiles_curated)
        surv_pairs.save(path=surv_pairs_path)

    def create_surv_cat_meta_report(self, version=None, config_version=None, config_name=None, create_new=False):
    
        group_bundle = self._get_group_bundle(
            version=version,
            version_name=None,
            version_create_new=False,
        
            config_name = config_name,
            config_version = config_version,
            config_version_create_new=create_new,
        )
        
        version_meta = group_bundle['version']['meta']
        config_meta = group_bundle['config']['meta']
        
        base_path =  group_bundle['config']['path']
        req_base_path = group_bundle['version']['path']
        
        profiles_path = base_path/"surv_profiles_curated.parquet"
        meta_report_path = base_path/"surv_cat_meta_report.xlsx"
        
        rename_mapping_path = req_base_path / "rename_mapping.yaml"
        #col_profiles_path = req_base_path/"col_profiles_curated.parquet"
        
        rename_mapping = BaseYAML.load(path=rename_mapping_path)
        
        if meta_report_path.exists():
            version = version_meta['version']
            config_version = config_meta['version']
            logger.info(f"Survival categorical metadata report already exists for version: {version} and config_version:{config_version}")
            return
        
        df = pd.read_parquet(self.df_path)
        self.surv_meta_report.create_cat_meta_report(df=df, rename_mapping=rename_mapping, profiles_path=profiles_path , report_path=meta_report_path)

    def create_surv_cat_meta_edit_schema(self, version=None, config_version=None):
        group_bundle = self._get_group_bundle(version=version, config_version=config_version)
    
        version_meta = group_bundle['version']['meta']
        config_meta = group_bundle['config']['meta']
        
        base_path =  group_bundle['config']['path']
        
        meta_report_curated_path = base_path/"surv_cat_meta_report_curated.xlsx"
        meta_edit_schema_path = base_path/"surv_cat_meta_edit_schema.parquet"
        if meta_edit_schema_path.exists():
            version = version_meta['version']
            config_version = config_meta['version']
            logger.info(f"Survival categorical metadata edit schema already exists for version: {version} and config_version:{config_version}")
            #return
            
        curated_meta_report = pd.ExcelFile(meta_report_curated_path)
        
        meta_edit_schema = self.surv_meta_report.get_surv_cat_meta_edit_schema(curated_meta_report=curated_meta_report)
        meta_edit_schema.save(path=meta_edit_schema_path)

    def create_curated_data(self, version=None, config_version=None):
    
        group_bundle = self._get_group_bundle(version=version, config_version=config_version)
        
        version_meta = group_bundle['version']['meta']
        config_meta = group_bundle['config']['meta']
        
        base_path =  group_bundle['config']['path']
        req_base_path = group_bundle['version']['path']
        
        config_group = group_bundle['config']['group']
        
        curated_data_group = config_group.require_group("curated_data")
        curated_base_path = BaseZARR.get_abs_path(curated_data_group)
        
        curated_df_path = curated_base_path / "df.parquet"
        curated_surv_pairs_path =  curated_base_path/"surv_pairs.parquet"
        curated_col_profiles_path =  curated_base_path/"col_profiles.parquet"
        
        if curated_df_path.exists() and curated_surv_pairs_path.exists() and curated_col_profiles_path.exists():
            version = version_meta['version']
            config_version = config_meta['version']
            logger.info(f"Curated data already exists for version:{version} and config_version:{config_version}")
            #return
        
        curated_data_meta = curated_data_group.attrs.get("meta", {})
        curated_data_meta["curated_data_exists"] =  False
        
        surv_pairs_path = base_path/"surv_pairs.parquet"
        rename_mapping_path = req_base_path / "rename_mapping.yaml"
        col_edit_schema_path = req_base_path / "col_edit_schema.parquet"
        cat_meta_edit_schema_path = base_path/"cat_meta_edit_schema.parquet"
        col_profiles_curated_path =  req_base_path/"col_profiles_curated.parquet"
        surv_cat_meta_edit_schema_path = base_path/"surv_cat_meta_edit_schema.parquet"
        
        rename_mapping = BaseYAML.load(path=rename_mapping_path)
        rename_mapping_swapped = {v: k for k, v in rename_mapping.items()}
        
        col_edit_schema = ColEditSchema.load(path=col_edit_schema_path)
        
        remove_cols = []
        for col_name, col_edit in col_edit_schema.edits.items():
            if col_edit.remove:
                remove_cols.append(col_name)
        
        cat_meta_edit_schema = CatMetaEditSchema.load(cat_meta_edit_schema_path)
        
        cat_meta_edits = cat_meta_edit_schema.cat_edits
        
        cat_rename_mapping = defaultdict(dict)
        for col_name, cat_edit in cat_meta_edits.items():
            for category, schema in cat_edit.items():
                if schema.category is not None and schema.rename_to is not None:
                    cat_rename_mapping[col_name][schema.category] = schema.rename_to
                elif schema.remove:
                    cat_rename_mapping[col_name][schema.category] = pd.NA
                else:
                    error_msg = f"Column {col_name} has neither a rename mapping nor needs to be removed, still present in schema."
                    raise ValueError(error_msg)
        
        surv_cat_meta_edit_schema = SurvCatMetaEditSchema.load(path=surv_cat_meta_edit_schema_path)
        
        surv_cat_meta_edits =  surv_cat_meta_edit_schema.cat_edits
        
        surv_cat_rename_mapping = defaultdict(dict)
        for col_name, surv_cat_edit in surv_cat_meta_edits.items():
            for category, schema in surv_cat_edit.items():
                if schema.category is not None and schema.rename_to is not None:
                    surv_cat_rename_mapping[col_name][schema.category] = schema.rename_to
                elif schema.remove:
                    surv_cat_rename_mapping[col_name][schema.category] = pd.NA
                else:
                    error_msg = f"Column {col_name} has neither a rename mapping nor needs to be removed, still present in schema."
                    raise ValueError(error_msg)
        
        
        surv_pairs = SurvPairs.load(path=surv_pairs_path)
        
        col_profiles_curated = self.col_report.load_col_profiles(path=col_profiles_curated_path)
        
        # Apply all the changes
        df = pd.read_parquet(path = self.df_path)
        
        df = df.drop(columns=remove_cols)
        df = df.rename(columns=rename_mapping_swapped)
        
        for col_name, cat_rename_map in cat_rename_mapping.items():
            df[col_name] = df[col_name].replace(cat_rename_map)
        
        for col_name, surv_cat_remame_map in surv_cat_rename_mapping.items():
            df[col_name] = df[col_name].replace(surv_cat_remame_map)
        
        df.to_parquet(path=curated_df_path)
        surv_pairs.save(path=curated_surv_pairs_path)
        self.col_report.save_col_profiles(col_profiles=col_profiles_curated, path=curated_col_profiles_path)
        
        curated_data_meta["curated_data_exists"] =  True
        curated_data_group.attrs["meta"] = curated_data_meta

    def get_curated_data_group(self,version = None, config_version = None):
    
        group_bundle = self._get_group_bundle(version=version, config_version=config_version)
        
        version_meta = group_bundle['version']['meta']
        config_meta = group_bundle['config']['meta']
        
        config_group = group_bundle['config']['group']
        
        curated_data_group = config_group.require_group("curated_data")
        curated_data_meta = curated_data_group.attrs.get("meta", {})
        
        if not curated_data_meta['curated_data_exists']:
            version = version_meta['version']
            config_version = config_meta['version']
            logger.info(f"Curated group does not exist for version:{version} and config_version:{config_version}")
            return
        else:
            return curated_data_group
    
