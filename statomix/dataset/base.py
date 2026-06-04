import pandas as pd
from pandas.testing import assert_frame_equal

from fileverse.logger import Logger
from fileverse.formats.zarr import BaseZARR
from fileverse.formats.yaml import BaseYAML

from statomix.reports.col_report import ColReport
from statomix.reports.meta_report import MetaReport

from statomix.reports.col_report import ColEditSchema


base_yaml = BaseYAML()
logger = Logger(name="BaseDataset").get_logger()


class BaseDataset:
    def __init__(self, dataset_name: str, root_group, df: pd.DataFrame = None):

        self.dataset_name = dataset_name

        self._create_groups(root_group=root_group)
        self._create_paths()

        self._create_source_df(df=df)
        self._col_report = ColReport()
        self._meta_report =  MetaReport()
        # self.create_col_report()

    def _create_groups(self, root_group):
        self.zarr_groups = {}
        self.zarr_groups["root"] = root_group.require_group(self.dataset_name)

        self.zarr_groups["df"] = self.zarr_groups["root"].require_group("df")
        if "source_df_exists" not in self.zarr_groups["df"].attrs:
            self.zarr_groups["df"].attrs["source_df_exists"] = False

        self.zarr_groups["col_report_default"] = self.zarr_groups["root"].require_group(
            "col_report_default"
        )
        self.zarr_groups["col_report_curated"] = self.zarr_groups["root"].require_group(
            "col_report_curated"
        )

        self.zarr_groups["meta_report_default"] = self.zarr_groups["root"].require_group(
            "meta_report_default"
        )

        self.zarr_groups["meta_report_curated"] = self.zarr_groups["root"].require_group(
            "meta_report_curated"
        )

    def _create_paths(self):
        self.paths = {}
        self.paths["source_df"] = (
            BaseZARR.get_abs_path(zarr_group=self.zarr_groups["df"])
            / "source_df.parquet"
        )

    def get_source_df(self):
        source_df_path = self.paths["source_df"]

        if source_df_path.exists():
            return pd.read_parquet(path=source_df_path)
        else:
            logger.error(f"source_df does not exist at {source_df_path}")
            return

    def _create_source_df(self, df: pd.DataFrame | None):
        source_df_path = self.paths["source_df"]

        if source_df_path.exists():

            if df is not None:

                existing_df = pd.read_parquet(source_df_path)

                try:
                    assert_frame_equal(left=existing_df, right=df)
                    logger.warning(
                        f"source_df already exists. The provided DataFrame was NOT saved to avoid overwriting."
                    )
                except AssertionError as e:
                    logger.warning(
                        f"source_df already exists. However, the provided DataFrame is NOT identical to the saved DataFrame."
                    )
                    logger.debug(str(e))

            return

        if df is not None:
            df.to_parquet(path=source_df_path, index=False)
            self.zarr_groups["df"].attrs["source_df_exists"] = True
            logger.info(f"Successfully created and saved new source_df.")
            return

        error_msg = (
            f"source_df doesn't exist at {source_df_path} and provided df is None."
        )
        logger.error(msg=error_msg)
        raise ValueError(error_msg)

    def create_col_report(
        self,
        report_type="default",
        create_new=False,
        password="statomix",
        lock=False,
        version = None
    ):
        # dataset = self.datasets[dataset_name]

        if report_type == "default":
            default_col_report_group = self.zarr_groups["col_report_default"]
            default_col_report_meta = default_col_report_group.attrs.get("col_report_default", {})

            if version is None:

                if "latest_version" not in default_col_report_meta:
                    version = 1
                    default_col_report_meta["latest_version"] = version
                    default_col_report_meta["version_history"] = []
                    version_meta = {} 
                    #default_col_report_meta[f"version{version}"] = {}
                    version_meta['col_report_exists'] = False
                else:
                    version = default_col_report_meta['latest_version']
                    version_meta = default_col_report_meta[f'version{version}']
            else:
                version_meta = default_col_report_meta[f'version{version}']


            if version_meta['col_report_exists'] and not create_new:
                logger.info(
                    msg=f"{Logger.Emojis.WARN} Default column report version {version} already exists. Set create_new=True to create a new version."
                )
                return
            elif create_new:
                version += 1
                version_meta = {}
                version_meta['col_report_exists'] = False


            report_path = (BaseZARR.get_abs_path(zarr_group=default_col_report_group)/ f"version{version}_col_report.xlsx")
            profiles_path = (BaseZARR.get_abs_path(zarr_group=default_col_report_group)/ f"version{version}_col_profile.parquet")

            col_profiles = self._col_report.create_col_profiles(df=self.get_source_df())
            self._col_report.save_col_profiles(profiles_path=profiles_path, col_profiles=col_profiles)

            self._col_report._create_col_report(
                df=self.get_source_df(),
                col_profiles=col_profiles,
                report_path=report_path,
                # profiles_path=profiles_path,
                password=password,
                lock=lock,
            )

            default_col_report_meta["version_history"].append(version)
            version_meta['col_report_exists'] = True
            
            default_col_report_meta["latest_version"] = version
            default_col_report_meta[f'version{version}'] = version_meta
            default_col_report_group.attrs['col_report_default'] = default_col_report_meta



    def _create_col_edit_schema(self, version=None):

        default_col_report_group = self.zarr_groups["col_report_default"]
        default_col_report_meta = default_col_report_group.attrs.get("col_report_default", {})

        curated_col_report_group = self.zarr_groups["col_report_curated"]
        version_meta = curated_col_report_group.attrs.get(f'version{version}', {})

        if version is None:
            version = default_col_report_meta["latest_version"]
            req_version_meta = default_col_report_meta[f'version{version}']

            if not req_version_meta['col_report_exists']:
                print(f"Default column report version {version} does not exist. Create one first.")
                return
        
        if not version_meta:
            version_meta['col_edit_schema_exists'] =  False

        col_edit_schema_path = (BaseZARR.get_abs_path(zarr_group=curated_col_report_group)/ f"version{version}_col_edit_schema.parquet")
        curated_report_path = (BaseZARR.get_abs_path(zarr_group=curated_col_report_group)/ f"version{version}_col_report.xlsx")
        rename_mapping_path = (BaseZARR.get_abs_path(zarr_group=curated_col_report_group)/ f"version{version}_rename_mapping.yaml")
        default_profiles_path = (BaseZARR.get_abs_path(zarr_group=default_col_report_group)/ f"version{version}_col_profile.parquet")

        if not curated_report_path.exists():
            error_msg = f"Curated column report version {version} does not exist at {curated_report_path}"
            raise FileNotFoundError(error_msg)

        curated_col_report = pd.ExcelFile(curated_report_path)

        if version_meta['col_edit_schema_exists']:
            logger.info(f"Column edit schema version {version} already exists.")
            return

        rename_mapping, col_edit_schema = self._col_report.get_col_edit_schema(curated_col_report)
        base_yaml.save(data=rename_mapping, path=rename_mapping_path)

        col_edit_schema.save(path=col_edit_schema_path)

        version_meta['col_edit_schema_exists'] = True
        curated_col_report_group.attrs[f"version{version}"] = version_meta

    def create_curated_data(self, version=None):

        self._create_col_edit_schema(version=version)

        default_col_report_group = self.zarr_groups["col_report_default"]
        default_col_report_meta = default_col_report_group.attrs.get("col_report_default", {})

        curated_col_report_group = self.zarr_groups["col_report_curated"]
        version_meta = curated_col_report_group.attrs.get(f'version{version}', {})

        if version is None:
            version = default_col_report_meta["latest_version"]
            
            #version_meta = curated_col_report_meta[f"version{version}"]
            curated_data_exists = version_meta.get('curated_data_exists', False)
            if curated_data_exists:
                print(f"Curated data for version {version} aleady exists.")
                return
            else:
                version_meta['curated_data_exists'] =  False
    
        col_edit_schema_path =  BaseZARR.get_abs_path(zarr_group=curated_col_report_group)/f"version{version}_col_edit_schema.parquet"
        curated_profiles_path = BaseZARR.get_abs_path(zarr_group=curated_col_report_group)/f"version{version}_col_profiles.parquet"
        curated_report_path = BaseZARR.get_abs_path(zarr_group=curated_col_report_group)/f"version{version}_col_report_curated.xlsx"
        rename_mapping_path = BaseZARR.get_abs_path(zarr_group=curated_col_report_group)/ f"version{version}_rename_mapping.yaml"
        default_profiles_path = BaseZARR.get_abs_path(zarr_group=default_col_report_group)/ f"version{version}_col_profile.parquet"

        # curated_meta = curated_zarr_group.attrs.get(f'version{version}', {})
        # curated_meta['curated_data_exists'] = False
        
        if not col_edit_schema_path.exists():
            error_msg = f"Column edit schema version {version} not found at: {col_edit_schema_path}. Run create_col_edit_schema(version={version}) first."
            raise FileNotFoundError(error_msg)
         
        if curated_profiles_path.exists() and curated_report_path.exists():
            logger.info(f'Curated data version{version} already exists.')
            return
        
        rename_mapping = base_yaml.load(rename_mapping_path)
        col_edit_schema = ColEditSchema.load(col_edit_schema_path)
        col_profiles = self._col_report.load_col_profiles(profiles_path=default_profiles_path)
    
        curated_col_profiles = self._col_report._get_curated_col_profiles(col_profiles=col_profiles, col_edit_schema=col_edit_schema)    
        self._col_report.save_col_profiles(col_profiles=curated_col_profiles, profiles_path=curated_profiles_path)
        
        self._col_report._create_col_report(
            df = self.get_source_df(),
            col_profiles=curated_col_profiles,
            report_path=curated_report_path,
            rename_mapping=rename_mapping,
            lock=True,
            password=None,
        )

        version_meta['curated_data_exists'] = True
        curated_col_report_group.attrs[f"version{version}"] = version_meta

    def create_meta_report(self, version=None, sub_version=None):
    
        default_col_report_group = self.zarr_groups["col_report_default"]
        default_col_report_meta = default_col_report_group.attrs.get("col_report_default", {})
        
        if version is None:
            version = default_col_report_meta["latest_version"]
        
        curated_col_report_group = self.zarr_groups["col_report_curated"]
        req_version_meta = curated_col_report_group.attrs.get(f'version{version}', {})
        
        if not req_version_meta.get("col_edit_schema_exists", False):
            print(f"Column edit schema for version {version} does not exist.")
            return
        
        rename_mapping_path = BaseZARR.get_abs_path(zarr_group=curated_col_report_group)/ f"version{version}_rename_mapping.yaml"
        curated_profiles_path = BaseZARR.get_abs_path(zarr_group=curated_col_report_group)/f"version{version}_col_profiles.parquet"
        
        rename_mapping = base_yaml.load(rename_mapping_path)
        curated_col_profiles = self._col_report.load_col_profiles(profiles_path=curated_profiles_path)
        
        default_meta_report_group = self.zarr_groups['meta_report_default']
        version_meta = default_meta_report_group.attrs.get(f'version{version}', {})
        
        if sub_version is None:
            if 'latest_sub_version' not in version_meta:
                sub_version = 1
                version_meta["latest_sub_version"] =  sub_version
                version_meta["sub_version_history"] = []
            else:
                sub_version = version_meta["latest_sub_version"]
                
        
        sub_version_meta = version_meta.get(f"sub_version{sub_version}", {})
        
        if sub_version_meta.get("meta_report_exists", False):
            sub_version_meta['meta_report_exists'] =  False

        if sub_version_meta['meta_report_exists']:
            print(f"A sub version {sub_version} reporta already exists.")
            return
        
        default_report_path = (
            BaseZARR.get_abs_path(zarr_group=default_meta_report_group)
            / f"version{version}_subversion{sub_version}_meta_report.xlsx"
        )
        
        self._meta_report._create_meta_report(
            df=self.get_source_df(), 
            col_profiles=curated_col_profiles,
            rename_mapping=rename_mapping,
            report_path=default_report_path,
        )
    
        sub_version_meta['meta_report_exists'] = True
        version_meta["latest_sub_version"] = sub_version
        version_meta[f"sub_version{sub_version}"] = sub_version_meta
        version_meta["sub_version_history"].append(sub_version)
        
        default_meta_report_group.attrs[f'version{version}'] = version_meta