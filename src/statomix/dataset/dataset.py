from statomix.dataset.base import BaseDataset
from statomix.pipelines.cleaner.cleaner import Cleaner
from statomix.pipelines.analyzer.analyzer import Analyzer

from fileverse.logger import Logger
from fileverse.formats.yaml import BaseYAML
from fileverse.formats.zarr import BaseZARR

logger = Logger(name="dataset").get_logger()

class Dataset(BaseDataset):
    def __init__(self, dataset_name, root_group, df = None):
        super().__init__(dataset_name=dataset_name, root_group=root_group, df=df)

        self.cleaner = Cleaner(df_path = self.paths['df']['source'], root_group=self.groups['cleaner'], dataset_name=dataset_name)
        self.analyzer = Analyzer(root_group =  self.groups['analyzer'], dataset_name=dataset_name)
    
    def configure_analyzer(self, version=None, config_version=None):
    
        # Resolve the requested cleaner version and configuration.
        cleaner_group_bundle = self.cleaner._get_group_bundle(
            version=version,
            config_version=config_version,
        )
    
        cleaner_version_meta = cleaner_group_bundle["version"]["meta"]
        cleaner_config_meta = cleaner_group_bundle["config"]["meta"]
    
        cleaner_version = cleaner_version_meta["version"]
        cleaner_config_version = cleaner_config_meta["version"]
    
        # Defensive checks that the requested identifiers were resolved correctly.
        if version is not None and cleaner_version != version:
            raise RuntimeError(
                f"Requested cleaner version {version}, "
                f"but version {cleaner_version} was resolved."
            )
    
        if (
            config_version is not None
            and cleaner_config_version != config_version
        ):
            raise RuntimeError(
                f"Requested cleaner config version {config_version}, "
                f"but config version {cleaner_config_version} was resolved."
            )
    
        # Confirm that all required curated outputs exist before creating
        # the corresponding analyzer hierarchy.
        curated_data_group = self.cleaner.get_curated_data_group(
            version=cleaner_version,
            config_version=cleaner_config_version,
        )
    
        if curated_data_group is None:
            raise FileNotFoundError(
                "Curated data is unavailable for "
                f"version:{cleaner_version} and "
                f"config_version:{cleaner_config_version}."
            )
    
        curated_data_base_path = BaseZARR.get_abs_path(
            curated_data_group
        )
    
        required_files = (
            "df",
            "surv_pairs",
            "col_profiles",
        )
    
        expected_analyzer_paths = {
            file_name: str(
                curated_data_base_path / f"{file_name}.parquet"
            )
            for file_name in required_files
        }
    
        # Create or retrieve the analyzer hierarchy using the exact
        # cleaner version and configuration identifiers.
        analyzer_group_bundle = (
            self.analyzer._require_exact_group_bundle(
                version=cleaner_version,
                config_version=cleaner_config_version,
                version_name=cleaner_version_meta.get("name"),
                config_name=cleaner_config_meta.get("name"),
            )
        )
    
        analyzer_config_group = analyzer_group_bundle["config"]["group"]
        analyzer_config_base_path = analyzer_group_bundle["config"]["path"]
        analyzer_config_meta = dict(
            analyzer_group_bundle["config"]["meta"]
        )
    
        group_analyzer_paths_path = (
            analyzer_config_base_path / "group_analyzer_path.yaml"
        )
    
        # Make the configuration operation idempotent. If the file already
        # exists, ensure that it contains the expected mapping.
        if group_analyzer_paths_path.exists():
            existing_analyzer_paths = BaseYAML.load(
                path=group_analyzer_paths_path
            )
    
            if existing_analyzer_paths != expected_analyzer_paths:
                raise RuntimeError(
                    "The existing analyzer path configuration does not "
                    "match the selected cleaner configuration.\n"
                    f"Expected: {expected_analyzer_paths}\n"
                    f"Existing: {existing_analyzer_paths}"
                )
    
            # Repair the metadata if the YAML exists but the flag is missing.
            if not analyzer_config_meta.get(
                "group_analyzer_exists", False
            ):
                analyzer_config_meta["group_analyzer_exists"] = True
                analyzer_config_group.attrs["meta"] = (
                    analyzer_config_meta
                )
    
            logger.info(
                "Analyzer data already exists for "
                f"version:{cleaner_version} and "
                f"config_version:{cleaner_config_version}"
            )
    
            return analyzer_group_bundle
    
        BaseYAML.save(
            data=expected_analyzer_paths,
            path=group_analyzer_paths_path,
        )
    
        analyzer_config_meta["group_analyzer_exists"] = True
        analyzer_config_meta["cleaner_source"] = {
            "version": cleaner_version,
            "config_version": cleaner_config_version,
        }
    
        analyzer_config_group.attrs["meta"] = analyzer_config_meta
    
        logger.info(
            "Analyzer configured for "
            f"version:{cleaner_version} and "
            f"config_version:{cleaner_config_version}"
        )
    
        #return analyzer_group_bundle

    # def configure_analyzer(self, version=None, config_version=None, create_new=False):

    #     #Cleaner
    #     cleaner_group_bundle = self.cleaner._get_group_bundle(version=version, config_version=config_version)
        
    #     cleaner_version_meta = cleaner_group_bundle['version']['meta']
    #     cleaner_config_meta = cleaner_group_bundle['config']['meta']
        
    #     if version is not None:
    #         assert cleaner_version_meta['version'] ==  version
    #     else:
    #         version = cleaner_version_meta['version']
        
    #     if config_version is not None:
    #         assert cleaner_config_meta['version'] ==  config_version
    #     else:
    #         config_version = cleaner_config_meta['version']
        
    #     curated_data_group = self.cleaner.get_curated_data_group(version=version, config_version=config_version)
        
    #     #Analyzer
    #     analyzer_group_bundle = self.analyzer._get_group_bundle(
    #         version=version, 
    #         version_create_new=create_new, 
    #         version_name=cleaner_group_bundle['version']['meta']['name'],
        
    #         config_version=config_version,
    #         config_name=cleaner_group_bundle['config']['meta']['name'],
    #         config_version_create_new=create_new
            
    #     )
        
    #     analyzer_config_base_path = analyzer_group_bundle['config']['path']
    #     analyzer_config_meta  = analyzer_group_bundle['config']['meta']
        
    #     group_analyzer_paths_path =  analyzer_config_base_path/"group_analyzer_path.yaml"
    #     if group_analyzer_paths_path.exists():
    #         cleaner_version = version#cleaner_version_meta["version"]
    #         cleaner_config_version =  config_version#cleaner_config_meta["version"]
    #         logger.info(f"Analyzer data already exists for version:{cleaner_version} and config_version:{cleaner_config_version}")
    #         return
        
    #     curated_data_group = self.cleaner.get_curated_data_group(version=version, config_version=config_version)
    #     curated_data_base_path = BaseZARR.get_abs_path(curated_data_group)
        
    #     files = ["df", "surv_pairs", "col_profiles"]
    #     group_analyzer_paths = {name: str(curated_data_base_path / f"{name}.parquet") for name in files}
        
    #     BaseYAML.save(data=group_analyzer_paths, path=group_analyzer_paths_path)
        
    #     analyzer_config_meta['group_analyzer_exists'] =  True
    #     analyzer_group_bundle['config']['group'].attrs['meta'] = analyzer_config_meta

    # def configure_analyzer(self, version = None, config_version = None, create_new=False):

    #     #Started with Cleaner to prep req data for Analyzer
    #     cleaner_version_group = self.cleaner.get_version_group(
    #         version=version, create_new=False, version_name=None
    #     )
        
    #     cleaner_config_group = self.cleaner.get_config_version_group(
    #         config_version=config_version,
    #         version_group=cleaner_version_group,
    #         config_name=None,
    #         create_new=False,
    #     )

    #     cleaner_version_meta = cleaner_version_group.attrs['meta']
    #     cleaner_config_meta = cleaner_version_group.attrs["meta"]
        
    #     if version is not None:
    #         assert cleaner_version_meta['version'] ==  version
        
    #     if config_version is not None:
    #         assert cleaner_config_meta['version'] ==  config_version
    
            

    #     #Analyzer Takes Overs
    #     analyzer_version_group = self.analyzer.get_version_group(
    #         version=version, create_new=create_new, version_name=cleaner_version_meta["name"]
    #     )
    #     analyzer_version_meta = analyzer_version_group.attrs["meta"]
    
    #     analyzer_config_group = self.analyzer.get_config_group(
    #         version=config_version, 
    #         version_group=analyzer_version_group, 
    #         config_name=cleaner_config_meta['name'],
    #         create_new=create_new
    #     )
    #     analyzer_config_base_path = BaseZARR.get_abs_path(analyzer_config_group)
    #     analyzer_config_meta  = analyzer_config_group.attrs["meta"]

    #     group_analyzer_paths_path =  analyzer_config_base_path/"group_analyzer_path.yaml"
    #     if group_analyzer_paths_path.exists():
    #         cleaner_version = cleaner_version_meta["version"]
    #         cleaner_config_version =  cleaner_config_meta["version"]
    #         logger.info(f"Analyzer data already exists for version:{cleaner_version} and config_version:{cleaner_config_version}")
    #         return
            
    #     curated_data_group = self.cleaner.get_curated_data_group(version=version, config_version=config_version)
    #     curated_data_base_path = BaseZARR.get_abs_path(curated_data_group)
        
    #     files = ["df", "surv_pairs", "col_profiles"]
    #     group_analyzer_paths = {name: str(curated_data_base_path / f"{name}.parquet") for name in files}
    
    #     BaseYAML.save(data=group_analyzer_paths, path=group_analyzer_paths_path)
    
    #     analyzer_config_meta['group_analyzer_exists'] =  True
    #     analyzer_config_group.attrs["meta"] = analyzer_config_meta
