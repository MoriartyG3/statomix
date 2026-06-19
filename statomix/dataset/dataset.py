from statomix.dataset.base import BaseDataset
from statomix.pipelines.cleaner.cleaner import Cleaner
from statomix.pipelines.analyzer.analyzer import Analyzer

from fileverse.logger import Logger
from fileverse.formats.yaml import BaseYAML
from fileverse.formats.zarr import BaseZARR

logger = Logger(name="Dataset").get_logger()

class Dataset(BaseDataset):
    def __init__(self, dataset_name, root_group, df = None):
        super().__init__(dataset_name=dataset_name, root_group=root_group, df=df)

        self.cleaner = Cleaner(df_path = self.paths['df']['source'], root_group=self.zarr_groups['cleaner'])
        self.analyzer = Analyzer(root_group =  self.zarr_groups['analyzer'])

    def configure_analyzer(self, version = None, config_version = None, create_new=False):

        #Started with Cleaner to prep req data for Analyzer
        cleaner_version_group = self.cleaner.get_version_group(
            version=version, create_new=False, version_name=None
        )
        
        cleaner_config_group = self.cleaner.get_config_version_group(
            config_version=config_version,
            version_group=cleaner_version_group,
            config_name=None,
            create_new=False,
        )

        cleaner_version_meta = cleaner_version_group.attrs['meta']
        cleaner_config_meta = cleaner_version_group.attrs["meta"]
        
        if version is not None:
            assert cleaner_version_meta['version'] ==  version
        
        if config_version is not None:
            assert cleaner_config_meta['version'] ==  config_version
    
        curated_data_group = self.cleaner.get_curated_data_group(version=version, config_version=config_version)

        #Analyzer Takes Overs
        analyzer_version_group = self.analyzer.get_version_group(
            version=version, create_new=create_new, version_name=cleaner_version_meta["name"]
        )
        analyzer_version_meta = analyzer_version_group.attrs["meta"]
    
        analyzer_config_group = self.analyzer.get_config_group(
            version=config_version, 
            version_group=analyzer_version_group, 
            config_name=cleaner_config_meta['name'],
            create_new=create_new
        )
        analyzer_config_base_path = BaseZARR.get_abs_path(analyzer_config_group)
        analyzer_config_meta  = analyzer_config_group.attrs["meta"]

        group_analyzer_paths_path =  analyzer_config_base_path/"group_analyzer_path.yaml"
        if group_analyzer_paths_path.exists():
            cleaner_version = cleaner_version_meta["version"]
            cleaner_config_version =  cleaner_config_meta["version"]
            logger.info(f"Analyzer data already exists for version:{cleaner_version} and config_version:{cleaner_config_version}")
            return
            
        curated_data_group = self.cleaner.get_curated_data_group(version=version, config_version=config_version)
        curated_data_base_path = BaseZARR.get_abs_path(curated_data_group)
        
        files = ["df", "surv_pairs", "col_profiles"]
        group_analyzer_paths = {name: str(curated_data_base_path / f"{name}.parquet") for name in files}
    
        BaseYAML.save(data=group_analyzer_paths, path=group_analyzer_paths_path)
    
        analyzer_config_meta['group_analyzer_exists'] =  True
        analyzer_config_group.attrs["meta"] = analyzer_config_meta