from pathlib import Path

from fileverse.logger import Logger
from fileverse.formats.yaml import BaseYAML
from fileverse.formats.zarr import BaseZARR

from statomix.pipelines.base import BasePipeline

from .group_analyzer import GroupAnalyzer

logger = Logger(name="Analyzer").get_logger()

class Analyzer(BasePipeline):
    def __init__(self, root_group):
        super().__init__(root_group=root_group)

    def _get_default_version_meta(self):
        return {}
        
    def _get_default_config_meta(self):
        return {"group_analyzer_exists":False}

    def _get_group_analyzer(self, version,  config_version):
        # version_group = self.get_version_group(version=version, create_new=False, version_name=None)
    
        # config_group = self.get_config_group(
        #     version=None,
        #     version_group=version_group,
        #     config_name=None,
        #     create_new=False
        # )
        
        # base_path = BaseZARR.get_abs_path(config_group)
        
        # group_analyzer_paths_path = base_path/"group_analyzer_path.yaml"

        group_bundle = self._get_group_bundle(version=version, config_version=config_version)
        group_analyzer_paths_path = group_bundle['config']['path']/"group_analyzer_path.yaml"
        
        if not group_analyzer_paths_path.exists():
            error_msg = f"Group analyzer paths does not exists t \n{group_analyzer_paths_path}\n."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        group_analyzer_paths = BaseYAML.load(path=group_analyzer_paths_path)
        for k, v in group_analyzer_paths.items():
            group_analyzer_paths[k] = Path(v)
    
        return GroupAnalyzer(paths=group_analyzer_paths)

    def create_summary_report(self, version, config_version):
        group_analyzer = self._get_group_analyzer(version=version, config_version=config_version)
    
        group_bundle = self._get_group_bundle(
            version=version, 
            config_version=config_version,
        )
        summary_report_path = group_bundle['config']['path']/"summary.xlsx"

        if summary_report_path.exists():
            logger.info(f"Summary report already exists.")
            return
            
        group_analyzer.create_summary_report(path=summary_report_path)
        
