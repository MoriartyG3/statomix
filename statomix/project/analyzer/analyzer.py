from statomix.pipelines.base import BasePipeline
from statomix.project.analyzer.analysis_config import _create_analysis_config

from fileverse.logger import Logger

logger = Logger(name="project_analyzer").get_logger()


class Analyzer(BasePipeline):
    def __init__(self, root_group, dataset_name):
        super().__init__(root_group=root_group, dataset_name=dataset_name, pipeline_name="project_analyzer")

    def _get_default_version_meta(self):
        return {}

    def _get_default_config_meta(self):
        return {}

    def create_analysis_config(
        self, project, version, config_version, analysis_name=None, create_new=False
    ):

        group_bundle = self._get_group_bundle(
            version=version, config_version=config_version
        )

        config_meta = group_bundle["config"]["meta"]
        if "analysis_config" not in config_meta:
            config_meta["analysis_config"] = {}
            config_meta["analysis_config"]["latest_version"] = 1
            config_meta["analysis_config"]["version_history"] = [1]
            config_meta["analysis_config"]["name"] = analysis_name
            group_bundle["config"]["group"].attrs["meta"] = config_meta
        elif create_new:
            latest_version = config_meta["analysis_config"]["latest_version"]
            latest_version += 1
            config_meta["analysis_config"]["version_history"].append(latest_version)
            config_meta["analysis_config"]["latest_version"] = latest_version
            group_bundle["config"]["group"].attrs["meta"] = config_meta

        version = config_meta["analysis_config"]["latest_version"]
        analysis_config_path = (
            group_bundle["config"]["path"] / f"analysis_config_version{version}.xlsx"
        )

        if analysis_config_path.exists():
            logger.info(
                f"Analysis configuration version:{version} already exists, Set create_new=True to create a new one."
            )
            return

        _create_analysis_config(
            project=project,
            version=None,
            config_version=None,
            path=analysis_config_path,
        )
