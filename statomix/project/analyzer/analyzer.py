from statomix.pipelines.base import BasePipeline

class Analyzer(BasePipeline):
    def __init__(self, root_group):
        super().__init__(root_group=root_group)

    def _get_default_version_meta(self):
        return {}
        
    def _get_default_config_meta(self):
        return {}