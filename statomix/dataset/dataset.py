from statomix.dataset.base import BaseDataset
from statomix.pipelines.cleaner.cleaner import Cleaner
from statomix.pipelines.analyzer.analyzer import Analyzer

class Dataset(BaseDataset):
    def __init__(self, dataset_name, root_group, df = None):
        super().__init__(dataset_name=dataset_name, root_group=root_group, df=df)

        self.cleaner = Cleaner(df_path = self.paths['df']['source'], root_group=self.zarr_groups['cleaner'])
        self.analyzer = Analyzer(root_group =  self.zarr_groups['analyzer'])