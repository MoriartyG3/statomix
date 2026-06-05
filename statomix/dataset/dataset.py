from statomix.reports.reports import Reports
from statomix.dataset.base import BaseDataset

class Dataset(BaseDataset):
    def __init__(self, dataset_name, root_group, df = None):
        super().__init__(dataset_name=dataset_name, root_group=root_group, df=df)

        self.reports = Reports(root_group=self.zarr_groups['reports'])