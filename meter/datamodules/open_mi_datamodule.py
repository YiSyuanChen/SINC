from ..datasets import OPENMIDataset
from .datamodule_base import BaseDataModule


class OPENMIDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return OPENMIDataset

    @property
    def dataset_cls_no_false(self):
        return OPENMIDataset

    @property
    def dataset_name(self):
        return "open_mi"
