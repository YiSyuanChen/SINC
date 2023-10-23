from ..datasets import SINCDataset
from .datamodule_base import BaseDataModule


class SINCDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return SINCDataset

    @property
    def dataset_cls_no_false(self):
        return SINCDataset

    @property
    def dataset_name(self):
        return "sinc"
