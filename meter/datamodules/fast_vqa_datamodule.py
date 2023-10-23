from ..datasets import FASTVQADataset
from .datamodule_base import BaseDataModule


class FASTVQADataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return FASTVQADataset

    @property
    def dataset_cls_no_false(self):
        return FASTVQADataset

    @property
    def dataset_name(self):
        return "fast_vqa"
