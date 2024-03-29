from ..datasets import MNREAuxiliaryDataset
from .datamodule_base import BaseDataModule


class MNREDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MNREAuxiliaryDataset

    @property
    def dataset_name(self):
        return "mnre"
