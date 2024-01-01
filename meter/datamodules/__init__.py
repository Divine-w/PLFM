from .twitter_datamodule import TwitterDataModule
from .weibo_datamodule import WeiboDataModule
from .bien_datamodule import BienDataModule
from .mnre_datamodule import MNREDataModule

_datamodules = {
    "twitter": TwitterDataModule,
    "weibo": WeiboDataModule,
    "bien": BienDataModule,
    "mnre": MNREDataModule
}
