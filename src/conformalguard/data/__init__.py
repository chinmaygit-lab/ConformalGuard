"""Dataset loading and splitting utilities."""

from conformalguard.data.datasets import DatasetBundle
from conformalguard.data.magic import (
    MAGIC_OPENML_DATA_ID,
    load_magic_telescope,
)
from conformalguard.data.splits import (
    DataSplit,
    stratified_train_conf_test_split,
)

__all__ = [
    "DatasetBundle",
    "DataSplit",
    "MAGIC_OPENML_DATA_ID",
    "load_magic_telescope",
    "stratified_train_conf_test_split",
]