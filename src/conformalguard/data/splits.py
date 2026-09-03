"""Reproducible train/conformalization/test splitting utilities."""

from dataclasses import dataclass
from math import isclose
from typing import Any

from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DataSplit:
    """Container for train, conformalization, and test partitions."""

    X_train: Any
    X_conf: Any
    X_test: Any
    y_train: Any
    y_conf: Any
    y_test: Any


def stratified_train_conf_test_split(
    X: Any,
    y: Any,
    *,
    train_size: float = 0.60,
    conf_size: float = 0.20,
    test_size: float = 0.20,
    random_state: int = 42,
) -> DataSplit:
    """Split classification data into stratified train/conf/test partitions."""

    sizes = (train_size, conf_size, test_size)

    if any(size <= 0 or size >= 1 for size in sizes):
        raise ValueError("train_size, conf_size, and test_size must be between 0 and 1.")

    if not isclose(sum(sizes), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train_size + conf_size + test_size must equal 1.0.")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        train_size=train_size,
        test_size=conf_size + test_size,
        stratify=y,
        random_state=random_state,
    )

    conf_fraction = conf_size / (conf_size + test_size)

    X_conf, X_test, y_conf, y_test = train_test_split(
        X_temp,
        y_temp,
        train_size=conf_fraction,
        test_size=1.0 - conf_fraction,
        stratify=y_temp,
        random_state=random_state,
    )

    return DataSplit(
        X_train=X_train,
        X_conf=X_conf,
        X_test=X_test,
        y_train=y_train,
        y_conf=y_conf,
        y_test=y_test,
    )