"""MAGIC Gamma Telescope dataset loader."""

import pandas as pd
from sklearn.datasets import fetch_openml

from conformalguard.data.datasets import DatasetBundle


MAGIC_OPENML_DATA_ID = 1120
MAGIC_EXPECTED_SAMPLES = 19020
MAGIC_EXPECTED_FEATURES = 10
MAGIC_EXPECTED_CLASSES = {"g", "h"}


def load_magic_telescope() -> DatasetBundle:
    """Load the pinned MAGIC Gamma Telescope dataset from OpenML."""

    X, y = fetch_openml(
        data_id=MAGIC_OPENML_DATA_ID,
        as_frame=True,
        return_X_y=True,
    )

    if not isinstance(X, pd.DataFrame):
        raise TypeError("MAGIC features must be returned as a pandas DataFrame.")

    if not isinstance(y, pd.Series):
        raise TypeError("MAGIC target must be returned as a pandas Series.")

    if X.shape != (MAGIC_EXPECTED_SAMPLES, MAGIC_EXPECTED_FEATURES):
        raise ValueError(
            "Unexpected MAGIC dataset shape: "
            f"{X.shape}; expected "
            f"({MAGIC_EXPECTED_SAMPLES}, {MAGIC_EXPECTED_FEATURES})."
        )

    if X.isna().any().any():
        raise ValueError("MAGIC dataset contains missing feature values.")

    observed_classes = set(y.astype(str).unique())

    if observed_classes != MAGIC_EXPECTED_CLASSES:
        raise ValueError(
            "Unexpected MAGIC target classes: "
            f"{sorted(observed_classes)}."
        )

    return DatasetBundle(
        name="MAGIC Gamma Telescope",
        source="OpenML",
        source_id=MAGIC_OPENML_DATA_ID,
        X=X,
        y=y,
    )