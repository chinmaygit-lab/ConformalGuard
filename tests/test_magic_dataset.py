from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from conformalguard.data import MAGIC_OPENML_DATA_ID, load_magic_telescope


def make_fake_magic():
    X = pd.DataFrame(
        np.zeros((19020, 10)),
        columns=[f"feature_{i}" for i in range(10)],
    )

    y = pd.Series(
        ["g"] * 12332 + ["h"] * 6688,
        name="class",
    )

    return X, y


@patch("conformalguard.data.magic.fetch_openml")
def test_magic_loader_uses_pinned_openml_dataset(mock_fetch):
    mock_fetch.return_value = make_fake_magic()

    dataset = load_magic_telescope()

    mock_fetch.assert_called_once_with(
        data_id=MAGIC_OPENML_DATA_ID,
        as_frame=True,
        return_X_y=True,
    )

    assert dataset.name == "MAGIC Gamma Telescope"
    assert dataset.source == "OpenML"
    assert dataset.source_id == 1120
    assert dataset.X.shape == (19020, 10)
    assert set(dataset.y.unique()) == {"g", "h"}


@patch("conformalguard.data.magic.fetch_openml")
def test_magic_loader_rejects_missing_features(mock_fetch):
    X, y = make_fake_magic()
    X.iloc[0, 0] = np.nan
    mock_fetch.return_value = X, y

    with pytest.raises(ValueError, match="missing"):
        load_magic_telescope()


@patch("conformalguard.data.magic.fetch_openml")
def test_magic_loader_rejects_unexpected_shape(mock_fetch):
    X, y = make_fake_magic()
    mock_fetch.return_value = X.iloc[:-1], y.iloc[:-1]

    with pytest.raises(ValueError, match="shape"):
        load_magic_telescope()