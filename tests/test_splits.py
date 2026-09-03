import numpy as np
import pytest

from conformalguard.data import stratified_train_conf_test_split


def make_dataset():
    X = np.arange(2000).reshape(1000, 2)
    y = np.array([0] * 700 + [1] * 300)
    return X, y


def test_default_split_sizes():
    X, y = make_dataset()

    split = stratified_train_conf_test_split(X, y, random_state=42)

    assert len(split.y_train) == 600
    assert len(split.y_conf) == 200
    assert len(split.y_test) == 200


def test_split_preserves_class_proportions():
    X, y = make_dataset()

    split = stratified_train_conf_test_split(X, y, random_state=42)

    assert np.mean(split.y_train) == pytest.approx(0.30, abs=0.01)
    assert np.mean(split.y_conf) == pytest.approx(0.30, abs=0.01)
    assert np.mean(split.y_test) == pytest.approx(0.30, abs=0.01)


def test_split_is_reproducible():
    X, y = make_dataset()

    first = stratified_train_conf_test_split(X, y, random_state=42)
    second = stratified_train_conf_test_split(X, y, random_state=42)

    assert np.array_equal(first.X_train, second.X_train)
    assert np.array_equal(first.X_conf, second.X_conf)
    assert np.array_equal(first.X_test, second.X_test)


def test_invalid_split_sizes_raise_error():
    X, y = make_dataset()

    with pytest.raises(ValueError):
        stratified_train_conf_test_split(
            X,
            y,
            train_size=0.50,
            conf_size=0.30,
            test_size=0.30,
        )