import numpy as np
import pandas as pd
import pytest

from conformalguard.shifts import apply_label_shift


def make_binary_data():
    X = pd.DataFrame(
        {
            "row_id": np.arange(100),
            "value": np.linspace(0.0, 1.0, 100),
        }
    )
    y = pd.Series(
        ["a"] * 50 + ["b"] * 50,
        name="target",
    )
    return X, y


def test_label_shift_reaches_requested_proportions():
    X, y = make_binary_data()

    result = apply_label_shift(
        X,
        y,
        target_proportions={"a": 0.80, "b": 0.20},
        n_samples=100,
        random_state=42,
    )

    counts = result.y_shifted.value_counts().to_dict()

    assert counts == {"a": 80, "b": 20}
    assert len(result.X_shifted) == 100
    assert len(result.y_shifted) == 100


def test_label_shift_is_reproducible():
    X, y = make_binary_data()

    first = apply_label_shift(
        X,
        y,
        target_proportions={"a": 0.70, "b": 0.30},
        random_state=42,
    )
    second = apply_label_shift(
        X,
        y,
        target_proportions={"a": 0.70, "b": 0.30},
        random_state=42,
    )

    pd.testing.assert_frame_equal(first.X_shifted, second.X_shifted)
    pd.testing.assert_series_equal(first.y_shifted, second.y_shifted)


def test_label_shift_preserves_feature_target_pairing():
    X, y = make_binary_data()

    result = apply_label_shift(
        X,
        y,
        target_proportions={"a": 0.25, "b": 0.75},
        random_state=73,
    )

    expected = np.where(
        result.X_shifted["row_id"].to_numpy() < 50,
        "a",
        "b",
    )

    assert np.array_equal(
        result.y_shifted.to_numpy(),
        expected,
    )


def test_invalid_target_proportions_raise_error():
    X, y = make_binary_data()

    with pytest.raises(ValueError):
        apply_label_shift(
            X,
            y,
            target_proportions={"a": 0.60, "b": 0.30},
        )


def test_target_classes_must_match_observed_classes():
    X, y = make_binary_data()

    with pytest.raises(ValueError):
        apply_label_shift(
            X,
            y,
            target_proportions={"a": 1.0},
        )
