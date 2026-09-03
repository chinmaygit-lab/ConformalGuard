import numpy as np
import pandas as pd
import pytest

from conformalguard.shifts import apply_covariate_mean_shift


def make_features():
    return pd.DataFrame(
        {
            "a": np.arange(100, dtype=float),
            "b": np.arange(100, dtype=float) * 2,
            "c": np.arange(100, dtype=float) * 3,
            "d": np.arange(100, dtype=float) * 4,
        }
    )


def test_covariate_shift_is_reproducible():
    X = make_features()

    first = apply_covariate_mean_shift(
        X,
        X,
        severity=1.0,
        feature_fraction=0.50,
        random_state=42,
    )

    second = apply_covariate_mean_shift(
        X,
        X,
        severity=1.0,
        feature_fraction=0.50,
        random_state=42,
    )

    assert first.shifted_columns == second.shifted_columns
    pd.testing.assert_frame_equal(
        first.X_shifted,
        second.X_shifted,
    )


def test_covariate_shift_changes_expected_number_of_features():
    X = make_features()

    result = apply_covariate_mean_shift(
        X,
        X,
        severity=1.0,
        feature_fraction=0.50,
        random_state=42,
    )

    assert len(result.shifted_columns) == 2

    unchanged = set(X.columns) - set(result.shifted_columns)

    for column in unchanged:
        assert np.array_equal(
            X[column].to_numpy(),
            result.X_shifted[column].to_numpy(),
        )


def test_zero_severity_leaves_values_unchanged():
    X = make_features()

    result = apply_covariate_mean_shift(
        X,
        X,
        severity=0.0,
        random_state=42,
    )

    pd.testing.assert_frame_equal(X, result.X_shifted)


def test_invalid_feature_fraction_raises_error():
    X = make_features()

    with pytest.raises(ValueError):
        apply_covariate_mean_shift(
            X,
            X,
            severity=1.0,
            feature_fraction=0.0,
        )


def test_negative_severity_raises_error():
    X = make_features()

    with pytest.raises(ValueError):
        apply_covariate_mean_shift(
            X,
            X,
            severity=-1.0,
        )