import numpy as np
import pandas as pd
import pytest

from conformalguard.shifts import apply_concept_shift


def make_binary_data():
    X = pd.DataFrame(
        {
            "signal": np.arange(100, dtype=float),
            "noise": np.linspace(1.0, 2.0, 100),
        }
    )
    y = pd.Series(
        ["a"] * 50 + ["b"] * 50,
        name="target",
    )
    return X, y


def test_concept_shift_is_reproducible():
    X, y = make_binary_data()

    first = apply_concept_shift(
        X,
        y,
        severity=0.50,
        feature="signal",
        random_state=42,
    )
    second = apply_concept_shift(
        X,
        y,
        severity=0.50,
        feature="signal",
        random_state=42,
    )

    assert first.shifted_positions == second.shifted_positions
    pd.testing.assert_frame_equal(first.X_shifted, second.X_shifted)
    pd.testing.assert_series_equal(first.y_shifted, second.y_shifted)


def test_concept_shift_preserves_features():
    X, y = make_binary_data()

    result = apply_concept_shift(
        X,
        y,
        severity=0.50,
        feature="signal",
        random_state=42,
    )

    pd.testing.assert_frame_equal(X, result.X_shifted)


def test_concept_shift_changes_only_feature_defined_region():
    X, y = make_binary_data()

    result = apply_concept_shift(
        X,
        y,
        severity=1.0,
        feature="signal",
        random_state=42,
    )

    threshold = X["signal"].median()

    assert result.shifted_positions
    assert all(
        X.iloc[position]["signal"] > threshold
        for position in result.shifted_positions
    )


def test_concept_shift_severity_controls_number_changed():
    X, y = make_binary_data()

    result = apply_concept_shift(
        X,
        y,
        severity=0.40,
        feature="signal",
        random_state=42,
    )

    # signal > median selects 50 rows; 40% of 50 = 20.
    assert result.n_shifted == 20
    assert len(result.shifted_positions) == 20


def test_zero_severity_leaves_targets_unchanged():
    X, y = make_binary_data()

    result = apply_concept_shift(
        X,
        y,
        severity=0.0,
        feature="signal",
        random_state=42,
    )

    pd.testing.assert_frame_equal(X, result.X_shifted)
    pd.testing.assert_series_equal(y, result.y_shifted)
    assert result.n_shifted == 0


def test_shifted_rows_receive_different_labels():
    X, y = make_binary_data()

    result = apply_concept_shift(
        X,
        y,
        severity=0.50,
        feature="signal",
        random_state=42,
    )

    for position in result.shifted_positions:
        assert (
            result.y_shifted.iloc[position]
            != y.iloc[position]
        )


def test_invalid_severity_raises_error():
    X, y = make_binary_data()

    with pytest.raises(ValueError):
        apply_concept_shift(
            X,
            y,
            severity=-0.1,
        )

    with pytest.raises(ValueError):
        apply_concept_shift(
            X,
            y,
            severity=1.1,
        )


def test_unknown_feature_raises_error():
    X, y = make_binary_data()

    with pytest.raises(ValueError):
        apply_concept_shift(
            X,
            y,
            severity=0.50,
            feature="missing",
        )


def test_single_class_target_raises_error():
    X, _ = make_binary_data()
    y = pd.Series(["a"] * len(X))

    with pytest.raises(ValueError):
        apply_concept_shift(
            X,
            y,
            severity=0.50,
            feature="signal",
        )
