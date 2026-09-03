import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from conformalguard.experiments.covariate_shift import (
    run_covariate_shift_sweep,
)


def make_binary_dataset():
    X, y = make_classification(
        n_samples=600,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )

    return (
        pd.DataFrame(
            X,
            columns=[f"x{i}" for i in range(X.shape[1])],
        ),
        pd.Series(y),
    )


def test_covariate_shift_sweep_returns_one_result_per_severity():
    X, y = make_binary_dataset()

    results = run_covariate_shift_sweep(
        X,
        y,
        severities=(0.0, 0.5, 1.0),
        random_state=42,
    )

    assert len(results) == 3
    assert [result.severity for result in results] == [
        0.0,
        0.5,
        1.0,
    ]


def test_covariate_shift_sweep_uses_same_features_across_severities():
    X, y = make_binary_dataset()

    results = run_covariate_shift_sweep(
        X,
        y,
        severities=(0.0, 0.5, 1.0),
        feature_fraction=0.50,
        random_state=42,
    )

    selected = {
        result.shifted_columns
        for result in results
    }

    assert len(selected) == 1
    assert len(results[0].shifted_columns) == 3


def test_covariate_shift_sweep_returns_valid_metrics():
    X, y = make_binary_dataset()

    result = run_covariate_shift_sweep(
        X,
        y,
        severities=(0.0,),
        confidence_level=0.90,
        random_state=42,
    )[0]

    assert 0.0 <= result.classification.accuracy <= 1.0
    assert 0.0 <= result.classification.macro_f1 <= 1.0
    assert 0.0 <= result.conformal.coverage <= 1.0
    assert result.conformal.coverage_gap >= 0.0
    assert result.conformal.average_set_size >= 0.0
    assert 0.0 <= result.conformal.empty_set_rate <= 1.0


def test_covariate_shift_sweep_rejects_empty_severities():
    X, y = make_binary_dataset()

    with pytest.raises(ValueError, match="severity"):
        run_covariate_shift_sweep(
            X,
            y,
            severities=(),
        )


def test_covariate_shift_sweep_rejects_aps_for_binary_target():
    X, y = make_binary_dataset()

    with pytest.raises(ValueError, match="Binary"):
        run_covariate_shift_sweep(
            X,
            y,
            conformity_score="aps",
        )