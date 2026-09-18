import pandas as pd
import pytest
from sklearn.datasets import make_classification

from conformalguard.experiments.concept_shift import (
    run_concept_shift_sweep,
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


def test_concept_shift_sweep_returns_one_result_per_severity():
    X, y = make_binary_dataset()

    severities = (0.0, 0.25, 0.50, 1.0)

    results = run_concept_shift_sweep(
        X,
        y,
        severities=severities,
        random_state=42,
    )

    assert len(results) == len(severities)
    assert [result.severity for result in results] == list(severities)


def test_concept_shift_sweep_records_shift_metadata():
    X, y = make_binary_dataset()

    result = run_concept_shift_sweep(
        X,
        y,
        severities=(0.50,),
        feature="x2",
        random_state=42,
    )[0]

    assert result.shifted_feature == "x2"
    assert isinstance(result.threshold, float)
    assert 0 < result.n_shifted <= result.n_test


def test_zero_concept_shift_records_no_changed_labels():
    X, y = make_binary_dataset()

    result = run_concept_shift_sweep(
        X,
        y,
        severities=(0.0,),
        random_state=42,
    )[0]

    assert result.severity == 0.0
    assert result.n_shifted == 0


def test_concept_shift_sweep_returns_valid_metrics():
    X, y = make_binary_dataset()

    result = run_concept_shift_sweep(
        X,
        y,
        severities=(0.75,),
        confidence_level=0.90,
        random_state=42,
    )[0]

    assert 0.0 <= result.classification.accuracy <= 1.0
    assert 0.0 <= result.classification.macro_f1 <= 1.0
    assert 0.0 <= result.conformal.coverage <= 1.0
    assert result.conformal.coverage_gap >= 0.0
    assert result.conformal.average_set_size >= 0.0
    assert 0.0 <= result.conformal.empty_set_rate <= 1.0


def test_concept_shift_sweep_rejects_empty_severities():
    X, y = make_binary_dataset()

    with pytest.raises(ValueError, match="severity"):
        run_concept_shift_sweep(
            X,
            y,
            severities=(),
        )


def test_concept_shift_sweep_rejects_invalid_severity():
    X, y = make_binary_dataset()

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        run_concept_shift_sweep(
            X,
            y,
            severities=(-0.10,),
        )


def test_concept_shift_sweep_rejects_aps_for_binary_target():
    X, y = make_binary_dataset()

    with pytest.raises(ValueError, match="Binary"):
        run_concept_shift_sweep(
            X,
            y,
            severities=(0.50,),
            conformity_score="aps",
        )
