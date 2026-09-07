import pandas as pd
import pytest
from sklearn.datasets import make_classification

from conformalguard.experiments.label_shift import (
    run_label_shift_sweep,
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


def test_label_shift_sweep_returns_one_result_per_target():
    X, y = make_binary_dataset()

    targets = (
        {0: 0.50, 1: 0.50},
        {0: 0.75, 1: 0.25},
        {0: 0.25, 1: 0.75},
    )

    results = run_label_shift_sweep(
        X,
        y,
        target_proportions=targets,
        random_state=42,
    )

    assert len(results) == 3
    assert [result.target_proportions for result in results] == [
        dict(target)
        for target in targets
    ]


def test_label_shift_sweep_records_requested_sample_counts():
    X, y = make_binary_dataset()

    result = run_label_shift_sweep(
        X,
        y,
        target_proportions=(
            {0: 0.75, 1: 0.25},
        ),
        random_state=42,
    )[0]

    assert sum(result.sampled_counts.values()) == result.n_test

    assert result.sampled_counts[0] / result.n_test == pytest.approx(
        0.75,
        abs=1.0 / result.n_test,
    )
    assert result.sampled_counts[1] / result.n_test == pytest.approx(
        0.25,
        abs=1.0 / result.n_test,
    )


def test_label_shift_sweep_returns_valid_metrics():
    X, y = make_binary_dataset()

    result = run_label_shift_sweep(
        X,
        y,
        target_proportions=(
            {0: 0.80, 1: 0.20},
        ),
        confidence_level=0.90,
        random_state=42,
    )[0]

    assert 0.0 <= result.classification.accuracy <= 1.0
    assert 0.0 <= result.classification.macro_f1 <= 1.0
    assert 0.0 <= result.conformal.coverage <= 1.0
    assert result.conformal.coverage_gap >= 0.0
    assert result.conformal.average_set_size >= 0.0
    assert 0.0 <= result.conformal.empty_set_rate <= 1.0


def test_label_shift_sweep_rejects_empty_targets():
    X, y = make_binary_dataset()

    with pytest.raises(ValueError, match="target"):
        run_label_shift_sweep(
            X,
            y,
            target_proportions=(),
        )


def test_label_shift_sweep_rejects_aps_for_binary_target():
    X, y = make_binary_dataset()

    with pytest.raises(ValueError, match="Binary"):
        run_label_shift_sweep(
            X,
            y,
            target_proportions=(
                {0: 0.50, 1: 0.50},
            ),
            conformity_score="aps",
        )