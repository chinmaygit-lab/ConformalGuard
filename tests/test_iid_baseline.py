import pytest
from sklearn.datasets import make_classification

from conformalguard.experiments import run_iid_baseline


def test_iid_baseline_returns_expected_partition_sizes():
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        class_sep=1.5,
        random_state=42,
    )

    result = run_iid_baseline(X, y, random_state=42)

    assert result.n_train == 600
    assert result.n_conf == 200
    assert result.n_test == 200


def test_iid_baseline_produces_valid_metrics():
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        class_sep=1.5,
        random_state=42,
    )

    result = run_iid_baseline(X, y, random_state=42)

    assert 0.0 <= result.metrics.accuracy <= 1.0
    assert 0.0 <= result.metrics.macro_f1 <= 1.0
    assert result.metrics.accuracy > 0.70
    assert result.metrics.macro_f1 > 0.70


def test_iid_baseline_is_reproducible():
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        class_sep=1.5,
        random_state=42,
    )

    first = run_iid_baseline(X, y, random_state=42)
    second = run_iid_baseline(X, y, random_state=42)

    assert first.metrics.accuracy == pytest.approx(second.metrics.accuracy)
    assert first.metrics.macro_f1 == pytest.approx(second.metrics.macro_f1)