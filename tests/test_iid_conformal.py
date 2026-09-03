import pytest
from sklearn.datasets import make_classification

from conformalguard.experiments import (
    SUPPORTED_CONFORMITY_SCORES,
    run_iid_conformal,
    run_iid_conformal_benchmark,
)


def make_multiclass_dataset():
    return make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=42,
    )


def test_iid_conformal_returns_expected_partition_sizes():
    X, y = make_multiclass_dataset()

    result = run_iid_conformal(X, y, random_state=42)

    assert result.n_train == 600
    assert result.n_conf == 200
    assert result.n_test == 200


def test_iid_conformal_returns_valid_metrics():
    X, y = make_multiclass_dataset()

    result = run_iid_conformal(
        X,
        y,
        confidence_level=0.90,
        random_state=42,
    )

    assert 0.0 <= result.classification.accuracy <= 1.0
    assert 0.0 <= result.classification.macro_f1 <= 1.0
    assert 0.0 <= result.conformal.coverage <= 1.0
    assert 0.0 <= result.conformal.coverage_gap <= 1.0
    assert result.conformal.average_set_size >= 0.0
    assert 0.0 <= result.conformal.empty_set_rate <= 1.0


def test_iid_conformal_is_reproducible():
    X, y = make_multiclass_dataset()

    first = run_iid_conformal(X, y, random_state=42)
    second = run_iid_conformal(X, y, random_state=42)

    assert first.conformal.coverage == pytest.approx(second.conformal.coverage)
    assert first.conformal.average_set_size == pytest.approx(
        second.conformal.average_set_size
    )


def test_invalid_confidence_level_raises_error():
    X, y = make_multiclass_dataset()

    with pytest.raises(ValueError):
        run_iid_conformal(
            X,
            y,
            confidence_level=1.20,
        )

def test_iid_conformal_benchmark_runs_all_supported_methods():
    X, y = make_multiclass_dataset()

    results = run_iid_conformal_benchmark(
        X,
        y,
        confidence_level=0.90,
        random_state=42,
    )

    assert tuple(result.conformity_score for result in results) == (
        SUPPORTED_CONFORMITY_SCORES
    )

    reference_accuracy = results[0].classification.accuracy
    reference_macro_f1 = results[0].classification.macro_f1

    for result in results:
        assert result.confidence_level == pytest.approx(0.90)
        assert result.classification.accuracy == pytest.approx(
            reference_accuracy
        )
        assert result.classification.macro_f1 == pytest.approx(
            reference_macro_f1
        )


def test_iid_conformal_benchmark_rejects_empty_method_list():
    X, y = make_multiclass_dataset()

    with pytest.raises(ValueError):
        run_iid_conformal_benchmark(
            X,
            y,
            conformity_scores=[],
        )


def test_invalid_conformity_score_raises_error():
    X, y = make_multiclass_dataset()

    with pytest.raises(ValueError):
        run_iid_conformal(
            X,
            y,
            conformity_score="unknown",
        )