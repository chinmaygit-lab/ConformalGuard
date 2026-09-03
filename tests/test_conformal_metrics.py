import numpy as np
import pytest

from conformalguard.metrics import evaluate_prediction_sets


def test_prediction_set_metrics_are_computed_correctly():
    y_true = np.array([0, 1, 2, 1])

    prediction_sets = np.array(
        [
            [True, False, False],
            [False, True, False],
            [True, False, False],
            [False, True, True],
        ]
    )

    metrics = evaluate_prediction_sets(
        y_true,
        prediction_sets,
        target_coverage=0.90,
    )

    assert metrics.coverage == pytest.approx(0.75)
    assert metrics.coverage_gap == pytest.approx(0.15)
    assert metrics.average_set_size == pytest.approx(1.25)
    assert metrics.empty_set_rate == pytest.approx(0.0)


def test_empty_prediction_sets_are_measured():
    y_true = np.array([0, 1])

    prediction_sets = np.array(
        [
            [True, False],
            [False, False],
        ]
    )

    metrics = evaluate_prediction_sets(
        y_true,
        prediction_sets,
        target_coverage=0.90,
    )

    assert metrics.coverage == pytest.approx(0.50)
    assert metrics.average_set_size == pytest.approx(0.50)
    assert metrics.empty_set_rate == pytest.approx(0.50)


def test_non_numeric_class_labels_are_supported():
    y_true = np.array(["cat", "dog"])

    prediction_sets = np.array(
        [
            [True, False],
            [False, True],
        ]
    )

    metrics = evaluate_prediction_sets(
        y_true,
        prediction_sets,
        target_coverage=0.90,
        classes=np.array(["cat", "dog"]),
    )

    assert metrics.coverage == pytest.approx(1.0)


def test_invalid_prediction_set_shape_raises_error():
    with pytest.raises(ValueError):
        evaluate_prediction_sets(
            np.array([0, 1]),
            np.array([True, False]),
            target_coverage=0.90,
        )