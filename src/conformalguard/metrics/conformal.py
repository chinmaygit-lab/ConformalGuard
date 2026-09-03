"""Metrics for conformal classification prediction sets."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConformalMetrics:
    """Reliability and efficiency metrics for prediction sets."""

    coverage: float
    coverage_gap: float
    average_set_size: float
    empty_set_rate: float


def evaluate_prediction_sets(
    y_true: Any,
    prediction_sets: Any,
    *,
    target_coverage: float,
    classes: Any | None = None,
) -> ConformalMetrics:
    """Evaluate marginal coverage and prediction-set efficiency."""

    y_true = np.asarray(y_true)
    prediction_sets = np.asarray(prediction_sets, dtype=bool)

    if prediction_sets.ndim != 2:
        raise ValueError("prediction_sets must have shape (n_samples, n_classes).")

    if len(y_true) != prediction_sets.shape[0]:
        raise ValueError("y_true and prediction_sets must contain the same number of samples.")

    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target_coverage must be between 0 and 1.")

    if classes is None:
        classes = np.arange(prediction_sets.shape[1])
    else:
        classes = np.asarray(classes)

    if len(classes) != prediction_sets.shape[1]:
        raise ValueError("classes must match the number of prediction-set columns.")

    class_to_index = {
        label: index
        for index, label in enumerate(classes.tolist())
    }

    try:
        true_class_indices = np.array(
            [class_to_index[label] for label in y_true.tolist()],
            dtype=int,
        )
    except KeyError as exc:
        raise ValueError(
            f"Unknown class label in y_true: {exc.args[0]!r}"
        ) from exc

    covered = prediction_sets[
        np.arange(len(y_true)),
        true_class_indices,
    ]

    set_sizes = prediction_sets.sum(axis=1)

    coverage = float(np.mean(covered))

    return ConformalMetrics(
        coverage=coverage,
        coverage_gap=abs(coverage - target_coverage),
        average_set_size=float(np.mean(set_sizes)),
        empty_set_rate=float(np.mean(set_sizes == 0)),
    )