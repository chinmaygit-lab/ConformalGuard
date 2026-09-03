"""Standard classification metrics."""

from dataclasses import dataclass
from typing import Any

from sklearn.metrics import accuracy_score, f1_score


@dataclass(frozen=True)
class ClassificationMetrics:
    """Point-prediction metrics for a classifier."""

    accuracy: float
    macro_f1: float


def evaluate_classifier(
    model: Any,
    X: Any,
    y: Any,
) -> ClassificationMetrics:
    """Evaluate a fitted classifier using accuracy and macro-F1."""

    predictions = model.predict(X)

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y, predictions)),
        macro_f1=float(f1_score(y, predictions, average="macro")),
    )