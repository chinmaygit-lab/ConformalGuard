"""IID baseline experiment for classification."""

from dataclasses import dataclass
from typing import Any

from conformalguard.data import stratified_train_conf_test_split
from conformalguard.metrics import ClassificationMetrics, evaluate_classifier
from conformalguard.models import make_logistic_regression


@dataclass(frozen=True)
class IIDBaselineResult:
    """Results from an IID logistic-regression baseline experiment."""

    n_train: int
    n_conf: int
    n_test: int
    metrics: ClassificationMetrics


def run_iid_baseline(
    X: Any,
    y: Any,
    *,
    random_state: int = 42,
) -> IIDBaselineResult:
    """Fit and evaluate the initial logistic-regression IID baseline."""

    split = stratified_train_conf_test_split(
        X,
        y,
        random_state=random_state,
    )

    model = make_logistic_regression()
    model.fit(split.X_train, split.y_train)

    metrics = evaluate_classifier(
        model,
        split.X_test,
        split.y_test,
    )

    return IIDBaselineResult(
        n_train=len(split.y_train),
        n_conf=len(split.y_conf),
        n_test=len(split.y_test),
        metrics=metrics,
    )