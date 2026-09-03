"""IID split-conformal classification experiment."""

from dataclasses import dataclass
from typing import Any

from mapie.classification import SplitConformalClassifier

from conformalguard.data import stratified_train_conf_test_split
from conformalguard.metrics import (
    ClassificationMetrics,
    ConformalMetrics,
    evaluate_classifier,
    evaluate_prediction_sets,
)
from conformalguard.models import make_logistic_regression


@dataclass(frozen=True)
class IIDConformalResult:
    """Results from an IID split-conformal classification experiment."""

    n_train: int
    n_conf: int
    n_test: int
    confidence_level: float
    conformity_score: str
    classification: ClassificationMetrics
    conformal: ConformalMetrics


def run_iid_conformal(
    X: Any,
    y: Any,
    *,
    confidence_level: float = 0.90,
    conformity_score: str = "lac",
    random_state: int = 42,
) -> IIDConformalResult:
    """Run a split-conformal IID experiment using logistic regression."""

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    split = stratified_train_conf_test_split(
        X,
        y,
        random_state=random_state,
    )

    model = make_logistic_regression()
    model.fit(split.X_train, split.y_train)

    classification_metrics = evaluate_classifier(
        model,
        split.X_test,
        split.y_test,
    )

    conformal_model = SplitConformalClassifier(
        estimator=model,
        confidence_level=confidence_level,
        conformity_score=conformity_score,
        prefit=True,
    )

    conformal_model.conformalize(
        split.X_conf,
        split.y_conf,
    )

    _, prediction_sets = conformal_model.predict_set(split.X_test)

    prediction_sets = prediction_sets[:, :, 0]

    conformal_metrics = evaluate_prediction_sets(
        split.y_test,
        prediction_sets,
        target_coverage=confidence_level,
        classes=model.classes_,
    )

    return IIDConformalResult(
        n_train=len(split.y_train),
        n_conf=len(split.y_conf),
        n_test=len(split.y_test),
        confidence_level=confidence_level,
        conformity_score=conformity_score,
        classification=classification_metrics,
        conformal=conformal_metrics,
    )