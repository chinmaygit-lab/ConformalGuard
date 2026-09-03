"""IID split-conformal classification experiments."""

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from mapie.classification import SplitConformalClassifier

from conformalguard.data import stratified_train_conf_test_split
from conformalguard.metrics import (
    ClassificationMetrics,
    ConformalMetrics,
    evaluate_classifier,
    evaluate_prediction_sets,
)
from conformalguard.models import make_logistic_regression


SUPPORTED_CONFORMITY_SCORES = ("lac", "aps", "raps")
BINARY_CONFORMITY_SCORES = ("lac",)


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


def _valid_conformity_scores(y: Any) -> tuple[str, ...]:
    """Return conformity scores supported for the supplied target."""

    target = np.asarray(y)

    if target.ndim != 1:
        raise ValueError("Classification target must be one-dimensional.")

    n_classes = len(np.unique(target))

    if n_classes < 2:
        raise ValueError("Classification requires at least two target classes.")

    if n_classes == 2:
        return BINARY_CONFORMITY_SCORES

    return SUPPORTED_CONFORMITY_SCORES


def run_iid_conformal(
    X: Any,
    y: Any,
    *,
    confidence_level: float = 0.90,
    conformity_score: str = "lac",
    random_state: int = 42,
) -> IIDConformalResult:
    """Run one split-conformal IID experiment."""

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    valid_scores = _valid_conformity_scores(y)

    if conformity_score not in valid_scores:
        raise ValueError(
            f"conformity_score {conformity_score!r} is not valid for this "
            f"target. Valid scores are {valid_scores}."
        )

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
        random_state=random_state,
    )

    conformal_model.conformalize(
        split.X_conf,
        split.y_conf,
    )

    _, prediction_sets = conformal_model.predict_set(split.X_test)

    conformal_metrics = evaluate_prediction_sets(
        split.y_test,
        prediction_sets[:, :, 0],
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


def run_iid_conformal_benchmark(
    X: Any,
    y: Any,
    *,
    confidence_level: float = 0.90,
    conformity_scores: Iterable[str] | None = None,
    random_state: int = 42,
) -> tuple[IIDConformalResult, ...]:
    """Compare valid conformal methods under identical IID settings."""

    valid_scores = _valid_conformity_scores(y)

    if conformity_scores is None:
        methods = valid_scores
    else:
        methods = tuple(conformity_scores)

    if not methods:
        raise ValueError("At least one conformity score is required.")

    invalid_methods = tuple(
        method for method in methods if method not in valid_scores
    )

    if invalid_methods:
        raise ValueError(
            f"Invalid conformity scores {invalid_methods} for this target. "
            f"Valid scores are {valid_scores}."
        )

    return tuple(
        run_iid_conformal(
            X,
            y,
            confidence_level=confidence_level,
            conformity_score=method,
            random_state=random_state,
        )
        for method in methods
    )