"""Experiments measuring conformal reliability under concept shift."""

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from mapie.classification import SplitConformalClassifier

from conformalguard.data import stratified_train_conf_test_split
from conformalguard.experiments.iid_conformal import (
    BINARY_CONFORMITY_SCORES,
    SUPPORTED_CONFORMITY_SCORES,
)
from conformalguard.metrics import (
    ClassificationMetrics,
    ConformalMetrics,
    evaluate_classifier,
    evaluate_prediction_sets,
)
from conformalguard.models import make_logistic_regression
from conformalguard.shifts import apply_concept_shift


@dataclass(frozen=True)
class ConceptShiftExperimentResult:
    """Metrics from one concept-shift severity."""

    n_train: int
    n_conf: int
    n_test: int
    confidence_level: float
    conformity_score: str
    random_state: int
    severity: float
    shifted_feature: str
    threshold: float
    n_shifted: int
    classification: ClassificationMetrics
    conformal: ConformalMetrics


def run_concept_shift_sweep(
    X: pd.DataFrame,
    y: Any,
    *,
    severities: Iterable[float] = (0.0, 0.25, 0.50, 0.75, 1.0),
    feature: str | None = None,
    confidence_level: float = 0.90,
    conformity_score: str = "lac",
    random_state: int = 42,
) -> tuple[ConceptShiftExperimentResult, ...]:
    """Evaluate one fitted conformal classifier across concept-shift severities."""

    severity_values = tuple(severities)

    if not severity_values:
        raise ValueError("At least one concept-shift severity is required.")

    for severity in severity_values:
        if not np.isfinite(severity):
            raise ValueError("severity must be finite.")

        if not 0.0 <= severity <= 1.0:
            raise ValueError("severity must be in [0, 1].")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1).")

    if conformity_score not in SUPPORTED_CONFORMITY_SCORES:
        raise ValueError(
            f"Unsupported conformity score: {conformity_score}."
        )

    target = np.asarray(y)

    if target.ndim != 1:
        raise ValueError("y must be one-dimensional.")

    n_classes = len(np.unique(target))

    if n_classes < 2:
        raise ValueError("y must contain at least two classes.")

    if (
        n_classes == 2
        and conformity_score not in BINARY_CONFORMITY_SCORES
    ):
        raise ValueError(
            "Binary targets support only the LAC conformity score."
        )

    split = stratified_train_conf_test_split(
        X,
        y,
        random_state=random_state,
    )

    model = make_logistic_regression()
    model.fit(split.X_train, split.y_train)

    conformal_classifier = SplitConformalClassifier(
        estimator=model,
        confidence_level=confidence_level,
        conformity_score=conformity_score,
        prefit=True,
        random_state=random_state,
    )

    conformal_classifier.conformalize(
        split.X_conf,
        split.y_conf,
    )

    if isinstance(split.y_test, pd.Series):
        y_test = split.y_test
    else:
        y_test = pd.Series(split.y_test)

    results = []

    for severity in severity_values:
        shifted = apply_concept_shift(
            split.X_test,
            y_test,
            severity=float(severity),
            feature=feature,
            random_state=random_state,
        )

        classification = evaluate_classifier(
            model,
            shifted.X_shifted,
            shifted.y_shifted,
        )

        _, prediction_sets = conformal_classifier.predict_set(
            shifted.X_shifted
        )

        conformal = evaluate_prediction_sets(
            shifted.y_shifted,
            prediction_sets[:, :, 0],
            target_coverage=confidence_level,
            classes=model.classes_,
        )

        results.append(
            ConceptShiftExperimentResult(
                n_train=len(split.X_train),
                n_conf=len(split.X_conf),
                n_test=len(split.X_test),
                confidence_level=confidence_level,
                conformity_score=conformity_score,
                random_state=random_state,
                severity=float(severity),
                shifted_feature=shifted.shifted_feature,
                threshold=shifted.threshold,
                n_shifted=shifted.n_shifted,
                classification=classification,
                conformal=conformal,
            )
        )

    return tuple(results)
