"""Experiments measuring conformal reliability under covariate shift."""

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
from conformalguard.shifts import apply_covariate_mean_shift


@dataclass(frozen=True)
class CovariateShiftExperimentResult:
    """Metrics from one covariate-shift severity."""

    n_train: int
    n_conf: int
    n_test: int
    confidence_level: float
    conformity_score: str
    random_state: int
    severity: float
    feature_fraction: float
    shifted_columns: tuple[str, ...]
    classification: ClassificationMetrics
    conformal: ConformalMetrics


def run_covariate_shift_sweep(
    X: pd.DataFrame,
    y: Any,
    *,
    severities: Iterable[float] = (0.0, 0.5, 1.0, 2.0),
    confidence_level: float = 0.90,
    feature_fraction: float = 0.50,
    conformity_score: str = "lac",
    random_state: int = 42,
) -> tuple[CovariateShiftExperimentResult, ...]:
    """Evaluate one fitted conformal classifier across shift severities."""

    severity_values = tuple(severities)

    if not severity_values:
        raise ValueError("At least one shift severity is required.")

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

    results = []

    for severity in severity_values:
        shifted = apply_covariate_mean_shift(
            split.X_test,
            split.X_train,
            severity=severity,
            feature_fraction=feature_fraction,
            random_state=random_state,
        )

        classification = evaluate_classifier(
            model,
            shifted.X_shifted,
            split.y_test,
        )

        _, prediction_sets = conformal_classifier.predict_set(
            shifted.X_shifted
        )

        conformal = evaluate_prediction_sets(
            split.y_test,
            prediction_sets[:, :, 0],
            target_coverage=confidence_level,
            classes=model.classes_,
        )

        results.append(
            CovariateShiftExperimentResult(
                n_train=len(split.X_train),
                n_conf=len(split.X_conf),
                n_test=len(split.X_test),
                confidence_level=confidence_level,
                conformity_score=conformity_score,
                random_state=random_state,
                severity=float(severity),
                feature_fraction=feature_fraction,
                shifted_columns=shifted.shifted_columns,
                classification=classification,
                conformal=conformal,
            )
        )

    return tuple(results)