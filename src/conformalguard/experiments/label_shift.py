"""Experiments measuring conformal reliability under label shift."""

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

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
from conformalguard.shifts import apply_label_shift


@dataclass(frozen=True)
class LabelShiftExperimentResult:
    """Metrics from one target label distribution."""

    n_train: int
    n_conf: int
    n_test: int
    confidence_level: float
    conformity_score: str
    random_state: int
    target_proportions: dict[Any, float]
    sampled_counts: dict[Any, int]
    classification: ClassificationMetrics
    conformal: ConformalMetrics


def run_label_shift_sweep(
    X: pd.DataFrame,
    y: Any,
    *,
    target_proportions: Iterable[Mapping[Any, float]],
    confidence_level: float = 0.90,
    conformity_score: str = "lac",
    random_state: int = 42,
) -> tuple[LabelShiftExperimentResult, ...]:
    """Evaluate one fitted conformal classifier across label distributions."""

    target_values = tuple(
        dict(target)
        for target in target_proportions
    )

    if not target_values:
        raise ValueError(
            "At least one target proportion mapping is required."
        )

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

    for target_distribution in target_values:
        shifted = apply_label_shift(
            split.X_test,
            split.y_test,
            target_proportions=target_distribution,
            n_samples=len(split.y_test),
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
            LabelShiftExperimentResult(
                n_train=len(split.X_train),
                n_conf=len(split.X_conf),
                n_test=shifted.n_samples,
                confidence_level=confidence_level,
                conformity_score=conformity_score,
                random_state=random_state,
                target_proportions=shifted.target_proportions,
                sampled_counts=shifted.sampled_counts,
                classification=classification,
                conformal=conformal,
            )
        )

    return tuple(results)