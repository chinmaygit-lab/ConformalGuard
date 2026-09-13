"""Controlled feature-dependent concept shifts for tabular data."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConceptShiftResult:
    """Feature-preserving data with conditionally shifted targets."""

    X_shifted: pd.DataFrame
    y_shifted: pd.Series
    shifted_positions: tuple[int, ...]
    shifted_feature: str
    threshold: float
    severity: float
    n_shifted: int


def apply_concept_shift(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    severity: float,
    feature: str | None = None,
    random_state: int = 42,
) -> ConceptShiftResult:
    """Alter labels conditionally within a feature-defined region."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series.")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")

    if len(y) == 0:
        raise ValueError("X and y must not be empty.")

    if not np.isfinite(severity):
        raise ValueError("severity must be finite.")

    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be in [0, 1].")

    observed_classes = tuple(pd.unique(y))

    if len(observed_classes) < 2:
        raise ValueError("y must contain at least two classes.")

    numeric_columns = tuple(
        column
        for column in X.columns
        if pd.api.types.is_numeric_dtype(X[column])
    )

    if not numeric_columns:
        raise TypeError("X must contain at least one numeric feature.")

    if feature is None:
        feature = numeric_columns[0]

    if feature not in X.columns:
        raise ValueError(f"Unknown feature: {feature!r}.")

    if not pd.api.types.is_numeric_dtype(X[feature]):
        raise TypeError(
            "The selected concept-shift feature must be numeric."
        )

    if X[feature].nunique(dropna=True) < 2:
        raise ValueError(
            "The selected feature must contain at least two distinct values."
        )

    if X[feature].isna().any():
        raise ValueError(
            "The selected feature must not contain missing values."
        )

    threshold = float(X[feature].median())

    candidate_positions = np.flatnonzero(
        X[feature].to_numpy(dtype=float) > threshold
    )

    if len(candidate_positions) == 0:
        raise ValueError(
            "The selected feature does not define a non-empty shift region."
        )

    if severity == 0.0:
        n_shifted = 0
    else:
        n_shifted = int(
            np.ceil(len(candidate_positions) * severity)
        )

    rng = np.random.default_rng(random_state)

    if n_shifted:
        shifted_positions_array = np.sort(
            rng.choice(
                candidate_positions,
                size=n_shifted,
                replace=False,
            )
        )
    else:
        shifted_positions_array = np.asarray([], dtype=int)

    class_mapping = {
        label: observed_classes[
            (index + 1) % len(observed_classes)
        ]
        for index, label in enumerate(observed_classes)
    }

    X_shifted = X.copy()
    y_shifted = y.copy()

    for position in shifted_positions_array:
        original_label = y.iloc[int(position)]
        y_shifted.iloc[int(position)] = class_mapping[original_label]

    shifted_positions = tuple(
        int(position)
        for position in shifted_positions_array
    )

    return ConceptShiftResult(
        X_shifted=X_shifted,
        y_shifted=y_shifted,
        shifted_positions=shifted_positions,
        shifted_feature=feature,
        threshold=threshold,
        severity=float(severity),
        n_shifted=n_shifted,
    )
