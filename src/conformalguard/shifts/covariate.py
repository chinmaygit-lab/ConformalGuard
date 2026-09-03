"""Controlled covariate mean shifts for numeric tabular data."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CovariateShiftResult:
    """Shifted features and metadata describing the intervention."""

    X_shifted: pd.DataFrame
    shifted_columns: tuple[str, ...]
    severity: float
    feature_fraction: float


def apply_covariate_mean_shift(
    X: pd.DataFrame,
    reference_X: pd.DataFrame,
    *,
    severity: float,
    feature_fraction: float = 0.50,
    random_state: int = 42,
) -> CovariateShiftResult:
    """Shift selected numeric features by severity × reference std."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if not isinstance(reference_X, pd.DataFrame):
        raise TypeError("reference_X must be a pandas DataFrame.")

    if tuple(X.columns) != tuple(reference_X.columns):
        raise ValueError("X and reference_X must have identical columns.")

    if severity < 0.0:
        raise ValueError("severity must be non-negative.")

    if not 0.0 < feature_fraction <= 1.0:
        raise ValueError("feature_fraction must be in (0, 1].")

    non_numeric = tuple(
        column
        for column in X.columns
        if not pd.api.types.is_numeric_dtype(X[column])
    )

    if non_numeric:
        raise TypeError(
            f"All features must be numeric; found {non_numeric}."
        )

    n_features = X.shape[1]
    n_shifted = max(1, int(np.ceil(n_features * feature_fraction)))

    rng = np.random.default_rng(random_state)

    selected_indices = np.sort(
        rng.choice(
            n_features,
            size=n_shifted,
            replace=False,
        )
    )

    shifted_columns = tuple(
        X.columns[index]
        for index in selected_indices
    )

    reference_std = reference_X.loc[:, shifted_columns].std(ddof=0)

    X_shifted = X.copy()

    X_shifted.loc[:, shifted_columns] = (
        X_shifted.loc[:, shifted_columns]
        + severity * reference_std
    )

    return CovariateShiftResult(
        X_shifted=X_shifted,
        shifted_columns=shifted_columns,
        severity=severity,
        feature_fraction=feature_fraction,
    )