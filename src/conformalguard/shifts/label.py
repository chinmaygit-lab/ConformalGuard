"""Controlled label-prior shifts for paired tabular data."""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelShiftResult:
    """Resampled features, targets, and metadata for a label shift."""

    X_shifted: pd.DataFrame
    y_shifted: pd.Series
    target_proportions: dict[Any, float]
    sampled_counts: dict[Any, int]
    n_samples: int


def apply_label_shift(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    target_proportions: Mapping[Any, float],
    n_samples: int | None = None,
    random_state: int = 42,
) -> LabelShiftResult:
    """Resample within observed classes to impose target class proportions."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series.")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")

    if len(y) == 0:
        raise ValueError("X and y must not be empty.")

    observed_classes = tuple(pd.unique(y))

    if set(target_proportions) != set(observed_classes):
        raise ValueError(
            "target_proportions classes must exactly match observed classes."
        )

    proportions = np.asarray(
        [float(target_proportions[label]) for label in observed_classes],
        dtype=float,
    )

    if not np.all(np.isfinite(proportions)):
        raise ValueError("target_proportions must contain finite values.")

    if np.any(proportions < 0.0):
        raise ValueError("target_proportions must be non-negative.")

    if not np.isclose(proportions.sum(), 1.0):
        raise ValueError("target_proportions must sum to 1.")

    if n_samples is None:
        n_samples = len(y)

    if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")

    desired_counts = proportions * int(n_samples)
    counts = np.floor(desired_counts).astype(int)
    remainder = int(n_samples) - int(counts.sum())

    if remainder:
        fractional = desired_counts - counts
        order = np.argsort(-fractional, kind="stable")
        counts[order[:remainder]] += 1

    rng = np.random.default_rng(random_state)
    sampled_positions: list[np.ndarray] = []

    for label, count in zip(observed_classes, counts, strict=True):
        if count == 0:
            continue

        class_positions = np.flatnonzero(y.to_numpy() == label)

        if len(class_positions) == 0:
            raise ValueError(f"Observed class {label!r} has no rows.")

        sampled_positions.append(
            rng.choice(
                class_positions,
                size=int(count),
                replace=True,
            )
        )

    positions = np.concatenate(sampled_positions)
    positions = positions[rng.permutation(len(positions))]

    X_shifted = X.iloc[positions].reset_index(drop=True)
    y_shifted = y.iloc[positions].reset_index(drop=True)

    normalized_targets = {
        label: float(proportion)
        for label, proportion in zip(
            observed_classes,
            proportions,
            strict=True,
        )
    }
    sampled_counts = {
        label: int(count)
        for label, count in zip(
            observed_classes,
            counts,
            strict=True,
        )
    }

    return LabelShiftResult(
        X_shifted=X_shifted,
        y_shifted=y_shifted,
        target_proportions=normalized_targets,
        sampled_counts=sampled_counts,
        n_samples=int(n_samples),
    )
