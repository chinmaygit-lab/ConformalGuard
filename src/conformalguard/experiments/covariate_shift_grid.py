"""Multi-seed covariate-shift experiments and summaries."""

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from conformalguard.experiments.covariate_shift import (
    CovariateShiftExperimentResult,
    run_covariate_shift_sweep,
)


@dataclass(frozen=True)
class CovariateShiftGridSummary:
    """Aggregate metrics for one covariate-shift severity."""

    severity: float
    n_runs: int
    confidence_level: float
    feature_fraction: float
    conformity_score: str
    mean_accuracy: float
    std_accuracy: float
    mean_macro_f1: float
    std_macro_f1: float
    mean_coverage: float
    std_coverage: float
    mean_coverage_gap: float
    std_coverage_gap: float
    mean_set_size: float
    std_set_size: float
    mean_empty_set_rate: float
    std_empty_set_rate: float


def run_covariate_shift_grid(
    X: pd.DataFrame,
    y: Any,
    *,
    severities: Iterable[float] = (0.0, 0.5, 1.0, 2.0),
    seeds: Iterable[int] = (11, 42, 73),
    confidence_level: float = 0.90,
    feature_fraction: float = 0.50,
    conformity_score: str = "lac",
) -> tuple[CovariateShiftExperimentResult, ...]:
    """Run covariate-shift sweeps across multiple random seeds."""

    severity_values = tuple(severities)
    random_seeds = tuple(seeds)

    if not severity_values:
        raise ValueError("At least one shift severity is required.")

    if not random_seeds:
        raise ValueError("At least one random seed is required.")

    results = []

    for seed in random_seeds:
        results.extend(
            run_covariate_shift_sweep(
                X,
                y,
                severities=severity_values,
                confidence_level=confidence_level,
                feature_fraction=feature_fraction,
                conformity_score=conformity_score,
                random_state=seed,
            )
        )

    return tuple(results)


def summarize_covariate_shift_grid(
    results: Iterable[CovariateShiftExperimentResult],
) -> tuple[CovariateShiftGridSummary, ...]:
    """Aggregate covariate-shift results by severity."""

    results = tuple(results)

    if not results:
        raise ValueError("At least one result is required.")

    confidence_levels = {
        result.confidence_level
        for result in results
    }
    feature_fractions = {
        result.feature_fraction
        for result in results
    }
    conformity_scores = {
        result.conformity_score
        for result in results
    }

    if len(confidence_levels) != 1:
        raise ValueError(
            "All results must use the same confidence level."
        )

    if len(feature_fractions) != 1:
        raise ValueError(
            "All results must use the same feature fraction."
        )

    if len(conformity_scores) != 1:
        raise ValueError(
            "All results must use the same conformity score."
        )

    confidence_level = next(iter(confidence_levels))
    feature_fraction = next(iter(feature_fractions))
    conformity_score = next(iter(conformity_scores))

    summaries = []

    for severity in sorted(
        {result.severity for result in results}
    ):
        group = tuple(
            result
            for result in results
            if result.severity == severity
        )

        def values(getter):
            return np.asarray(
                [getter(result) for result in group],
                dtype=float,
            )

        accuracy = values(
            lambda r: r.classification.accuracy
        )
        macro_f1 = values(
            lambda r: r.classification.macro_f1
        )
        coverage = values(
            lambda r: r.conformal.coverage
        )
        coverage_gap = values(
            lambda r: r.conformal.coverage_gap
        )
        set_size = values(
            lambda r: r.conformal.average_set_size
        )
        empty_rate = values(
            lambda r: r.conformal.empty_set_rate
        )

        ddof = 1 if len(group) > 1 else 0

        summaries.append(
            CovariateShiftGridSummary(
                severity=severity,
                n_runs=len(group),
                confidence_level=confidence_level,
                feature_fraction=feature_fraction,
                conformity_score=conformity_score,
                mean_accuracy=float(np.mean(accuracy)),
                std_accuracy=float(
                    np.std(accuracy, ddof=ddof)
                ),
                mean_macro_f1=float(np.mean(macro_f1)),
                std_macro_f1=float(
                    np.std(macro_f1, ddof=ddof)
                ),
                mean_coverage=float(np.mean(coverage)),
                std_coverage=float(
                    np.std(coverage, ddof=ddof)
                ),
                mean_coverage_gap=float(
                    np.mean(coverage_gap)
                ),
                std_coverage_gap=float(
                    np.std(coverage_gap, ddof=ddof)
                ),
                mean_set_size=float(np.mean(set_size)),
                std_set_size=float(
                    np.std(set_size, ddof=ddof)
                ),
                mean_empty_set_rate=float(
                    np.mean(empty_rate)
                ),
                std_empty_set_rate=float(
                    np.std(empty_rate, ddof=ddof)
                ),
            )
        )

    return tuple(summaries)