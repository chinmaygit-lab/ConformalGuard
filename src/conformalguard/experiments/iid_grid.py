"""Grid experiments across random seeds and coverage levels."""

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from conformalguard.experiments.iid_conformal import (
    IIDConformalResult,
    run_iid_conformal,
)


@dataclass(frozen=True)
class IIDGridSummary:
    """Aggregate statistics for one target coverage level."""

    confidence_level: float
    n_runs: int
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


def run_iid_grid(
    X: Any,
    y: Any,
    *,
    confidence_levels: Iterable[float] = (0.80, 0.90, 0.95),
    seeds: Iterable[int] = (11, 42, 73),
    conformity_score: str = "lac",
) -> tuple[IIDConformalResult, ...]:
    """Run IID conformal experiments over coverage levels and seeds."""

    levels = tuple(confidence_levels)
    random_seeds = tuple(seeds)

    if not levels:
        raise ValueError("At least one confidence level is required.")

    if not random_seeds:
        raise ValueError("At least one random seed is required.")

    return tuple(
        run_iid_conformal(
            X,
            y,
            confidence_level=level,
            conformity_score=conformity_score,
            random_state=seed,
        )
        for seed in random_seeds
        for level in levels
    )


def summarize_iid_grid(
    results: Iterable[IIDConformalResult],
) -> tuple[IIDGridSummary, ...]:
    """Aggregate grid results by confidence level."""

    results = tuple(results)

    if not results:
        raise ValueError("At least one result is required.")

    summaries = []

    confidence_levels = sorted(
        {result.confidence_level for result in results}
    )

    for level in confidence_levels:
        group = tuple(
            result
            for result in results
            if result.confidence_level == level
        )

        def values(getter):
            return np.asarray(
                [getter(result) for result in group],
                dtype=float,
            )

        accuracy = values(lambda r: r.classification.accuracy)
        macro_f1 = values(lambda r: r.classification.macro_f1)
        coverage = values(lambda r: r.conformal.coverage)
        coverage_gap = values(lambda r: r.conformal.coverage_gap)
        set_size = values(lambda r: r.conformal.average_set_size)
        empty_rate = values(lambda r: r.conformal.empty_set_rate)

        ddof = 1 if len(group) > 1 else 0

        summaries.append(
            IIDGridSummary(
                confidence_level=level,
                n_runs=len(group),
                mean_accuracy=float(np.mean(accuracy)),
                std_accuracy=float(np.std(accuracy, ddof=ddof)),
                mean_macro_f1=float(np.mean(macro_f1)),
                std_macro_f1=float(np.std(macro_f1, ddof=ddof)),
                mean_coverage=float(np.mean(coverage)),
                std_coverage=float(np.std(coverage, ddof=ddof)),
                mean_coverage_gap=float(np.mean(coverage_gap)),
                std_coverage_gap=float(np.std(coverage_gap, ddof=ddof)),
                mean_set_size=float(np.mean(set_size)),
                std_set_size=float(np.std(set_size, ddof=ddof)),
                mean_empty_set_rate=float(np.mean(empty_rate)),
                std_empty_set_rate=float(np.std(empty_rate, ddof=ddof)),
            )
        )

    return tuple(summaries)