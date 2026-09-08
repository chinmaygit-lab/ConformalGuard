"""Multi-seed label-shift experiments and summaries."""

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from conformalguard.experiments.label_shift import (
    LabelShiftExperimentResult,
    run_label_shift_sweep,
)


@dataclass(frozen=True)
class LabelShiftGridSummary:
    """Aggregate metrics for one target label distribution."""

    target_proportions: dict[Any, float]
    n_runs: int
    confidence_level: float
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


def run_label_shift_grid(
    X: pd.DataFrame,
    y: Any,
    *,
    target_proportions: Iterable[Mapping[Any, float]],
    seeds: Iterable[int] = (11, 42, 73),
    confidence_level: float = 0.90,
    conformity_score: str = "lac",
) -> tuple[LabelShiftExperimentResult, ...]:
    """Run label-shift sweeps across multiple random seeds."""

    target_values = tuple(
        dict(target)
        for target in target_proportions
    )
    random_seeds = tuple(seeds)

    if not target_values:
        raise ValueError(
            "At least one target proportion mapping is required."
        )

    if not random_seeds:
        raise ValueError("At least one random seed is required.")

    results = []

    for seed in random_seeds:
        results.extend(
            run_label_shift_sweep(
                X,
                y,
                target_proportions=target_values,
                confidence_level=confidence_level,
                conformity_score=conformity_score,
                random_state=seed,
            )
        )

    return tuple(results)


def _target_key(
    target_proportions: Mapping[Any, float],
) -> tuple[tuple[Any, float], ...]:
    """Return a deterministic hashable key for a label distribution."""

    return tuple(
        sorted(
            (
                (label, float(proportion))
                for label, proportion in target_proportions.items()
            ),
            key=lambda item: repr(item[0]),
        )
    )


def summarize_label_shift_grid(
    results: Iterable[LabelShiftExperimentResult],
) -> tuple[LabelShiftGridSummary, ...]:
    """Aggregate label-shift results by target distribution."""

    results = tuple(results)

    if not results:
        raise ValueError("At least one result is required.")

    confidence_levels = {
        result.confidence_level
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

    if len(conformity_scores) != 1:
        raise ValueError(
            "All results must use the same conformity score."
        )

    confidence_level = next(iter(confidence_levels))
    conformity_score = next(iter(conformity_scores))

    summaries = []

    target_keys = []
    seen = set()

    for result in results:
        key = _target_key(result.target_proportions)

        if key not in seen:
            seen.add(key)
            target_keys.append(key)

    for target_key in target_keys:
        group = tuple(
            result
            for result in results
            if _target_key(result.target_proportions) == target_key
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
            LabelShiftGridSummary(
                target_proportions=dict(target_key),
                n_runs=len(group),
                confidence_level=confidence_level,
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