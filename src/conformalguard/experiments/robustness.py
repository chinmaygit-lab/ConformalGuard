"""Combined IID and distribution-shift robustness benchmark."""

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import pandas as pd

from conformalguard.experiments.covariate_shift import (
    CovariateShiftExperimentResult,
)
from conformalguard.experiments.covariate_shift_grid import (
    CovariateShiftGridSummary,
    run_covariate_shift_grid,
    summarize_covariate_shift_grid,
)
from conformalguard.experiments.iid_conformal import IIDConformalResult
from conformalguard.experiments.iid_grid import (
    IIDGridSummary,
    run_iid_grid,
    summarize_iid_grid,
)
from conformalguard.experiments.label_shift import (
    LabelShiftExperimentResult,
)
from conformalguard.experiments.label_shift_grid import (
    LabelShiftGridSummary,
    run_label_shift_grid,
    summarize_label_shift_grid,
)


@dataclass(frozen=True)
class RobustnessBenchmarkResult:
    """Combined IID, covariate-shift, and label-shift benchmark results."""

    confidence_level: float
    conformity_score: str
    seeds: tuple[int, ...]

    iid_results: tuple[IIDConformalResult, ...]
    iid_summary: IIDGridSummary

    covariate_shift_results: tuple[
        CovariateShiftExperimentResult, ...
    ]
    covariate_shift_summary: tuple[
        CovariateShiftGridSummary, ...
    ]

    label_shift_results: tuple[
        LabelShiftExperimentResult, ...
    ]
    label_shift_summary: tuple[
        LabelShiftGridSummary, ...
    ]


def run_robustness_benchmark(
    X: pd.DataFrame,
    y: Any,
    *,
    label_target_proportions: Iterable[Mapping[Any, float]],
    covariate_severities: Iterable[float] = (0.0, 0.5, 1.0, 2.0),
    seeds: Iterable[int] = (11, 42, 73),
    confidence_level: float = 0.90,
    feature_fraction: float = 0.50,
    conformity_score: str = "lac",
) -> RobustnessBenchmarkResult:
    """Run the parallel IID, covariate-shift, and label-shift grids."""

    severity_values = tuple(covariate_severities)
    target_values = tuple(
        dict(target)
        for target in label_target_proportions
    )
    random_seeds = tuple(seeds)

    if not severity_values:
        raise ValueError(
            "At least one covariate-shift severity is required."
        )

    if not target_values:
        raise ValueError(
            "At least one label target proportion mapping is required."
        )

    if not random_seeds:
        raise ValueError("At least one random seed is required.")

    iid_results = run_iid_grid(
        X,
        y,
        confidence_levels=(confidence_level,),
        seeds=random_seeds,
        conformity_score=conformity_score,
    )

    covariate_results = run_covariate_shift_grid(
        X,
        y,
        severities=severity_values,
        seeds=random_seeds,
        confidence_level=confidence_level,
        feature_fraction=feature_fraction,
        conformity_score=conformity_score,
    )

    label_results = run_label_shift_grid(
        X,
        y,
        target_proportions=target_values,
        seeds=random_seeds,
        confidence_level=confidence_level,
        conformity_score=conformity_score,
    )

    iid_summary = summarize_iid_grid(iid_results)
    covariate_summary = summarize_covariate_shift_grid(
        covariate_results
    )
    label_summary = summarize_label_shift_grid(label_results)

    return RobustnessBenchmarkResult(
        confidence_level=confidence_level,
        conformity_score=conformity_score,
        seeds=random_seeds,
        iid_results=iid_results,
        iid_summary=iid_summary[0],
        covariate_shift_results=covariate_results,
        covariate_shift_summary=covariate_summary,
        label_shift_results=label_results,
        label_shift_summary=label_summary,
    )
