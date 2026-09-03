"""Reproducible experiment pipelines."""

from conformalguard.experiments.covariate_shift import (
    CovariateShiftExperimentResult,
    run_covariate_shift_sweep,
)
from conformalguard.experiments.covariate_shift_grid import (
    CovariateShiftGridSummary,
    run_covariate_shift_grid,
    summarize_covariate_shift_grid,
)
from conformalguard.experiments.iid_baseline import (
    IIDBaselineResult,
    run_iid_baseline,
)
from conformalguard.experiments.iid_conformal import (
    BINARY_CONFORMITY_SCORES,
    SUPPORTED_CONFORMITY_SCORES,
    IIDConformalResult,
    run_iid_conformal,
    run_iid_conformal_benchmark,
)
from conformalguard.experiments.iid_grid import (
    IIDGridSummary,
    run_iid_grid,
    summarize_iid_grid,
)

__all__ = [
    "BINARY_CONFORMITY_SCORES",
    "SUPPORTED_CONFORMITY_SCORES",
    "CovariateShiftExperimentResult",
    "CovariateShiftGridSummary",
    "IIDBaselineResult",
    "IIDConformalResult",
    "IIDGridSummary",
    "run_covariate_shift_grid",
    "run_covariate_shift_sweep",
    "run_iid_baseline",
    "run_iid_conformal",
    "run_iid_conformal_benchmark",
    "run_iid_grid",
    "summarize_covariate_shift_grid",
    "summarize_iid_grid",
]