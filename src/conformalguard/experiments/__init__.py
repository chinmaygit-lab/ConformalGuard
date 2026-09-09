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
from conformalguard.experiments.label_shift import (
    LabelShiftExperimentResult,
    run_label_shift_sweep,
)
from conformalguard.experiments.label_shift_grid import (
    LabelShiftGridSummary,
    run_label_shift_grid,
    summarize_label_shift_grid,
)
from conformalguard.experiments.robustness import (
    RobustnessBenchmarkResult,
    run_robustness_benchmark,
)
from conformalguard.experiments.robustness_reporting import (
    robustness_summary_frame,
    robustness_summary_records,
    write_robustness_summary_csv,
    write_robustness_summary_json,
)

__all__ = [
    "BINARY_CONFORMITY_SCORES",
    "SUPPORTED_CONFORMITY_SCORES",
    "CovariateShiftExperimentResult",
    "CovariateShiftGridSummary",
    "IIDBaselineResult",
    "IIDConformalResult",
    "IIDGridSummary",
    "LabelShiftExperimentResult",
    "LabelShiftGridSummary",
    "RobustnessBenchmarkResult",
    "run_covariate_shift_grid",
    "run_covariate_shift_sweep",
    "run_iid_baseline",
    "run_iid_conformal",
    "run_iid_conformal_benchmark",
    "run_iid_grid",
    "run_label_shift_grid",
    "run_label_shift_sweep",
    "run_robustness_benchmark",
    "robustness_summary_frame",
    "robustness_summary_records",
    "summarize_covariate_shift_grid",
    "summarize_iid_grid",
    "summarize_label_shift_grid",
    "write_robustness_summary_csv",
    "write_robustness_summary_json",
]