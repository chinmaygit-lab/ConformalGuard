"""Degradation analysis for robustness benchmark results."""

from pathlib import Path
from typing import Final

import pandas as pd

from conformalguard.experiments.robustness import (
    RobustnessBenchmarkResult,
)
from conformalguard.experiments.robustness_reporting import (
    robustness_summary_frame,
)


_DELTA_METRICS: Final = {
    "accuracy": "mean_accuracy",
    "macro_f1": "mean_macro_f1",
    "coverage": "mean_coverage",
    "coverage_gap": "mean_coverage_gap",
    "set_size": "mean_set_size",
    "empty_set_rate": "mean_empty_set_rate",
}


def robustness_degradation_frame(
    result: RobustnessBenchmarkResult,
) -> pd.DataFrame:
    """Compare every shifted benchmark condition with the IID baseline.

    Delta values are defined as:

        shifted metric - IID metric

    Negative accuracy, macro-F1, or coverage deltas therefore represent
    degradation relative to IID performance.
    """

    frame = robustness_summary_frame(result)

    iid_rows = frame.loc[
        frame["experiment"] == "iid"
    ]

    if len(iid_rows) != 1:
        raise ValueError(
            "Robustness degradation analysis requires exactly "
            "one IID summary row."
        )

    iid = iid_rows.iloc[0]

    degraded = (
        frame.loc[frame["experiment"] != "iid"]
        .copy()
        .reset_index(drop=True)
    )

    for metric, column in _DELTA_METRICS.items():
        baseline_column = f"iid_{column}"
        delta_column = f"{metric}_delta"

        baseline = float(iid[column])

        degraded[baseline_column] = baseline
        degraded[delta_column] = (
            degraded[column].astype(float)
            - baseline
        )

    return degraded


def write_robustness_degradation_csv(
    result: RobustnessBenchmarkResult,
    path: str | Path,
) -> Path:
    """Write IID-relative degradation analysis to CSV."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    robustness_degradation_frame(result).to_csv(
        output,
        index=False,
    )

    return output


def write_robustness_degradation_json(
    result: RobustnessBenchmarkResult,
    path: str | Path,
) -> Path:
    """Write IID-relative degradation analysis to JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    robustness_degradation_frame(result).to_json(
        output,
        orient="records",
        indent=2,
    )

    return output
