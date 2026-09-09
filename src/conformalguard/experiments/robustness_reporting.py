"""Reporting helpers for combined robustness benchmarks."""

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from conformalguard.experiments.robustness import (
    RobustnessBenchmarkResult,
)


ROBUSTNESS_SUMMARY_COLUMNS = (
    "experiment",
    "condition",
    "confidence_level",
    "conformity_score",
    "n_runs",
    "severity",
    "feature_fraction",
    "target_proportions",
    "mean_accuracy",
    "std_accuracy",
    "mean_macro_f1",
    "std_macro_f1",
    "mean_coverage",
    "std_coverage",
    "mean_coverage_gap",
    "std_coverage_gap",
    "mean_set_size",
    "std_set_size",
    "mean_empty_set_rate",
    "std_empty_set_rate",
)


def _format_target_proportions(
    target_proportions: Mapping[Any, float],
) -> str:
    """Return a deterministic readable label-distribution string."""

    items = sorted(
        target_proportions.items(),
        key=lambda item: repr(item[0]),
    )

    return ", ".join(
        f"{label!r}:{float(proportion):.12g}"
        for label, proportion in items
    )


def _metric_values(summary: Any) -> dict[str, Any]:
    """Return the common aggregate metrics from one grid summary."""

    return {
        "n_runs": summary.n_runs,
        "mean_accuracy": summary.mean_accuracy,
        "std_accuracy": summary.std_accuracy,
        "mean_macro_f1": summary.mean_macro_f1,
        "std_macro_f1": summary.std_macro_f1,
        "mean_coverage": summary.mean_coverage,
        "std_coverage": summary.std_coverage,
        "mean_coverage_gap": summary.mean_coverage_gap,
        "std_coverage_gap": summary.std_coverage_gap,
        "mean_set_size": summary.mean_set_size,
        "std_set_size": summary.std_set_size,
        "mean_empty_set_rate": summary.mean_empty_set_rate,
        "std_empty_set_rate": summary.std_empty_set_rate,
    }


def robustness_summary_records(
    result: RobustnessBenchmarkResult,
) -> tuple[dict[str, Any], ...]:
    """Convert a benchmark result into normalized summary records."""

    records: list[dict[str, Any]] = []

    iid = result.iid_summary
    records.append(
        {
            "experiment": "iid",
            "condition": "iid",
            "confidence_level": iid.confidence_level,
            "conformity_score": result.conformity_score,
            "severity": None,
            "feature_fraction": None,
            "target_proportions": None,
            **_metric_values(iid),
        }
    )

    for summary in result.covariate_shift_summary:
        records.append(
            {
                "experiment": "covariate_shift",
                "condition": (
                    f"severity={float(summary.severity):.12g}"
                ),
                "confidence_level": summary.confidence_level,
                "conformity_score": summary.conformity_score,
                "severity": summary.severity,
                "feature_fraction": summary.feature_fraction,
                "target_proportions": None,
                **_metric_values(summary),
            }
        )

    for summary in result.label_shift_summary:
        target_text = _format_target_proportions(
            summary.target_proportions
        )

        records.append(
            {
                "experiment": "label_shift",
                "condition": f"target={target_text}",
                "confidence_level": summary.confidence_level,
                "conformity_score": summary.conformity_score,
                "severity": None,
                "feature_fraction": None,
                "target_proportions": target_text,
                **_metric_values(summary),
            }
        )

    return tuple(records)


def robustness_summary_frame(
    result: RobustnessBenchmarkResult,
) -> pd.DataFrame:
    """Return a tidy DataFrame containing benchmark summaries."""

    return pd.DataFrame.from_records(
        robustness_summary_records(result),
        columns=ROBUSTNESS_SUMMARY_COLUMNS,
    )


def write_robustness_summary_csv(
    result: RobustnessBenchmarkResult,
    path: str | Path,
) -> Path:
    """Write normalized benchmark summaries to CSV."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    robustness_summary_frame(result).to_csv(
        output,
        index=False,
    )

    return output


def write_robustness_summary_json(
    result: RobustnessBenchmarkResult,
    path: str | Path,
) -> Path:
    """Write normalized benchmark summaries to JSON records."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    robustness_summary_frame(result).to_json(
        output,
        orient="records",
        indent=2,
    )

    return output
