"""Visualization helpers for robustness benchmark summaries."""

from typing import Final

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from conformalguard.experiments.robustness import (
    RobustnessBenchmarkResult,
)
from conformalguard.experiments.robustness_reporting import (
    robustness_summary_frame,
)


_METRIC_COLUMNS: Final = {
    "accuracy": (
        "mean_accuracy",
        "std_accuracy",
        "Accuracy",
    ),
    "macro_f1": (
        "mean_macro_f1",
        "std_macro_f1",
        "Macro F1",
    ),
    "coverage": (
        "mean_coverage",
        "std_coverage",
        "Coverage",
    ),
    "coverage_gap": (
        "mean_coverage_gap",
        "std_coverage_gap",
        "Coverage gap",
    ),
    "set_size": (
        "mean_set_size",
        "std_set_size",
        "Average prediction-set size",
    ),
    "empty_set_rate": (
        "mean_empty_set_rate",
        "std_empty_set_rate",
        "Empty-set rate",
    ),
}


SUPPORTED_ROBUSTNESS_METRICS = tuple(_METRIC_COLUMNS)


def _condition_label(
    experiment: str,
    condition: str,
) -> str:
    """Return a concise display label for one benchmark condition."""

    if experiment == "iid":
        return "IID"

    if experiment == "covariate_shift":
        return f"Covariate\n{condition}"

    if experiment == "label_shift":
        return f"Label\n{condition}"

    return f"{experiment}\n{condition}"


def plot_robustness_metric(
    result: RobustnessBenchmarkResult,
    metric: str,
    *,
    ax: Axes | None = None,
    show_std: bool = True,
) -> Axes:
    """Plot one aggregate metric across all benchmark conditions."""

    try:
        mean_column, std_column, ylabel = _METRIC_COLUMNS[metric]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_ROBUSTNESS_METRICS)
        raise ValueError(
            f"Unsupported robustness metric {metric!r}. "
            f"Supported metrics: {supported}."
        ) from exc

    frame = robustness_summary_frame(result)

    if ax is None:
        _, ax = plt.subplots()

    positions = list(range(len(frame)))
    values = frame[mean_column].to_numpy(dtype=float)

    errors = None
    if show_std:
        errors = frame[std_column].to_numpy(dtype=float)

    labels = [
        _condition_label(
            experiment,
            condition,
        )
        for experiment, condition in zip(
            frame["experiment"],
            frame["condition"],
        )
    ]

    ax.bar(
        positions,
        values,
        yerr=errors,
        capsize=4 if show_std else 0,
    )

    ax.set_xticks(
        positions,
        labels,
    )
    ax.set_xlabel("Experiment condition")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Robustness benchmark: {ylabel}")

    return ax


def plot_robustness_accuracy(
    result: RobustnessBenchmarkResult,
    *,
    ax: Axes | None = None,
    show_std: bool = True,
) -> Axes:
    """Plot mean classification accuracy across benchmark conditions."""

    return plot_robustness_metric(
        result,
        "accuracy",
        ax=ax,
        show_std=show_std,
    )


def plot_robustness_coverage(
    result: RobustnessBenchmarkResult,
    *,
    ax: Axes | None = None,
    show_std: bool = True,
) -> Axes:
    """Plot mean conformal coverage across benchmark conditions."""

    return plot_robustness_metric(
        result,
        "coverage",
        ax=ax,
        show_std=show_std,
    )


def plot_robustness_coverage_gap(
    result: RobustnessBenchmarkResult,
    *,
    ax: Axes | None = None,
    show_std: bool = True,
) -> Axes:
    """Plot mean coverage gap across benchmark conditions."""

    return plot_robustness_metric(
        result,
        "coverage_gap",
        ax=ax,
        show_std=show_std,
    )


def plot_robustness_set_size(
    result: RobustnessBenchmarkResult,
    *,
    ax: Axes | None = None,
    show_std: bool = True,
) -> Axes:
    """Plot mean prediction-set size across benchmark conditions."""

    return plot_robustness_metric(
        result,
        "set_size",
        ax=ax,
        show_std=show_std,
    )
