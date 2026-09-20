"""Visualization helpers for IID-relative robustness degradation."""

from typing import Final

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from conformalguard.experiments.robustness import (
    RobustnessBenchmarkResult,
)
from conformalguard.experiments.robustness_degradation import (
    robustness_degradation_frame,
)
from conformalguard.experiments.robustness_plotting import (
    _condition_label,
)


_DEGRADATION_METRICS: Final = {
    "accuracy": (
        "accuracy_delta",
        "Accuracy delta vs IID",
        "Accuracy",
    ),
    "macro_f1": (
        "macro_f1_delta",
        "Macro F1 delta vs IID",
        "Macro F1",
    ),
    "coverage": (
        "coverage_delta",
        "Coverage delta vs IID",
        "Coverage",
    ),
    "coverage_gap": (
        "coverage_gap_delta",
        "Coverage-gap delta vs IID",
        "Coverage gap",
    ),
    "set_size": (
        "set_size_delta",
        "Set-size delta vs IID",
        "Prediction-set size",
    ),
    "empty_set_rate": (
        "empty_set_rate_delta",
        "Empty-set-rate delta vs IID",
        "Empty-set rate",
    ),
}


SUPPORTED_DEGRADATION_METRICS = tuple(
    _DEGRADATION_METRICS
)


def plot_robustness_degradation_metric(
    result: RobustnessBenchmarkResult,
    metric: str,
    *,
    ax: Axes | None = None,
    show_values: bool = True,
) -> Axes:
    """Plot one robustness metric delta relative to IID."""

    try:
        delta_column, ylabel, title_metric = (
            _DEGRADATION_METRICS[metric]
        )
    except KeyError as exc:
        supported = ", ".join(
            SUPPORTED_DEGRADATION_METRICS
        )
        raise ValueError(
            f"Unsupported degradation metric {metric!r}. "
            f"Supported metrics: {supported}."
        ) from exc

    frame = robustness_degradation_frame(result)

    if ax is None:
        _, ax = plt.subplots()

    positions = list(range(len(frame)))

    values = frame[delta_column].to_numpy(
        dtype=float
    )

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

    bars = ax.bar(
        positions,
        values,
    )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_xticks(
        positions,
        labels,
    )

    ax.set_xlabel("Shift condition")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"Robustness degradation vs IID: {title_metric}"
    )

    if show_values:
        for bar, value in zip(bars, values):
            vertical_offset = (
                3
                if value >= 0.0
                else -3
            )

            vertical_alignment = (
                "bottom"
                if value >= 0.0
                else "top"
            )

            ax.annotate(
                f"{value:+.3f}",
                xy=(
                    bar.get_x()
                    + bar.get_width() / 2,
                    value,
                ),
                xytext=(0, vertical_offset),
                textcoords="offset points",
                ha="center",
                va=vertical_alignment,
            )

    return ax


def plot_robustness_accuracy_degradation(
    result: RobustnessBenchmarkResult,
    *,
    ax: Axes | None = None,
    show_values: bool = True,
) -> Axes:
    """Plot accuracy change relative to IID."""

    return plot_robustness_degradation_metric(
        result,
        "accuracy",
        ax=ax,
        show_values=show_values,
    )


def plot_robustness_coverage_degradation(
    result: RobustnessBenchmarkResult,
    *,
    ax: Axes | None = None,
    show_values: bool = True,
) -> Axes:
    """Plot coverage change relative to IID."""

    return plot_robustness_degradation_metric(
        result,
        "coverage",
        ax=ax,
        show_values=show_values,
    )