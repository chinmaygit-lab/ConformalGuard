import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes

from conformalguard.experiments import (
    CovariateShiftGridSummary,
    IIDGridSummary,
    LabelShiftGridSummary,
    RobustnessBenchmarkResult,
    plot_robustness_accuracy,
    plot_robustness_coverage,
    plot_robustness_coverage_gap,
    plot_robustness_metric,
    plot_robustness_set_size,
)


def _metric_kwargs(
    *,
    accuracy,
    coverage,
):
    return {
        "n_runs": 3,
        "mean_accuracy": accuracy,
        "std_accuracy": 0.02,
        "mean_macro_f1": accuracy - 0.01,
        "std_macro_f1": 0.03,
        "mean_coverage": coverage,
        "std_coverage": 0.01,
        "mean_coverage_gap": 0.90 - coverage,
        "std_coverage_gap": 0.01,
        "mean_set_size": 1.25,
        "std_set_size": 0.10,
        "mean_empty_set_rate": 0.01,
        "std_empty_set_rate": 0.005,
    }


def _benchmark_result():
    iid = IIDGridSummary(
        confidence_level=0.90,
        **_metric_kwargs(
            accuracy=0.91,
            coverage=0.89,
        ),
    )

    covariate = CovariateShiftGridSummary(
        severity=1.0,
        confidence_level=0.90,
        feature_fraction=0.50,
        conformity_score="lac",
        **_metric_kwargs(
            accuracy=0.84,
            coverage=0.82,
        ),
    )

    label = LabelShiftGridSummary(
        target_proportions={0: 0.80, 1: 0.20},
        confidence_level=0.90,
        conformity_score="lac",
        **_metric_kwargs(
            accuracy=0.86,
            coverage=0.85,
        ),
    )

    return RobustnessBenchmarkResult(
        confidence_level=0.90,
        conformity_score="lac",
        seeds=(11, 42, 73),
        iid_results=(),
        iid_summary=iid,
        covariate_shift_results=(),
        covariate_shift_summary=(covariate,),
        label_shift_results=(),
        label_shift_summary=(label,),
    )


def test_plot_robustness_metric_returns_axes():
    ax = plot_robustness_metric(
        _benchmark_result(),
        "coverage",
    )

    assert isinstance(ax, Axes)
    assert ax.get_ylabel() == "Coverage"
    assert ax.get_xlabel() == "Experiment condition"
    assert len(ax.patches) == 3

    labels = [
        tick.get_text()
        for tick in ax.get_xticklabels()
    ]

    assert labels == [
        "IID",
        "Covariate\nseverity=1",
        "Label\ntarget=0:0.8, 1:0.2",
    ]

    plt.close(ax.figure)


def test_plot_robustness_metric_uses_supplied_axes():
    figure, supplied_ax = plt.subplots()

    returned_ax = plot_robustness_metric(
        _benchmark_result(),
        "coverage",
        ax=supplied_ax,
        show_std=False,
    )

    assert returned_ax is supplied_ax
    assert len(returned_ax.patches) == 3

    plt.close(figure)


def test_plot_robustness_metric_rejects_unknown_metric():
    with pytest.raises(
        ValueError,
        match="Unsupported robustness metric",
    ):
        plot_robustness_metric(
            _benchmark_result(),
            "not-a-metric",
        )


def test_plot_robustness_accuracy_wrapper():
    ax = plot_robustness_accuracy(
        _benchmark_result(),
        show_std=False,
    )

    assert ax.get_ylabel() == "Accuracy"
    assert "Accuracy" in ax.get_title()

    plt.close(ax.figure)


def test_plot_robustness_coverage_gap_wrapper():
    ax = plot_robustness_coverage_gap(
        _benchmark_result(),
        show_std=False,
    )

    assert ax.get_ylabel() == "Coverage gap"
    assert "Coverage gap" in ax.get_title()

    plt.close(ax.figure)


def test_plot_robustness_set_size_wrapper():
    ax = plot_robustness_set_size(
        _benchmark_result(),
        show_std=False,
    )

    assert (
        ax.get_ylabel()
        == "Average prediction-set size"
    )

    plt.close(ax.figure)


def test_plot_robustness_coverage_wrapper():
    ax = plot_robustness_coverage(
        _benchmark_result(),
        show_std=False,
    )

    assert ax.get_ylabel() == "Coverage"

    plt.close(ax.figure)
