import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes

from conformalguard.experiments import (
    ConceptShiftGridSummary,
    CovariateShiftGridSummary,
    IIDGridSummary,
    LabelShiftGridSummary,
    RobustnessBenchmarkResult,
    plot_robustness_accuracy_degradation,
    plot_robustness_coverage_degradation,
    plot_robustness_degradation_metric,
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

    concept = ConceptShiftGridSummary(
        severity=0.50,
        confidence_level=0.90,
        conformity_score="lac",
        shifted_feature="x0",
        **_metric_kwargs(
            accuracy=0.73,
            coverage=0.66,
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
        concept_shift_results=(),
        concept_shift_summary=(concept,),
    )


def test_plot_degradation_metric_returns_axes():
    ax = plot_robustness_degradation_metric(
        _benchmark_result(),
        "coverage",
    )

    assert isinstance(ax, Axes)

    assert ax.get_ylabel() == "Coverage delta vs IID"
    assert ax.get_xlabel() == "Shift condition"

    assert "Coverage" in ax.get_title()
    assert "IID" in ax.get_title()

    assert len(ax.patches) == 3

    heights = [
        patch.get_height()
        for patch in ax.patches
    ]

    assert heights == pytest.approx(
        [-0.07, -0.04, -0.23]
    )

    labels = [
        tick.get_text()
        for tick in ax.get_xticklabels()
    ]

    assert labels == [
        "Covariate\nseverity=1",
        "Label\ntarget=0:0.8, 1:0.2",
        "Concept\nseverity=0.5; feature=x0",
    ]

    plt.close(ax.figure)


def test_plot_degradation_metric_draws_zero_reference():
    ax = plot_robustness_degradation_metric(
        _benchmark_result(),
        "accuracy",
        show_values=False,
    )

    zero_lines = [
        line
        for line in ax.lines
        if all(
            float(value) == 0.0
            for value in line.get_ydata()
        )
    ]

    assert zero_lines

    plt.close(ax.figure)


def test_plot_degradation_metric_uses_supplied_axes():
    figure, supplied_ax = plt.subplots()

    returned_ax = plot_robustness_degradation_metric(
        _benchmark_result(),
        "coverage",
        ax=supplied_ax,
        show_values=False,
    )

    assert returned_ax is supplied_ax
    assert len(returned_ax.patches) == 3

    plt.close(figure)


def test_plot_degradation_metric_rejects_unknown_metric():
    with pytest.raises(
        ValueError,
        match="Unsupported degradation metric",
    ):
        plot_robustness_degradation_metric(
            _benchmark_result(),
            "not-a-metric",
        )


def test_accuracy_degradation_wrapper():
    ax = plot_robustness_accuracy_degradation(
        _benchmark_result(),
        show_values=False,
    )

    assert ax.get_ylabel() == "Accuracy delta vs IID"

    plt.close(ax.figure)


def test_coverage_degradation_wrapper():
    ax = plot_robustness_coverage_degradation(
        _benchmark_result(),
        show_values=False,
    )

    assert ax.get_ylabel() == "Coverage delta vs IID"

    plt.close(ax.figure)