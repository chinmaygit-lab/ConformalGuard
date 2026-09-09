import pytest

from conformalguard.experiments import (
    RobustnessBenchmarkResult,
    run_robustness_benchmark,
)


def test_robustness_benchmark_runs_all_three_grids(monkeypatch):
    calls = {}

    iid_results = ("iid-result",)
    covariate_results = ("covariate-result",)
    label_results = ("label-result",)

    def fake_iid_grid(
        X,
        y,
        *,
        confidence_levels,
        seeds,
        conformity_score,
    ):
        calls["iid"] = (
            X,
            y,
            tuple(confidence_levels),
            tuple(seeds),
            conformity_score,
        )
        return iid_results

    def fake_covariate_grid(
        X,
        y,
        *,
        severities,
        seeds,
        confidence_level,
        feature_fraction,
        conformity_score,
    ):
        calls["covariate"] = (
            tuple(severities),
            tuple(seeds),
            confidence_level,
            feature_fraction,
            conformity_score,
        )
        return covariate_results

    def fake_label_grid(
        X,
        y,
        *,
        target_proportions,
        seeds,
        confidence_level,
        conformity_score,
    ):
        calls["label"] = (
            tuple(target_proportions),
            tuple(seeds),
            confidence_level,
            conformity_score,
        )
        return label_results

    def fake_iid_summary(results):
        calls["iid_summary"] = results
        return ("iid-summary",)

    def fake_covariate_summary(results):
        calls["covariate_summary"] = results
        return ("covariate-summary",)

    def fake_label_summary(results):
        calls["label_summary"] = results
        return ("label-summary",)

    module = "conformalguard.experiments.robustness."

    monkeypatch.setattr(
        module + "run_iid_grid",
        fake_iid_grid,
    )
    monkeypatch.setattr(
        module + "run_covariate_shift_grid",
        fake_covariate_grid,
    )
    monkeypatch.setattr(
        module + "run_label_shift_grid",
        fake_label_grid,
    )
    monkeypatch.setattr(
        module + "summarize_iid_grid",
        fake_iid_summary,
    )
    monkeypatch.setattr(
        module + "summarize_covariate_shift_grid",
        fake_covariate_summary,
    )
    monkeypatch.setattr(
        module + "summarize_label_shift_grid",
        fake_label_summary,
    )

    targets = (
        {0: 0.50, 1: 0.50},
        {0: 0.80, 1: 0.20},
    )

    result = run_robustness_benchmark(
        X="features",
        y="target",
        label_target_proportions=targets,
        covariate_severities=(0.0, 1.0),
        seeds=(11, 42),
        confidence_level=0.90,
        feature_fraction=0.50,
        conformity_score="lac",
    )

    assert isinstance(result, RobustnessBenchmarkResult)
    assert result.confidence_level == pytest.approx(0.90)
    assert result.conformity_score == "lac"
    assert result.seeds == (11, 42)

    assert result.iid_results == iid_results
    assert result.iid_summary == "iid-summary"

    assert result.covariate_shift_results == covariate_results
    assert result.covariate_shift_summary == (
        "covariate-summary",
    )

    assert result.label_shift_results == label_results
    assert result.label_shift_summary == ("label-summary",)

    assert calls["iid"] == (
        "features",
        "target",
        (0.90,),
        (11, 42),
        "lac",
    )

    assert calls["covariate"] == (
        (0.0, 1.0),
        (11, 42),
        0.90,
        0.50,
        "lac",
    )

    assert calls["label"] == (
        targets,
        (11, 42),
        0.90,
        "lac",
    )

    assert calls["iid_summary"] == iid_results
    assert calls["covariate_summary"] == covariate_results
    assert calls["label_summary"] == label_results


def test_robustness_benchmark_rejects_empty_seeds():
    with pytest.raises(ValueError, match="random seed"):
        run_robustness_benchmark(
            X="features",
            y="target",
            label_target_proportions=(
                {0: 0.50, 1: 0.50},
            ),
            seeds=(),
        )


def test_robustness_benchmark_rejects_empty_covariate_severities():
    with pytest.raises(ValueError, match="severity"):
        run_robustness_benchmark(
            X="features",
            y="target",
            label_target_proportions=(
                {0: 0.50, 1: 0.50},
            ),
            covariate_severities=(),
        )


def test_robustness_benchmark_rejects_empty_label_targets():
    with pytest.raises(ValueError, match="target proportion"):
        run_robustness_benchmark(
            X="features",
            y="target",
            label_target_proportions=(),
        )
