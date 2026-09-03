import pytest

from conformalguard.experiments.iid_conformal import IIDConformalResult
from conformalguard.experiments.iid_grid import (
    run_iid_grid,
    summarize_iid_grid,
)
from conformalguard.metrics import ClassificationMetrics, ConformalMetrics


def make_result(
    confidence_level,
    coverage,
    coverage_gap,
    set_size,
    empty_rate,
    accuracy=0.80,
    macro_f1=0.75,
):
    return IIDConformalResult(
        n_train=600,
        n_conf=200,
        n_test=200,
        confidence_level=confidence_level,
        conformity_score="lac",
        classification=ClassificationMetrics(
            accuracy=accuracy,
            macro_f1=macro_f1,
        ),
        conformal=ConformalMetrics(
            coverage=coverage,
            coverage_gap=coverage_gap,
            average_set_size=set_size,
            empty_set_rate=empty_rate,
        ),
    )


def test_run_iid_grid_builds_seed_coverage_cartesian_product(monkeypatch):
    calls = []

    def fake_run(
        X,
        y,
        *,
        confidence_level,
        conformity_score,
        random_state,
    ):
        calls.append(
            (confidence_level, conformity_score, random_state)
        )
        return (confidence_level, random_state)

    monkeypatch.setattr(
        "conformalguard.experiments.iid_grid.run_iid_conformal",
        fake_run,
    )

    results = run_iid_grid(
        X="features",
        y="target",
        confidence_levels=(0.80, 0.90),
        seeds=(11, 42),
        conformity_score="lac",
    )

    assert len(results) == 4
    assert calls == [
        (0.80, "lac", 11),
        (0.90, "lac", 11),
        (0.80, "lac", 42),
        (0.90, "lac", 42),
    ]


def test_run_iid_grid_rejects_empty_confidence_levels():
    with pytest.raises(ValueError, match="confidence level"):
        run_iid_grid(
            X="features",
            y="target",
            confidence_levels=(),
        )


def test_run_iid_grid_rejects_empty_seed_list():
    with pytest.raises(ValueError, match="random seed"):
        run_iid_grid(
            X="features",
            y="target",
            seeds=(),
        )


def test_summarize_iid_grid_aggregates_by_confidence_level():
    results = (
        make_result(0.80, 0.79, 0.01, 1.00, 0.00, accuracy=0.78),
        make_result(0.80, 0.81, 0.01, 1.20, 0.00, accuracy=0.82),
        make_result(0.90, 0.89, 0.01, 1.30, 0.00, accuracy=0.78),
        make_result(0.90, 0.91, 0.01, 1.50, 0.02, accuracy=0.82),
    )

    summaries = summarize_iid_grid(results)

    assert len(summaries) == 2

    low, high = summaries

    assert low.confidence_level == pytest.approx(0.80)
    assert low.n_runs == 2
    assert low.mean_accuracy == pytest.approx(0.80)
    assert low.mean_coverage == pytest.approx(0.80)
    assert low.mean_coverage_gap == pytest.approx(0.01)
    assert low.mean_set_size == pytest.approx(1.10)
    assert low.mean_empty_set_rate == pytest.approx(0.00)

    assert high.confidence_level == pytest.approx(0.90)
    assert high.n_runs == 2
    assert high.mean_accuracy == pytest.approx(0.80)
    assert high.mean_coverage == pytest.approx(0.90)
    assert high.mean_coverage_gap == pytest.approx(0.01)
    assert high.mean_set_size == pytest.approx(1.40)
    assert high.mean_empty_set_rate == pytest.approx(0.01)


def test_summarize_iid_grid_rejects_empty_results():
    with pytest.raises(ValueError, match="result"):
        summarize_iid_grid(())