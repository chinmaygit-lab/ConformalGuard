import pytest

from conformalguard.experiments.covariate_shift import (
    CovariateShiftExperimentResult,
)
from conformalguard.experiments.covariate_shift_grid import (
    run_covariate_shift_grid,
    summarize_covariate_shift_grid,
)
from conformalguard.metrics import (
    ClassificationMetrics,
    ConformalMetrics,
)


def make_result(
    *,
    severity,
    seed,
    accuracy,
    coverage,
    confidence_level=0.90,
):
    return CovariateShiftExperimentResult(
        n_train=600,
        n_conf=200,
        n_test=200,
        confidence_level=confidence_level,
        conformity_score="lac",
        random_state=seed,
        severity=severity,
        feature_fraction=0.50,
        shifted_columns=("a", "b"),
        classification=ClassificationMetrics(
            accuracy=accuracy,
            macro_f1=accuracy - 0.05,
        ),
        conformal=ConformalMetrics(
            coverage=coverage,
            coverage_gap=abs(
                confidence_level - coverage
            ),
            average_set_size=1.20 + severity,
            empty_set_rate=0.0,
        ),
    )


def test_covariate_shift_grid_runs_every_seed(monkeypatch):
    calls = []

    def fake_sweep(
        X,
        y,
        *,
        severities,
        confidence_level,
        feature_fraction,
        conformity_score,
        random_state,
    ):
        calls.append((random_state, tuple(severities)))
        return tuple(
            (random_state, severity)
            for severity in severities
        )

    monkeypatch.setattr(
        "conformalguard.experiments.covariate_shift_grid."
        "run_covariate_shift_sweep",
        fake_sweep,
    )

    results = run_covariate_shift_grid(
        X="features",
        y="target",
        severities=(0.0, 1.0),
        seeds=(11, 42, 73),
    )

    assert len(results) == 6
    assert calls == [
        (11, (0.0, 1.0)),
        (42, (0.0, 1.0)),
        (73, (0.0, 1.0)),
    ]


def test_covariate_shift_grid_rejects_empty_seeds():
    with pytest.raises(ValueError, match="random seed"):
        run_covariate_shift_grid(
            X="features",
            y="target",
            seeds=(),
        )


def test_summarize_covariate_shift_grid_groups_by_severity():
    results = (
        make_result(
            severity=0.0,
            seed=11,
            accuracy=0.80,
            coverage=0.89,
        ),
        make_result(
            severity=0.0,
            seed=42,
            accuracy=0.82,
            coverage=0.91,
        ),
        make_result(
            severity=1.0,
            seed=11,
            accuracy=0.60,
            coverage=0.84,
        ),
        make_result(
            severity=1.0,
            seed=42,
            accuracy=0.64,
            coverage=0.86,
        ),
    )

    summaries = summarize_covariate_shift_grid(
        results
    )

    assert len(summaries) == 2

    iid, shifted = summaries

    assert iid.severity == pytest.approx(0.0)
    assert iid.n_runs == 2
    assert iid.mean_accuracy == pytest.approx(0.81)
    assert iid.mean_coverage == pytest.approx(0.90)

    assert shifted.severity == pytest.approx(1.0)
    assert shifted.n_runs == 2
    assert shifted.mean_accuracy == pytest.approx(0.62)
    assert shifted.mean_coverage == pytest.approx(0.85)


def test_summarize_covariate_shift_grid_rejects_empty_results():
    with pytest.raises(ValueError, match="result"):
        summarize_covariate_shift_grid(())


def test_summarize_rejects_mixed_confidence_levels():
    results = (
        make_result(
            severity=0.0,
            seed=11,
            accuracy=0.80,
            coverage=0.89,
            confidence_level=0.90,
        ),
        make_result(
            severity=0.0,
            seed=42,
            accuracy=0.80,
            coverage=0.94,
            confidence_level=0.95,
        ),
    )

    with pytest.raises(
        ValueError,
        match="confidence level",
    ):
        summarize_covariate_shift_grid(results)