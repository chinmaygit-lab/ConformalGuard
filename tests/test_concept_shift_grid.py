import pytest

from conformalguard.experiments.concept_shift import (
    ConceptShiftExperimentResult,
)
from conformalguard.experiments.concept_shift_grid import (
    run_concept_shift_grid,
    summarize_concept_shift_grid,
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
    conformity_score="lac",
    shifted_feature="x0",
):
    return ConceptShiftExperimentResult(
        n_train=600,
        n_conf=200,
        n_test=200,
        confidence_level=confidence_level,
        conformity_score=conformity_score,
        random_state=seed,
        severity=severity,
        shifted_feature=shifted_feature,
        threshold=0.0,
        n_shifted=int(round(100 * severity)),
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


def test_concept_shift_grid_runs_every_seed(monkeypatch):
    calls = []

    def fake_sweep(
        X,
        y,
        *,
        severities,
        feature,
        confidence_level,
        conformity_score,
        random_state,
    ):
        calls.append(
            (
                random_state,
                tuple(severities),
                feature,
            )
        )

        return tuple(
            (random_state, severity)
            for severity in severities
        )

    monkeypatch.setattr(
        "conformalguard.experiments.concept_shift_grid."
        "run_concept_shift_sweep",
        fake_sweep,
    )

    results = run_concept_shift_grid(
        X="features",
        y="target",
        severities=(0.0, 0.50),
        seeds=(11, 42, 73),
        feature="x2",
    )

    assert len(results) == 6

    assert calls == [
        (11, (0.0, 0.50), "x2"),
        (42, (0.0, 0.50), "x2"),
        (73, (0.0, 0.50), "x2"),
    ]


def test_concept_shift_grid_rejects_empty_seeds():
    with pytest.raises(ValueError, match="random seed"):
        run_concept_shift_grid(
            X="features",
            y="target",
            seeds=(),
        )


def test_concept_shift_grid_rejects_empty_severities():
    with pytest.raises(ValueError, match="severity"):
        run_concept_shift_grid(
            X="features",
            y="target",
            severities=(),
        )


def test_summarize_concept_shift_grid_groups_by_severity():
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
            severity=0.50,
            seed=11,
            accuracy=0.60,
            coverage=0.84,
        ),
        make_result(
            severity=0.50,
            seed=42,
            accuracy=0.64,
            coverage=0.86,
        ),
    )

    summaries = summarize_concept_shift_grid(
        results
    )

    assert len(summaries) == 2

    iid, shifted = summaries

    assert iid.severity == pytest.approx(0.0)
    assert iid.n_runs == 2
    assert iid.shifted_feature == "x0"
    assert iid.mean_accuracy == pytest.approx(0.81)
    assert iid.mean_coverage == pytest.approx(0.90)

    assert shifted.severity == pytest.approx(0.50)
    assert shifted.n_runs == 2
    assert shifted.shifted_feature == "x0"
    assert shifted.mean_accuracy == pytest.approx(0.62)
    assert shifted.mean_coverage == pytest.approx(0.85)


def test_summarize_concept_shift_grid_rejects_empty_results():
    with pytest.raises(ValueError, match="result"):
        summarize_concept_shift_grid(())


def test_summarize_rejects_mixed_confidence_levels():
    results = (
        make_result(
            severity=0.50,
            seed=11,
            accuracy=0.80,
            coverage=0.89,
            confidence_level=0.90,
        ),
        make_result(
            severity=0.50,
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
        summarize_concept_shift_grid(results)


def test_summarize_rejects_mixed_conformity_scores():
    results = (
        make_result(
            severity=0.50,
            seed=11,
            accuracy=0.80,
            coverage=0.89,
            conformity_score="lac",
        ),
        make_result(
            severity=0.50,
            seed=42,
            accuracy=0.80,
            coverage=0.89,
            conformity_score="aps",
        ),
    )

    with pytest.raises(
        ValueError,
        match="conformity score",
    ):
        summarize_concept_shift_grid(results)


def test_summarize_rejects_mixed_shifted_features():
    results = (
        make_result(
            severity=0.50,
            seed=11,
            accuracy=0.80,
            coverage=0.89,
            shifted_feature="x0",
        ),
        make_result(
            severity=0.50,
            seed=42,
            accuracy=0.80,
            coverage=0.89,
            shifted_feature="x2",
        ),
    )

    with pytest.raises(
        ValueError,
        match="shifted feature",
    ):
        summarize_concept_shift_grid(results)
