import pytest

from conformalguard.experiments.label_shift import (
    LabelShiftExperimentResult,
)
from conformalguard.experiments.label_shift_grid import (
    run_label_shift_grid,
    summarize_label_shift_grid,
)
from conformalguard.metrics import (
    ClassificationMetrics,
    ConformalMetrics,
)


def make_result(
    *,
    target_proportions,
    seed,
    accuracy,
    coverage,
    confidence_level=0.90,
):
    return LabelShiftExperimentResult(
        n_train=600,
        n_conf=200,
        n_test=200,
        confidence_level=confidence_level,
        conformity_score="lac",
        random_state=seed,
        target_proportions=dict(target_proportions),
        sampled_counts={
            label: int(round(200 * proportion))
            for label, proportion in target_proportions.items()
        },
        classification=ClassificationMetrics(
            accuracy=accuracy,
            macro_f1=accuracy - 0.05,
        ),
        conformal=ConformalMetrics(
            coverage=coverage,
            coverage_gap=abs(
                confidence_level - coverage
            ),
            average_set_size=1.20,
            empty_set_rate=0.0,
        ),
    )


def test_label_shift_grid_runs_every_seed(monkeypatch):
    calls = []

    def fake_sweep(
        X,
        y,
        *,
        target_proportions,
        confidence_level,
        conformity_score,
        random_state,
    ):
        targets = tuple(
            dict(target)
            for target in target_proportions
        )
        calls.append((random_state, targets))
        return tuple(
            (random_state, target)
            for target in targets
        )

    monkeypatch.setattr(
        "conformalguard.experiments.label_shift_grid."
        "run_label_shift_sweep",
        fake_sweep,
    )

    targets = (
        {0: 0.50, 1: 0.50},
        {0: 0.80, 1: 0.20},
    )

    results = run_label_shift_grid(
        X="features",
        y="target",
        target_proportions=targets,
        seeds=(11, 42, 73),
    )

    assert len(results) == 6
    assert calls == [
        (11, targets),
        (42, targets),
        (73, targets),
    ]


def test_label_shift_grid_rejects_empty_seeds():
    with pytest.raises(ValueError, match="random seed"):
        run_label_shift_grid(
            X="features",
            y="target",
            target_proportions=({0: 0.50, 1: 0.50},),
            seeds=(),
        )


def test_label_shift_grid_rejects_empty_targets():
    with pytest.raises(ValueError, match="target proportion"):
        run_label_shift_grid(
            X="features",
            y="target",
            target_proportions=(),
        )


def test_summarize_label_shift_grid_groups_by_target_distribution():
    balanced = {0: 0.50, 1: 0.50}
    shifted = {0: 0.80, 1: 0.20}

    results = (
        make_result(
            target_proportions=balanced,
            seed=11,
            accuracy=0.80,
            coverage=0.89,
        ),
        make_result(
            target_proportions=balanced,
            seed=42,
            accuracy=0.82,
            coverage=0.91,
        ),
        make_result(
            target_proportions=shifted,
            seed=11,
            accuracy=0.60,
            coverage=0.84,
        ),
        make_result(
            target_proportions=shifted,
            seed=42,
            accuracy=0.64,
            coverage=0.86,
        ),
    )

    summaries = summarize_label_shift_grid(results)

    assert len(summaries) == 2

    iid, shifted_summary = summaries

    assert iid.target_proportions == balanced
    assert iid.n_runs == 2
    assert iid.mean_accuracy == pytest.approx(0.81)
    assert iid.mean_coverage == pytest.approx(0.90)

    assert shifted_summary.target_proportions == shifted
    assert shifted_summary.n_runs == 2
    assert shifted_summary.mean_accuracy == pytest.approx(0.62)
    assert shifted_summary.mean_coverage == pytest.approx(0.85)


def test_summarize_label_shift_grid_rejects_empty_results():
    with pytest.raises(ValueError, match="result"):
        summarize_label_shift_grid(())


def test_summarize_rejects_mixed_confidence_levels():
    target = {0: 0.50, 1: 0.50}

    results = (
        make_result(
            target_proportions=target,
            seed=11,
            accuracy=0.80,
            coverage=0.89,
            confidence_level=0.90,
        ),
        make_result(
            target_proportions=target,
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
        summarize_label_shift_grid(results)