import pandas as pd
import pytest

from conformalguard.guard import (
    ConformalGuard,
    GuardConfig,
    _default_label_targets,
    _validate_feature_frame,
)


def test_default_binary_label_targets_are_deterministic():
    targets = _default_label_targets(pd.Series([0, 1, 0, 1]))
    assert targets == (
        {0: 0.5, 1: 0.5},
        {0: 0.7, 1: 0.3},
        {0: 0.3, 1: 0.7},
    )


def test_multiclass_label_targets_sum_to_one():
    targets = _default_label_targets(pd.Series(["a", "b", "c", "a"]))
    assert len(targets) == 3
    for target in targets:
        assert sum(target.values()) == pytest.approx(1.0)


def test_config_rejects_invalid_confidence():
    with pytest.raises(ValueError, match="confidence_level"):
        GuardConfig(confidence_level=1.0)


def test_config_rejects_empty_seeds():
    with pytest.raises(ValueError, match="random seed"):
        GuardConfig(seeds=())


def test_non_numeric_features_fail_with_clear_message():
    frame = pd.DataFrame({"age": [1, 2], "city": ["a", "b"]})
    with pytest.raises(ValueError, match="Non-numeric"):
        _validate_feature_frame(frame)


def test_summary_requires_a_run_first():
    with pytest.raises(RuntimeError, match="guard.run"):
        ConformalGuard().summary()


def test_config_rejects_negative_covariate_severity():
    with pytest.raises(ValueError, match="covariate_severities"):
        GuardConfig(covariate_severities=(0.0, -0.5))


def test_config_rejects_concept_severity_above_one():
    with pytest.raises(ValueError, match="concept_severities"):
        GuardConfig(concept_severities=(0.0, 1.25))

def test_worst_shift_reports_largest_coverage_and_accuracy_drop(
    monkeypatch,
):
    guard = ConformalGuard()

    degradation = pd.DataFrame(
        [
            {
                "experiment": "covariate_shift",
                "condition": "severity=0.5",
                "coverage_delta": -0.10,
                "accuracy_delta": -0.05,
            },
            {
                "experiment": "covariate_shift",
                "condition": "severity=2",
                "coverage_delta": -0.42,
                "accuracy_delta": -0.31,
            },
            {
                "experiment": "concept_shift",
                "condition": "severity=1; feature=0",
                "coverage_delta": -0.35,
                "accuracy_delta": -0.44,
            },
        ]
    )

    monkeypatch.setattr(
        guard,
        "degradation",
        lambda: degradation,
    )

    result = guard.worst_shift()

    assert result["worst_coverage_experiment"] == "covariate_shift"
    assert result["worst_coverage_condition"] == "severity=2"
    assert result["worst_coverage_delta"] == pytest.approx(-0.42)

    assert result["worst_accuracy_experiment"] == "concept_shift"
    assert (
        result["worst_accuracy_condition"]
        == "severity=1; feature=0"
    )
    assert result["worst_accuracy_delta"] == pytest.approx(-0.44)


def test_worst_shift_rejects_empty_degradation_frame(
    monkeypatch,
):
    guard = ConformalGuard()

    monkeypatch.setattr(
        guard,
        "degradation",
        lambda: pd.DataFrame(),
    )

    with pytest.raises(
        RuntimeError,
        match="No shifted conditions",
    ):
        guard.worst_shift()
