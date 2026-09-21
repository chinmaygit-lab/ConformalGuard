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
