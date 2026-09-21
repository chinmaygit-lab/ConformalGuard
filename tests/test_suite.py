import pandas as pd
import pytest

from conformalguard.suite import (
    available_datasets,
    load_builtin_dataset,
    suite_scorecard,
)


def test_available_datasets_are_network_free_builtins():
    assert available_datasets() == ("breast_cancer", "iris", "wine", "digits")


@pytest.mark.parametrize("name", ["breast_cancer", "iris", "wine", "digits"])
def test_builtin_dataset_has_aligned_numeric_data(name):
    dataset = load_builtin_dataset(name)
    assert dataset.name == name
    assert len(dataset.X) == len(dataset.y)
    assert len(dataset.X) > 0
    assert all(pd.api.types.is_numeric_dtype(dataset.X[column]) for column in dataset.X)


def test_unknown_dataset_has_clear_error():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_builtin_dataset("not-a-dataset")


def test_scorecard_extracts_worst_shift_values():
    summary = pd.DataFrame(
        [
            {
                "dataset": "tiny",
                "experiment": "iid",
                "mean_accuracy": 0.9,
                "mean_coverage": 0.95,
            },
            {
                "dataset": "tiny",
                "experiment": "covariate_shift",
                "mean_accuracy": 0.7,
                "mean_coverage": 0.8,
            },
        ]
    )
    degradation = pd.DataFrame(
        [
            {
                "dataset": "tiny",
                "experiment": "covariate_shift",
                "condition": "severity=2",
                "mean_accuracy": 0.7,
                "mean_coverage": 0.8,
                "accuracy_delta": -0.2,
                "coverage_delta": -0.15,
            }
        ]
    )
    card = suite_scorecard(summary, degradation)
    assert card.loc[0, "dataset"] == "tiny"
    assert card.loc[0, "worst_accuracy_delta"] == pytest.approx(-0.2)
    assert card.loc[0, "worst_coverage_delta"] == pytest.approx(-0.15)
    assert card.loc[0, "worst_accuracy_experiment"] == "covariate_shift"
    assert card.loc[0, "worst_accuracy_condition"] == "severity=2"
    assert card.loc[0, "worst_coverage_experiment"] == "covariate_shift"
    assert card.loc[0, "worst_coverage_condition"] == "severity=2"
