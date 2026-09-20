from types import SimpleNamespace

from conformalguard.experiments.robustness_plotting import (
    _condition_label,
)
from conformalguard.experiments.robustness_reporting import (
    robustness_summary_records,
)


def make_metrics_summary(**extra):
    values = {
        "n_runs": 3,
        "confidence_level": 0.90,
        "conformity_score": "lac",
        "mean_accuracy": 0.80,
        "std_accuracy": 0.02,
        "mean_macro_f1": 0.79,
        "std_macro_f1": 0.03,
        "mean_coverage": 0.89,
        "std_coverage": 0.01,
        "mean_coverage_gap": -0.01,
        "std_coverage_gap": 0.01,
        "mean_set_size": 1.20,
        "std_set_size": 0.05,
        "mean_empty_set_rate": 0.01,
        "std_empty_set_rate": 0.005,
    }
    values.update(extra)
    return SimpleNamespace(**values)


def test_reporting_includes_concept_shift_summary():
    iid = make_metrics_summary()

    concept = make_metrics_summary(
        severity=0.50,
        shifted_feature="x0",
    )

    result = SimpleNamespace(
        conformity_score="lac",
        iid_summary=iid,
        covariate_shift_summary=(),
        label_shift_summary=(),
        concept_shift_summary=(concept,),
    )

    records = robustness_summary_records(result)

    assert len(records) == 2

    concept_record = records[1]

    assert concept_record["experiment"] == "concept_shift"
    assert concept_record["condition"] == (
        "severity=0.5; feature=x0"
    )
    assert concept_record["severity"] == 0.50
    assert concept_record["feature_fraction"] is None
    assert concept_record["target_proportions"] is None
    assert concept_record["mean_accuracy"] == 0.80
    assert concept_record["mean_coverage"] == 0.89


def test_plot_label_formats_concept_shift():
    label = _condition_label(
        "concept_shift",
        "severity=0.5; feature=x0",
    )

    assert label == (
        "Concept\nseverity=0.5; feature=x0"
    )
