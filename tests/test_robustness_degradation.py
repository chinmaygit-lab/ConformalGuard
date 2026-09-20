from types import SimpleNamespace

import pytest

from conformalguard.experiments import (
    robustness_degradation_frame,
    write_robustness_degradation_csv,
    write_robustness_degradation_json,
)


def metrics(
    *,
    accuracy,
    macro_f1,
    coverage,
    coverage_gap,
    set_size,
    empty_set_rate,
):
    return {
        "n_runs": 3,
        "confidence_level": 0.90,
        "conformity_score": "lac",
        "mean_accuracy": accuracy,
        "std_accuracy": 0.02,
        "mean_macro_f1": macro_f1,
        "std_macro_f1": 0.03,
        "mean_coverage": coverage,
        "std_coverage": 0.01,
        "mean_coverage_gap": coverage_gap,
        "std_coverage_gap": 0.01,
        "mean_set_size": set_size,
        "std_set_size": 0.10,
        "mean_empty_set_rate": empty_set_rate,
        "std_empty_set_rate": 0.005,
    }


def benchmark_result():
    iid = SimpleNamespace(
        **metrics(
            accuracy=0.91,
            macro_f1=0.90,
            coverage=0.89,
            coverage_gap=0.01,
            set_size=1.20,
            empty_set_rate=0.02,
        )
    )

    covariate = SimpleNamespace(
        severity=1.0,
        feature_fraction=0.50,
        **metrics(
            accuracy=0.84,
            macro_f1=0.82,
            coverage=0.82,
            coverage_gap=0.08,
            set_size=1.35,
            empty_set_rate=0.04,
        )
    )

    label = SimpleNamespace(
        target_proportions={0: 0.80, 1: 0.20},
        **metrics(
            accuracy=0.86,
            macro_f1=0.85,
            coverage=0.85,
            coverage_gap=0.05,
            set_size=1.25,
            empty_set_rate=0.03,
        )
    )

    concept = SimpleNamespace(
        severity=0.50,
        shifted_feature="x0",
        **metrics(
            accuracy=0.73,
            macro_f1=0.70,
            coverage=0.66,
            coverage_gap=0.24,
            set_size=1.40,
            empty_set_rate=0.06,
        )
    )

    return SimpleNamespace(
        conformity_score="lac",
        iid_summary=iid,
        covariate_shift_summary=(covariate,),
        label_shift_summary=(label,),
        concept_shift_summary=(concept,),
    )


def test_degradation_frame_contains_only_shifted_conditions():
    frame = robustness_degradation_frame(
        benchmark_result()
    )

    assert list(frame["experiment"]) == [
        "covariate_shift",
        "label_shift",
        "concept_shift",
    ]

    assert "iid" not in set(frame["experiment"])


def test_degradation_is_measured_relative_to_iid():
    frame = robustness_degradation_frame(
        benchmark_result()
    )

    covariate = frame.iloc[0]

    assert covariate["iid_mean_accuracy"] == pytest.approx(
        0.91
    )
    assert covariate["mean_accuracy"] == pytest.approx(
        0.84
    )
    assert covariate["accuracy_delta"] == pytest.approx(
        -0.07
    )

    assert covariate["iid_mean_coverage"] == pytest.approx(
        0.89
    )
    assert covariate["mean_coverage"] == pytest.approx(
        0.82
    )
    assert covariate["coverage_delta"] == pytest.approx(
        -0.07
    )


def test_degradation_computes_all_supported_metric_deltas():
    frame = robustness_degradation_frame(
        benchmark_result()
    )

    concept = frame.iloc[2]

    assert concept["accuracy_delta"] == pytest.approx(-0.18)
    assert concept["macro_f1_delta"] == pytest.approx(-0.20)
    assert concept["coverage_delta"] == pytest.approx(-0.23)

    assert concept["coverage_gap_delta"] == pytest.approx(
        0.23
    )

    assert concept["set_size_delta"] == pytest.approx(
        0.20
    )

    assert concept["empty_set_rate_delta"] == pytest.approx(
        0.04
    )


def test_degradation_preserves_shift_condition_metadata():
    frame = robustness_degradation_frame(
        benchmark_result()
    )

    covariate = frame.iloc[0]
    label = frame.iloc[1]
    concept = frame.iloc[2]

    assert covariate["severity"] == pytest.approx(1.0)
    assert covariate["feature_fraction"] == pytest.approx(
        0.50
    )

    assert label["target_proportions"] == "0:0.8, 1:0.2"

    assert concept["severity"] == pytest.approx(0.50)
    assert concept["condition"] == (
        "severity=0.5; feature=x0"
    )


def test_write_degradation_csv(tmp_path):
    output = write_robustness_degradation_csv(
        benchmark_result(),
        tmp_path / "reports" / "degradation.csv",
    )

    assert output.exists()

    content = output.read_text(encoding="utf-8")

    assert "accuracy_delta" in content
    assert "coverage_delta" in content
    assert "concept_shift" in content


def test_write_degradation_json(tmp_path):
    output = write_robustness_degradation_json(
        benchmark_result(),
        tmp_path / "reports" / "degradation.json",
    )

    assert output.exists()

    content = output.read_text(encoding="utf-8")

    assert '"accuracy_delta"' in content
    assert '"coverage_delta"' in content
    assert '"concept_shift"' in content
