import json

import pandas as pd

from conformalguard.experiments import (
    CovariateShiftGridSummary,
    IIDGridSummary,
    LabelShiftGridSummary,
    RobustnessBenchmarkResult,
    robustness_summary_frame,
    robustness_summary_records,
    write_robustness_summary_csv,
    write_robustness_summary_json,
)


def _metric_kwargs(
    *,
    accuracy,
    coverage,
):
    return {
        "n_runs": 3,
        "mean_accuracy": accuracy,
        "std_accuracy": 0.02,
        "mean_macro_f1": accuracy - 0.01,
        "std_macro_f1": 0.03,
        "mean_coverage": coverage,
        "std_coverage": 0.01,
        "mean_coverage_gap": 0.90 - coverage,
        "std_coverage_gap": 0.01,
        "mean_set_size": 1.25,
        "std_set_size": 0.10,
        "mean_empty_set_rate": 0.01,
        "std_empty_set_rate": 0.005,
    }


def _benchmark_result():
    iid = IIDGridSummary(
        confidence_level=0.90,
        **_metric_kwargs(
            accuracy=0.91,
            coverage=0.89,
        ),
    )

    covariate = CovariateShiftGridSummary(
        severity=1.0,
        confidence_level=0.90,
        feature_fraction=0.50,
        conformity_score="lac",
        **_metric_kwargs(
            accuracy=0.84,
            coverage=0.82,
        ),
    )

    label = LabelShiftGridSummary(
        target_proportions={1: 0.20, 0: 0.80},
        confidence_level=0.90,
        conformity_score="lac",
        **_metric_kwargs(
            accuracy=0.86,
            coverage=0.85,
        ),
    )

    return RobustnessBenchmarkResult(
        confidence_level=0.90,
        conformity_score="lac",
        seeds=(11, 42, 73),
        iid_results=(),
        iid_summary=iid,
        covariate_shift_results=(),
        covariate_shift_summary=(covariate,),
        label_shift_results=(),
        label_shift_summary=(label,),
    )


def test_robustness_summary_records_normalize_experiments():
    records = robustness_summary_records(
        _benchmark_result()
    )

    assert len(records) == 3

    iid, covariate, label = records

    assert iid["experiment"] == "iid"
    assert iid["condition"] == "iid"
    assert iid["severity"] is None
    assert iid["target_proportions"] is None

    assert covariate["experiment"] == "covariate_shift"
    assert covariate["condition"] == "severity=1"
    assert covariate["severity"] == 1.0
    assert covariate["feature_fraction"] == 0.50

    assert label["experiment"] == "label_shift"
    assert label["condition"] == "target=0:0.8, 1:0.2"
    assert label["target_proportions"] == "0:0.8, 1:0.2"


def test_target_distribution_text_is_deterministic():
    first = _benchmark_result()

    reordered_label = LabelShiftGridSummary(
        target_proportions={0: 0.80, 1: 0.20},
        confidence_level=0.90,
        conformity_score="lac",
        **_metric_kwargs(
            accuracy=0.86,
            coverage=0.85,
        ),
    )

    second = RobustnessBenchmarkResult(
        confidence_level=first.confidence_level,
        conformity_score=first.conformity_score,
        seeds=first.seeds,
        iid_results=first.iid_results,
        iid_summary=first.iid_summary,
        covariate_shift_results=first.covariate_shift_results,
        covariate_shift_summary=first.covariate_shift_summary,
        label_shift_results=(),
        label_shift_summary=(reordered_label,),
    )

    first_text = robustness_summary_records(first)[2][
        "target_proportions"
    ]
    second_text = robustness_summary_records(second)[2][
        "target_proportions"
    ]

    assert first_text == second_text
    assert first_text == "0:0.8, 1:0.2"


def test_robustness_summary_frame_has_tidy_rows():
    frame = robustness_summary_frame(
        _benchmark_result()
    )

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 3

    assert list(frame["experiment"]) == [
        "iid",
        "covariate_shift",
        "label_shift",
    ]

    assert frame.loc[0, "mean_coverage"] == 0.89
    assert frame.loc[1, "mean_coverage"] == 0.82
    assert frame.loc[2, "mean_coverage"] == 0.85


def test_write_robustness_summary_csv(tmp_path):
    output = write_robustness_summary_csv(
        _benchmark_result(),
        tmp_path / "nested" / "summary.csv",
    )

    assert output.exists()

    frame = pd.read_csv(output)

    assert len(frame) == 3
    assert list(frame["experiment"]) == [
        "iid",
        "covariate_shift",
        "label_shift",
    ]


def test_write_robustness_summary_json(tmp_path):
    output = write_robustness_summary_json(
        _benchmark_result(),
        tmp_path / "nested" / "summary.json",
    )

    assert output.exists()

    payload = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert len(payload) == 3
    assert payload[0]["experiment"] == "iid"
    assert payload[1]["condition"] == "severity=1"
    assert (
        payload[2]["target_proportions"]
        == "0:0.8, 1:0.2"
    )
