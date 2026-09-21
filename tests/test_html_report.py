from pathlib import Path

import pandas as pd

from conformalguard.html_report import write_html_report


def test_html_report_contains_core_metrics_and_plot_references(tmp_path: Path):
    summary = pd.DataFrame(
        [
            {"experiment": "iid", "condition": "iid", "mean_coverage": 0.90},
            {
                "experiment": "covariate_shift",
                "condition": "severity=1",
                "mean_coverage": 0.61,
            },
        ]
    )
    degradation = pd.DataFrame(
        [{"experiment": "covariate_shift", "coverage_delta": -0.29}]
    )

    output = write_html_report(summary, degradation, tmp_path / "report.html")
    text = output.read_text(encoding="utf-8")

    assert "IID coverage" in text
    assert "0.900" in text
    assert "-0.290" in text
    assert "robustness_coverage.png" in text
    assert "arbitrary real-world shift" in text
