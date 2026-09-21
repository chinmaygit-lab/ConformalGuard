"""Dependency-light static HTML reporting for ConformalGuard."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


def write_html_report(
    summary: pd.DataFrame,
    degradation: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "ConformalGuard robustness report",
) -> Path:
    """Write a standalone report page referencing plots beside the HTML file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    iid = summary.loc[summary["experiment"] == "iid"] if "experiment" in summary else pd.DataFrame()
    iid_coverage = _first_number(iid, "mean_coverage")
    shifted_coverage = _numeric_series(
        summary.loc[summary["experiment"] != "iid"] if "experiment" in summary else summary,
        "mean_coverage",
    )
    coverage_delta = _numeric_series(degradation, "coverage_delta")

    cards = [
        ("IID coverage", _fmt(iid_coverage)),
        (
            "Lowest shifted coverage",
            _fmt(float(shifted_coverage.min())) if not shifted_coverage.empty else "n/a",
        ),
        (
            "Largest coverage drop",
            _fmt(float(coverage_delta.min())) if not coverage_delta.empty else "n/a",
        ),
        ("Conditions evaluated", str(len(summary))),
    ]

    cards_html = "".join(
        f'<div class="card"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'
        for label, value in cards
    )

    summary_table = summary.to_html(index=False, border=0, classes="dataframe", escape=True)
    degradation_table = degradation.to_html(index=False, border=0, classes="dataframe", escape=True)

    plots = [
        ("Coverage", "robustness_coverage.png"),
        ("Accuracy", "robustness_accuracy.png"),
        ("Coverage degradation", "robustness_coverage_degradation.png"),
        ("Accuracy degradation", "robustness_accuracy_degradation.png"),
    ]
    plot_html = "".join(
        f'<figure><img src="{escape(filename)}" alt="{escape(label)}"><figcaption>{escape(label)}</figcaption></figure>'
        for label, filename in plots
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
:root {{ color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
body {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 64px; line-height: 1.5; }}
h1 {{ margin-bottom: 6px; }}
.lead {{ margin-top: 0; opacity: .78; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 24px 0; }}
.card {{ border: 1px solid color-mix(in srgb, currentColor 22%, transparent); border-radius: 12px; padding: 16px; }}
.card span {{ display: block; font-size: .86rem; opacity: .72; }}
.card strong {{ display: block; font-size: 1.5rem; margin-top: 4px; }}
.plots {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
figure {{ margin: 0; }}
img {{ width: 100%; height: auto; border-radius: 10px; border: 1px solid color-mix(in srgb, currentColor 18%, transparent); }}
figcaption {{ text-align: center; opacity: .75; margin-top: 6px; }}
.table-wrap {{ overflow-x: auto; margin: 14px 0 30px; }}
table {{ border-collapse: collapse; width: 100%; font-size: .88rem; }}
th, td {{ border-bottom: 1px solid color-mix(in srgb, currentColor 18%, transparent); text-align: left; padding: 8px; white-space: nowrap; }}
th {{ position: sticky; top: 0; }}
.note {{ padding: 12px 14px; border-left: 4px solid currentColor; opacity: .8; }}
</style>
</head>
<body>
<h1>{escape(title)}</h1>
<p class="lead">IID and controlled distribution-shift evaluation for conformal classification.</p>
<div class="cards">{cards_html}</div>
<p class="note">Negative accuracy, macro-F1, or coverage deltas mean degradation relative to the IID reference. These results describe the configured experiment, not a guarantee under arbitrary real-world shift.</p>
<h2>Plots</h2>
<div class="plots">{plot_html}</div>
<h2>Summary</h2>
<div class="table-wrap">{summary_table}</div>
<h2>IID-relative degradation</h2>
<div class="table-wrap">{degradation_table}</div>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")
    return output


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def _first_number(frame: pd.DataFrame, column: str) -> float | None:
    values = _numeric_series(frame, column)
    return float(values.iloc[0]) if not values.empty else None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
