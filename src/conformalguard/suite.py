"""Cross-dataset benchmark suite for ConformalGuard."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine

from conformalguard.guard import ConformalGuard, GuardConfig

_DATASET_LOADERS = {
    "breast_cancer": load_breast_cancer,
    "iris": load_iris,
    "wine": load_wine,
    "digits": load_digits,
}


@dataclass(frozen=True)
class BuiltinDataset:
    """A built-in, network-free classification dataset."""

    name: str
    X: pd.DataFrame
    y: pd.Series


@dataclass(frozen=True)
class SuiteResult:
    """Aggregate result from several ConformalGuard benchmark runs."""

    dataset_names: tuple[str, ...]
    summary: pd.DataFrame
    degradation: pd.DataFrame
    scorecard: pd.DataFrame
    output_dir: Path | None = None


def available_datasets() -> tuple[str, ...]:
    """Return built-in benchmark dataset names."""
    return tuple(_DATASET_LOADERS)


def load_builtin_dataset(name: str) -> BuiltinDataset:
    """Load one supported scikit-learn dataset without network access."""
    try:
        loader = _DATASET_LOADERS[name]
    except KeyError as exc:
        choices = ", ".join(available_datasets())
        raise ValueError(
            f"Unknown dataset {name!r}. Available datasets: {choices}"
        ) from exc

    bunch = loader(as_frame=True)
    X = bunch.data.copy().reset_index(drop=True)
    y = pd.Series(bunch.target, name="target").reset_index(drop=True)
    return BuiltinDataset(name=name, X=X, y=y)


def run_builtin_suite(
    datasets: Iterable[str] | None = None,
    *,
    config: GuardConfig | None = None,
    output_dir: str | Path | None = None,
) -> SuiteResult:
    """Run ConformalGuard across several built-in datasets."""
    names = (
        tuple(datasets)
        if datasets is not None
        else (
            "breast_cancer",
            "iris",
            "wine",
        )
    )
    if not names:
        raise ValueError("At least one dataset is required.")

    unknown = [name for name in names if name not in _DATASET_LOADERS]
    if unknown:
        choices = ", ".join(available_datasets())
        raise ValueError(
            f"Unknown dataset(s): {', '.join(unknown)}. Available datasets: {choices}"
        )

    output = Path(output_dir) if output_dir is not None else None
    summaries: list[pd.DataFrame] = []
    degradations: list[pd.DataFrame] = []
    effective_config = config or GuardConfig()

    for name in names:
        dataset = load_builtin_dataset(name)
        guard = ConformalGuard(effective_config)
        guard.run(dataset.X, dataset.y)

        summary = guard.summary().copy()
        summary.insert(0, "dataset", name)
        summaries.append(summary)

        degradation = guard.degradation().copy()
        degradation.insert(0, "dataset", name)
        degradations.append(degradation)

        if output is not None:
            guard.save_report(output / name)

    summary_frame = pd.concat(summaries, ignore_index=True)
    degradation_frame = pd.concat(degradations, ignore_index=True)
    scorecard = suite_scorecard(summary_frame, degradation_frame)

    result = SuiteResult(
        dataset_names=names,
        summary=summary_frame,
        degradation=degradation_frame,
        scorecard=scorecard,
        output_dir=output,
    )

    if output is not None:
        write_suite_report(result, output)

    return result


def suite_scorecard(
    summary: pd.DataFrame,
    degradation: pd.DataFrame,
) -> pd.DataFrame:
    """Create one compact robustness row per dataset."""
    rows: list[dict[str, object]] = []

    for dataset, dataset_summary in summary.groupby("dataset", sort=False):
        iid = dataset_summary.loc[dataset_summary["experiment"] == "iid"]
        if len(iid) != 1:
            raise ValueError(f"Dataset {dataset!r} must contain exactly one IID row.")

        shifted = degradation.loc[degradation["dataset"] == dataset]
        if shifted.empty:
            raise ValueError(
                f"Dataset {dataset!r} does not contain shifted conditions."
            )

        iid_row = iid.iloc[0]

        worst_accuracy = shifted.loc[shifted["accuracy_delta"].idxmin()]
        worst_coverage = shifted.loc[shifted["coverage_delta"].idxmin()]

        rows.append(
            {
                "dataset": dataset,
                "iid_accuracy": float(iid_row["mean_accuracy"]),
                "iid_coverage": float(iid_row["mean_coverage"]),
                "worst_shift_accuracy": float(worst_accuracy["mean_accuracy"]),
                "worst_accuracy_experiment": str(worst_accuracy["experiment"]),
                "worst_accuracy_condition": str(worst_accuracy["condition"]),
                "worst_accuracy_delta": float(worst_accuracy["accuracy_delta"]),
                "worst_shift_coverage": float(worst_coverage["mean_coverage"]),
                "worst_coverage_experiment": str(worst_coverage["experiment"]),
                "worst_coverage_condition": str(worst_coverage["condition"]),
                "worst_coverage_delta": float(worst_coverage["coverage_delta"]),
            }
        )

    return pd.DataFrame(rows)


def write_suite_report(result: SuiteResult, output_dir: str | Path) -> dict[str, Path]:
    """Write aggregate suite CSV, JSON, and static HTML outputs."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_csv": output / "suite_summary.csv",
        "summary_json": output / "suite_summary.json",
        "degradation_csv": output / "suite_degradation.csv",
        "degradation_json": output / "suite_degradation.json",
        "scorecard_csv": output / "suite_scorecard.csv",
        "scorecard_json": output / "suite_scorecard.json",
        "html_report": output / "suite_report.html",
    }
    result.summary.to_csv(paths["summary_csv"], index=False)
    result.summary.to_json(paths["summary_json"], orient="records", indent=2)
    result.degradation.to_csv(paths["degradation_csv"], index=False)
    result.degradation.to_json(paths["degradation_json"], orient="records", indent=2)
    result.scorecard.to_csv(paths["scorecard_csv"], index=False)
    result.scorecard.to_json(paths["scorecard_json"], orient="records", indent=2)

    scorecard_html = result.scorecard.to_html(
        index=False, float_format=lambda x: f"{x:.4f}"
    )
    summary_html = result.summary.to_html(
        index=False, float_format=lambda x: f"{x:.4f}"
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ConformalGuard multi-dataset benchmark</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 18px}}table{{border-collapse:collapse;width:100%;margin:1rem 0 2rem;font-size:.92rem}}th,td{{border:1px solid #ddd;padding:7px 9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#f5f5f5}}.note{{color:#555}}</style></head><body>
<h1>ConformalGuard multi-dataset benchmark</h1>
<p class="note">Datasets: {", ".join(result.dataset_names)}. Values are empirical benchmark results, not universal guarantees.</p>
<h2>Robustness scorecard</h2>{scorecard_html}
<h2>All benchmark conditions</h2>{summary_html}
</body></html>"""
    paths["html_report"].write_text(html, encoding="utf-8")
    return paths
