"""High-level ConformalGuard facade.

This module intentionally keeps imports from the experiment stack lazy so the
public object can be imported quickly and tested independently of a benchmark
run.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class GuardConfig:
    """Configuration for a full robustness stress test."""

    confidence_level: float = 0.90
    conformity_score: str = "lac"
    seeds: tuple[int, ...] = (11, 42, 73)
    covariate_severities: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
    concept_severities: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
    feature_fraction: float = 0.50
    concept_feature: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0 and 1.")
        if not self.seeds:
            raise ValueError("At least one random seed is required.")
        if not self.covariate_severities:
            raise ValueError("At least one covariate severity is required.")
        if any(value < 0.0 for value in self.covariate_severities):
            raise ValueError("covariate_severities must be non-negative.")
        if not self.concept_severities:
            raise ValueError("At least one concept severity is required.")
        if any(not 0.0 <= value <= 1.0 for value in self.concept_severities):
            raise ValueError("concept_severities must be between 0 and 1.")
        if not 0.0 < self.feature_fraction <= 1.0:
            raise ValueError("feature_fraction must be in (0, 1].")


class ConformalGuard:
    """One-object interface for running and reporting robustness benchmarks.

    Example
    -------
    >>> guard = ConformalGuard()
    >>> result = guard.run(X, y)
    >>> print(guard.summary())
    >>> guard.save_report("artifacts")
    """

    def __init__(
        self,
        config: GuardConfig | None = None,
        *,
        label_target_proportions: Iterable[Mapping[Any, float]] | None = None,
    ) -> None:
        self.config = config or GuardConfig()
        self.label_target_proportions = (
            tuple(dict(item) for item in label_target_proportions)
            if label_target_proportions is not None
            else None
        )
        self.result_: Any | None = None

    def run(self, X: pd.DataFrame, y: Sequence[Any] | pd.Series) -> Any:
        """Run IID plus covariate-, label-, and concept-shift stress tests."""
        X_frame = _validate_feature_frame(X)
        y_series = _validate_target(y, expected_rows=len(X_frame))
        targets = self.label_target_proportions or _default_label_targets(y_series)

        from conformalguard.experiments.robustness import run_robustness_benchmark

        self.result_ = run_robustness_benchmark(
            X_frame,
            y_series,
            label_target_proportions=targets,
            covariate_severities=self.config.covariate_severities,
            concept_severities=self.config.concept_severities,
            concept_feature=self.config.concept_feature,
            seeds=self.config.seeds,
            confidence_level=self.config.confidence_level,
            feature_fraction=self.config.feature_fraction,
            conformity_score=self.config.conformity_score,
        )
        return self.result_

    stress_test = run

    def summary(self) -> pd.DataFrame:
        """Return the normalized benchmark summary."""
        result = self._require_result()
        from conformalguard.experiments.robustness_reporting import (
            robustness_summary_frame,
        )

        return robustness_summary_frame(result)

    def degradation(self) -> pd.DataFrame:
        """Return every shifted condition relative to the IID baseline."""
        result = self._require_result()
        from conformalguard.experiments.robustness_degradation import (
            robustness_degradation_frame,
        )

        return robustness_degradation_frame(result)

    def save_report(self, output_dir: str | Path = "artifacts") -> dict[str, Path]:
        """Write CSV/JSON summaries and four standard plots to ``output_dir``."""
        result = self._require_result()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        from conformalguard.experiments.robustness_degradation import (
            write_robustness_degradation_csv,
            write_robustness_degradation_json,
        )
        from conformalguard.experiments.robustness_degradation_plotting import (
            plot_robustness_accuracy_degradation,
            plot_robustness_coverage_degradation,
        )
        from conformalguard.experiments.robustness_plotting import (
            plot_robustness_accuracy,
            plot_robustness_coverage,
        )
        from conformalguard.experiments.robustness_reporting import (
            write_robustness_summary_csv,
            write_robustness_summary_json,
        )

        paths: dict[str, Path] = {
            "summary_csv": write_robustness_summary_csv(
                result, output / "robustness_summary.csv"
            ),
            "summary_json": write_robustness_summary_json(
                result, output / "robustness_summary.json"
            ),
            "degradation_csv": write_robustness_degradation_csv(
                result, output / "robustness_degradation.csv"
            ),
            "degradation_json": write_robustness_degradation_json(
                result, output / "robustness_degradation.json"
            ),
        }

        import matplotlib.pyplot as plt

        plotters = {
            "accuracy_plot": (
                plot_robustness_accuracy,
                output / "robustness_accuracy.png",
            ),
            "coverage_plot": (
                plot_robustness_coverage,
                output / "robustness_coverage.png",
            ),
            "accuracy_degradation_plot": (
                plot_robustness_accuracy_degradation,
                output / "robustness_accuracy_degradation.png",
            ),
            "coverage_degradation_plot": (
                plot_robustness_coverage_degradation,
                output / "robustness_coverage_degradation.png",
            ),
        }
        for key, (plotter, path) in plotters.items():
            ax = plotter(result)
            ax.figure.tight_layout()
            ax.figure.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(ax.figure)
            paths[key] = path

        from conformalguard.html_report import write_html_report

        paths["html_report"] = write_html_report(
            self.summary(),
            self.degradation(),
            output / "report.html",
        )
        return paths

    def _require_result(self) -> Any:
        if self.result_ is None:
            raise RuntimeError("Run guard.run(X, y) before requesting results.")
        return self.result_


def _validate_feature_frame(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if X.empty:
        raise ValueError("X must contain at least one row.")
    if X.columns.duplicated().any():
        raise ValueError("X contains duplicate column names.")
    non_numeric = [
        name for name in X.columns if not pd.api.types.is_numeric_dtype(X[name])
    ]
    if non_numeric:
        joined = ", ".join(map(str, non_numeric[:5]))
        raise ValueError(
            "ConformalGuard's current controlled shift benchmark expects numeric "
            f"feature columns. Non-numeric columns: {joined}"
        )
    return X


def _validate_target(y: Sequence[Any] | pd.Series, *, expected_rows: int) -> pd.Series:
    series = y if isinstance(y, pd.Series) else pd.Series(y)
    series = series.reset_index(drop=True)
    if len(series) != expected_rows:
        raise ValueError("X and y must contain the same number of rows.")
    if series.isna().any():
        raise ValueError("y must not contain missing values.")
    if series.nunique(dropna=True) < 2:
        raise ValueError("y must contain at least two classes.")
    return series


def _default_label_targets(y: pd.Series) -> tuple[dict[Any, float], ...]:
    """Create deterministic label-shift scenarios that sum exactly to one."""
    labels = sorted(pd.unique(y), key=repr)
    n_labels = len(labels)
    if n_labels < 2:
        raise ValueError("At least two classes are required for label shift.")

    uniform = {label: 1.0 / n_labels for label in labels}
    if n_labels == 2:
        first, second = labels
        return (
            uniform,
            {first: 0.70, second: 0.30},
            {first: 0.30, second: 0.70},
        )

    denominator = n_labels + 1.0
    first_heavy = {label: 1.0 / denominator for label in labels}
    first_heavy[labels[0]] = 2.0 / denominator
    last_heavy = {label: 1.0 / denominator for label in labels}
    last_heavy[labels[-1]] = 2.0 / denominator
    return uniform, first_heavy, last_heavy
