"""Command-line interface for ConformalGuard."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from conformalguard.guard import ConformalGuard, GuardConfig


def _csv_numbers(value: str, *, cast=float) -> tuple:
    try:
        items = tuple(
            cast(piece.strip()) for piece in value.split(",") if piece.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not items:
        raise argparse.ArgumentTypeError("Provide at least one comma-separated value.")
    return items


def _float_tuple(value: str) -> tuple[float, ...]:
    return _csv_numbers(value, cast=float)


def _int_tuple(value: str) -> tuple[int, ...]:
    return _csv_numbers(value, cast=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conformalguard",
        description="Stress-test conformal prediction under controlled distribution shift.",
    )
    parser.add_argument(
        "--version", action="store_true", help="Print version and exit."
    )
    subparsers = parser.add_subparsers(dest="command")

    demo = subparsers.add_parser("demo", help="Run the built-in breast-cancer demo.")
    _add_common_arguments(demo)

    benchmark = subparsers.add_parser(
        "benchmark", help="Run a benchmark on a CSV file."
    )
    benchmark.add_argument("csv", type=Path, help="Input CSV file.")
    benchmark.add_argument("--target", required=True, help="Target column name.")
    _add_common_arguments(benchmark)

    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--conformity-score", default="lac")
    parser.add_argument("--seeds", type=_int_tuple, default=(11, 42, 73))
    parser.add_argument(
        "--covariate-severities",
        type=_float_tuple,
        default=(0.0, 0.5, 1.0, 2.0),
    )
    parser.add_argument(
        "--concept-severities",
        type=_float_tuple,
        default=(0.0, 0.25, 0.50, 0.75, 1.0),
    )
    parser.add_argument("--feature-fraction", type=float, default=0.50)
    parser.add_argument("--concept-feature", default=None)


def _config_from_args(args: argparse.Namespace) -> GuardConfig:
    return GuardConfig(
        confidence_level=args.confidence,
        conformity_score=args.conformity_score,
        seeds=tuple(args.seeds),
        covariate_severities=tuple(args.covariate_severities),
        concept_severities=tuple(args.concept_severities),
        feature_fraction=args.feature_fraction,
        concept_feature=args.concept_feature,
    )


def _run_guard(X: pd.DataFrame, y: pd.Series, args: argparse.Namespace) -> int:
    guard = ConformalGuard(_config_from_args(args))
    guard.run(X, y)
    summary = guard.summary()
    paths = guard.save_report(args.out)

    display_columns = [
        "experiment",
        "condition",
        "mean_accuracy",
        "mean_coverage",
        "mean_set_size",
    ]
    available = [name for name in display_columns if name in summary.columns]
    print(summary[available].to_string(index=False))
    print(f"\nSaved report to: {args.out.resolve()}")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from conformalguard import __version__

        print(__version__)
        return 0

    if args.command == "demo":
        from sklearn.datasets import load_breast_cancer

        data = load_breast_cancer(as_frame=True)
        return _run_guard(data.data, data.target, args)

    if args.command == "benchmark":
        if not args.csv.exists():
            parser.error(f"CSV file does not exist: {args.csv}")
        frame = pd.read_csv(args.csv)
        if args.target not in frame.columns:
            parser.error(f"Target column {args.target!r} was not found in {args.csv}.")
        y = frame.pop(args.target)
        return _run_guard(frame, y, args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
