from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

from conformalguard.experiments import (
    plot_robustness_accuracy,
    plot_robustness_coverage,
    robustness_summary_frame,
    run_robustness_benchmark,
    write_robustness_summary_csv,
    write_robustness_summary_json,
)


def main() -> None:
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    result = run_robustness_benchmark(
        X,
        y,
        label_target_proportions=(
            {0: 0.50, 1: 0.50},
            {0: 0.70, 1: 0.30},
            {0: 0.30, 1: 0.70},
        ),
        covariate_severities=(0.0, 0.5, 1.0, 2.0),
        seeds=(11, 42, 73),
        confidence_level=0.90,
    )

    frame = robustness_summary_frame(result)
    print(frame.to_string(index=False))

    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    write_robustness_summary_csv(
        result,
        output_dir / "robustness_summary.csv",
    )

    write_robustness_summary_json(
        result,
        output_dir / "robustness_summary.json",
    )

    ax = plot_robustness_accuracy(result)
    ax.figure.tight_layout()
    ax.figure.savefig(
        output_dir / "robustness_accuracy.png",
        dpi=150,
    )
    plt.close(ax.figure)

    ax = plot_robustness_coverage(result)
    ax.figure.tight_layout()
    ax.figure.savefig(
        output_dir / "robustness_coverage.png",
        dpi=150,
    )
    plt.close(ax.figure)


if __name__ == "__main__":
    main()
