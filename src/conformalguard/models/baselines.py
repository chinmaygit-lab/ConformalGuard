"""CPU-friendly baseline classifiers for ConformalGuard."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic_regression() -> Pipeline:
    """Create a scaled logistic-regression classification pipeline."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )