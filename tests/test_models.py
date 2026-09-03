import numpy as np
from sklearn.datasets import make_classification

from conformalguard.models import make_logistic_regression


def test_logistic_regression_baseline_fits_and_predicts():
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )

    model = make_logistic_regression()
    model.fit(X, y)

    predictions = model.predict(X[:20])

    assert predictions.shape == (20,)
    assert set(np.unique(predictions)).issubset({0, 1})