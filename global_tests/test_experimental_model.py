"""Regression tests for the separate high-capacity research ensemble."""

from __future__ import annotations

import numpy as np
import pandas as pd

from source.ml.experimental import MeanEstimator, add_experimental_interactions
from source.ml.v2 import EpisodeDataset


class _FakeEstimator:
    """Minimal sklearn-shaped object for testing ensemble arithmetic."""

    def __init__(self, prediction: float, probability: float) -> None:
        self.prediction = prediction
        self.probability = probability

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.prediction, dtype=float)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(features), self.probability, dtype=float)
        return np.column_stack((1.0 - positive, positive))


def test_mean_estimator_averages_point_and_probability_predictions() -> None:
    """The ensemble must average members without changing the row count."""
    features = pd.DataFrame({"x": [1.0, 2.0]})
    ensemble = MeanEstimator((_FakeEstimator(1.0, 0.2), _FakeEstimator(3.0, 0.6)))
    np.testing.assert_allclose(ensemble.predict(features), [2.0, 2.0])
    np.testing.assert_allclose(ensemble.predict_proba(features), [[0.6, 0.4], [0.6, 0.4]])


def test_experimental_interactions_are_as_of_only_and_ordered() -> None:
    """Derived nonlinear columns use only existing point-in-time columns."""
    frame = pd.DataFrame(
        {
            "baseline": [2.0, 4.0],
            "ht:P8__lag_0m": [3.0, 5.0],
            "ht:F19__lag_0m": [2.0, 4.0],
            "ht:T11__lag_0m": [10.0, 20.0],
            "pak_slope_60m": [0.1, -0.2],
            "ht:F19__delta_60m": [0.5, -0.5],
            "ht:T11__delta_60m": [1.0, -1.0],
        }
    )
    dataset = EpisodeDataset(frame, tuple(frame.columns), "baseline")
    enriched = add_experimental_interactions(dataset)
    assert len(enriched.frame) == len(frame)
    assert enriched.feature_names[: len(frame.columns)] == tuple(frame.columns)
    assert "exp_p8_x_f19" in enriched.feature_names
    assert "exp_sulfur_sq" in enriched.feature_names
    np.testing.assert_allclose(enriched.frame["exp_p8_x_f19"], [6.0, 20.0])
    np.testing.assert_allclose(enriched.frame["exp_sulfur_sq"], [4.0, 16.0])
    assert not any(name.startswith("y_") for name in enriched.feature_names)
