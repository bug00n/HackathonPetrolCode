"""Regression tests for episode-aware multi-horizon sulfur forecasting."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from source.contracts import SourceKind
from source.ml.features import SupervisedDataset
from source.ml.v2 import (
    EpisodeSafetyPredictor,
    build_episode_dataset,
    episode_sample_weights,
    event_metrics,
    rolling_month_folds,
    select_event_threshold,
)


def _event_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "as_of": pd.date_range("2025-01-01", periods=7, freq="10min", tz="UTC"),
            "target_available_at": pd.date_range(
                "2025-01-01 01:00", periods=7, freq="10min", tz="UTC"
            ),
            "baseline": [9.0] * 7,
            "crossing_60m": [True, True, True, False, True, False, False],
            "event_id": [1.0, 1.0, 1.0, np.nan, 2.0, np.nan, np.nan],
        }
    )


def test_long_positive_episode_has_same_total_weight_as_short_episode() -> None:
    frame = _event_frame()
    weights = episode_sample_weights(frame)

    assert weights[:3].sum() == pytest.approx(weights[4])


def test_event_metrics_count_one_long_episode_once() -> None:
    frame = _event_frame()
    probability = np.asarray([0.8, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1])

    metrics = event_metrics(frame, probability, 0.5)

    assert metrics["event_count"] == 2
    assert metrics["detected_events"] == 1
    assert metrics["event_false_negative_rate"] == pytest.approx(0.5)


def test_event_threshold_respects_false_alarm_budget() -> None:
    frame = _event_frame()
    probability = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.1])

    policy, metrics = select_event_threshold(frame, probability, false_alarm_budget=1 / 3)

    assert metrics["row_false_positive_rate"] <= 1 / 3
    assert metrics["event_false_positive_rate"] <= 1 / 3
    assert policy.threshold > 0.4


def test_rolling_month_fold_purges_labels_published_after_boundary() -> None:
    frame = pd.DataFrame(
        {
            "as_of": pd.to_datetime(["2024-01-01", "2024-01-31", "2024-02-01"], utc=True),
            "target_available_at": pd.to_datetime(
                ["2024-01-01 01:00", "2024-02-01 02:00", "2024-02-01 01:00"], utc=True
            ),
        }
    )

    folds = rolling_month_folds(frame, "2024-02-01", "2024-03-01")

    assert folds[0][0].tolist() == [0]
    assert folds[0][1].tolist() == [2]


def test_current_violation_always_triggers_event_alarm() -> None:
    class Delta:
        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(frame))

    class Risk:
        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            probability = np.full(len(frame), 0.1)
            return np.column_stack((1 - probability, probability))

    class Calibrator:
        def predict(self, probability: np.ndarray) -> np.ndarray:
            return probability

    predictor = EpisodeSafetyPredictor(
        "baseline",
        Delta(),
        Delta(),
        {horizon: Risk() for horizon in (10, 20, 30, 60)},
        {horizon: Calibrator() for horizon in (10, 20, 30, 60)},  # type: ignore[arg-type]
        SimpleNamespace(threshold=0.5),
        SimpleNamespace(),
    )

    assert predictor.predict_alarm(pd.DataFrame({"baseline": [9.0, 11.0]})).tolist() == [
        False,
        True,
    ]


def test_episode_dataset_builds_future_labels_and_causal_features() -> None:
    times = pd.date_range("2024-01-01", periods=25, freq="10min", tz="UTC")
    quality = pd.DataFrame(
        {
            "observation_id": [f"pak-{index}" for index in range(len(times))],
            "signal_id": ["ht:2:Mg.Sulfur"] * len(times),
            "source": ["pak"] * len(times),
            "validity": ["valid"] * len(times),
            "measured_at": times,
            "available_at": times,
            "unit": ["mg/kg"] * len(times),
            "value": [9.0] * 7 + [11.0] * 3 + [9.0] * 15,
        }
    )
    as_of = times[:19]
    base_frame = pd.DataFrame(
        {
            "as_of": as_of,
            "target_at": as_of + pd.Timedelta(minutes=60),
            "target_available_at": as_of + pd.Timedelta(minutes=60),
            "y": quality["value"].to_numpy()[6:25],
            "baseline": quality["value"].to_numpy()[:19],
            "ht:2:Mg.Sulfur__pak_last": quality["value"].to_numpy()[:19],
            "ht:2:Mg.Sulfur__pak_lag_30m": [np.nan] * 3 + quality["value"].to_list()[:16],
            "ht:2:Mg.Sulfur__pak_lag_60m": [np.nan] * 6 + quality["value"].to_list()[:13],
            "ht:2:Mg.Sulfur__pak_lag_180m": [np.nan] * 18 + quality["value"].to_list()[:1],
        }
    )
    names = tuple(base_frame.columns[5:])
    base = SupervisedDataset(
        base_frame,
        names,
        "ht:2:Mg.Sulfur",
        "mg/kg",
        SourceKind.PAK,
        SourceKind.PAK,
        "ht:2:Mg.Sulfur__pak_last",
        {},
        {},
    )
    data = SimpleNamespace(quality=quality)

    result = build_episode_dataset(data, base)  # type: ignore[arg-type]

    assert result.frame.loc[1, "crossing_60m"]
    assert result.frame.loc[1, "event_id"] == 1
    assert result.frame.loc[5, "crossing_20m"]
    assert not result.frame.loc[5, "crossing_10m"]
    assert "pak_range_60m" in result.feature_names
    assert "pak_minutes_since_crossing_10" in result.feature_names
