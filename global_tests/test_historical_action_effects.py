"""Tests for shadow-only matched historical action effects."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from source.ml.action_effects import (
    ACTION_HORIZONS,
    HistoricalActionEffectModel,
    build_historical_action_dataset,
)
from source.ml.uncertainty import ApplicabilityResult


class _Estimator:
    def __init__(self, action_coefficient: float = 0.0) -> None:
        self.action_coefficient = action_coefficient

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["baseline_sulfur"].to_numpy(dtype=float) + self.action_coefficient * frame[
            "delta_ht:P8"
        ].to_numpy(dtype=float)


class _Applicable:
    def assess(self, _: pd.DataFrame) -> ApplicabilityResult:
        return ApplicabilityResult(True, None, ())


def _prepared_data() -> SimpleNamespace:
    timestamp = pd.date_range("2023-01-01", periods=240, freq="10min", tz="UTC")
    p8 = np.full(len(timestamp), 0.15)
    f19 = np.full(len(timestamp), 200.0)
    p8[30:60] = 0.17
    f19[100:130] = 215.0
    sulfur = 8.0 + np.linspace(0.0, 0.5, len(timestamp))
    telemetry = pd.DataFrame(
        {
            "timestamp": timestamp,
            "ht:P8": p8,
            "ht:F19": f19,
            "ht:F26": np.full(len(timestamp), 350.0),
        }
    )
    quality = pd.DataFrame(
        {
            "signal_id": ["ht:2:Mg.Sulfur"] * len(timestamp),
            "source": ["pak"] * len(timestamp),
            "validity": ["valid"] * len(timestamp),
            "unit": ["mg/kg"] * len(timestamp),
            "measured_at": timestamp,
            "value": sulfur,
        }
    )
    return SimpleNamespace(telemetry=telemetry, quality=quality)


def test_historical_dataset_uses_only_p8_f19_actions_and_f26_as_context() -> None:
    dataset = build_historical_action_dataset(
        _prepared_data(),  # type: ignore[arg-type]
        train_end="2024-01-01",
        threshold_quantile=0.9,
    )

    assert set(dataset.thresholds) == {"ht:P8", "ht:F19"}
    assert "delta_ht:F26" not in dataset.frame
    assert dataset.frame["pair_id"].value_counts().eq(2).all()
    assert dataset.frame.loc[dataset.frame["is_action_episode"], "delta_ht:P8"].ne(0).any()
    assert dataset.frame.loc[dataset.frame["is_action_episode"], "delta_ht:F19"].ne(0).any()


def test_historical_estimate_is_ui_ready_but_never_advisory() -> None:
    model = HistoricalActionEffectModel(
        {horizon: _Estimator(-10.0) for horizon in ACTION_HORIZONS},
        {horizon: _Estimator(-10.0) for horizon in ACTION_HORIZONS},
        {horizon: 0.5 for horizon in ACTION_HORIZONS},
        _Applicable(),  # type: ignore[arg-type]
        {"ht:P8": (-0.02, 0.02), "ht:F19": (-20.0, 20.0)},
        {},
    )
    state = {
        "baseline_sulfur": 9.0,
        "sulfur_slope_60m": 0.0,
        "ht:P8": 0.15,
        "ht:F19": 200.0,
        "ht:F26": 350.0,
    }

    estimate = model.estimate(state, "ht:P8", 0.01)
    payload = estimate.as_ui_payload()

    assert estimate.effect_by_horizon[60] == pytest.approx(-0.1)
    assert payload["advisory"] is False
    assert payload["title"] == "Модельный эффект по историческим эпизодам"
    assert model.supports_actions is False


def test_unseen_delta_or_unsafe_upper_abstains() -> None:
    model = HistoricalActionEffectModel(
        {horizon: _Estimator() for horizon in ACTION_HORIZONS},
        {horizon: _Estimator() for horizon in ACTION_HORIZONS},
        {horizon: 2.0 for horizon in ACTION_HORIZONS},
        _Applicable(),  # type: ignore[arg-type]
        {"ht:P8": (-0.02, 0.02), "ht:F19": (-20.0, 20.0)},
        {},
    )
    state = {
        "baseline_sulfur": 9.0,
        "sulfur_slope_60m": 0.0,
        "ht:P8": 0.15,
        "ht:F19": 200.0,
        "ht:F26": 350.0,
    }

    estimate = model.estimate(state, "ht:P8", 0.5)

    assert estimate.applicable is False
    assert "ACTION_DELTA_OUT_OF_OBSERVED_RANGE" in estimate.reason_codes
    assert "SULFUR_UPPER_LIMIT_60M" in estimate.reason_codes
