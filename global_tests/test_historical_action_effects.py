"""Tests for shadow-only matched historical action effects."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from source.ml.action_effects import (
    ACTION_HORIZONS,
    HistoricalActionDataset,
    HistoricalActionEffectModel,
    build_historical_action_dataset,
    evaluate_temporal_residualization,
    load_historical_action_model,
    save_historical_action_model,
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
            "ht:T11": np.full(len(timestamp), 100.0),
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


def test_historical_dataset_uses_only_p8_f19_actions_and_process_context() -> None:
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
        "ht:T11": 100.0,
        "ht:F26": 350.0,
    }

    estimate = model.estimate(state, "ht:P8", 0.01)
    payload = estimate.as_ui_payload()

    assert estimate.effect_by_horizon[60] == pytest.approx(-0.1)
    assert payload["advisory"] is False
    assert payload["title"] == "Модельный эффект по историческим эпизодам"
    assert payload["within_observed_domain"] is True
    assert payload["model_validated"] is False
    assert payload["safety_passes"] is True
    assert model.supports_actions is False


def test_historical_action_model_cannot_enable_controls() -> None:
    with pytest.raises(ValueError, match="research-only"):
        HistoricalActionEffectModel(
            {horizon: _Estimator() for horizon in ACTION_HORIZONS},
            {horizon: _Estimator() for horizon in ACTION_HORIZONS},
            {horizon: 0.0 for horizon in ACTION_HORIZONS},
            _Applicable(),  # type: ignore[arg-type]
            {"ht:P8": (-0.02, 0.02), "ht:F19": (-20.0, 20.0)},
            {},
            True,
        )


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
        "ht:T11": 100.0,
        "ht:F26": 350.0,
    }

    estimate = model.estimate(state, "ht:P8", 0.5)

    assert estimate.applicable is False
    assert "ACTION_DELTA_OUT_OF_OBSERVED_RANGE" in estimate.reason_codes
    assert "SULFUR_UPPER_LIMIT_60M" in estimate.reason_codes
    assert estimate.within_observed_domain is False
    assert estimate.safety_passes is False


def test_shadow_action_artifact_round_trip_is_never_advisory(tmp_path: Path) -> None:
    model = HistoricalActionEffectModel(
        {horizon: _Estimator() for horizon in ACTION_HORIZONS},
        {horizon: _Estimator() for horizon in ACTION_HORIZONS},
        {horizon: 0.0 for horizon in ACTION_HORIZONS},
        _Applicable(),  # type: ignore[arg-type]
        {"ht:P8": (-0.02, 0.02), "ht:F19": (-20.0, 20.0)},
        {"controls": ["ht:P8", "ht:F19"], "evidence_gate_passed": False},
    )

    directory = save_historical_action_model(
        tmp_path / "action-shadow-fixture", model, training_dataset_id="fixture"
    )
    loaded = load_historical_action_model(directory, trusted=True)

    assert loaded.supports_actions is False
    assert loaded.report["evidence_gate_passed"] is False
    with pytest.raises(ValueError, match="explicitly trusted"):
        load_historical_action_model(directory)
    with pytest.raises(ValueError, match="different prepared dataset"):
        load_historical_action_model(directory, trusted=True, expected_dataset_id="other")


def test_temporal_residualization_returns_research_only_report() -> None:
    timestamp = pd.date_range("2024-01-01", periods=120, freq="10min", tz="UTC")
    index = np.arange(len(timestamp), dtype=float)
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "baseline_sulfur": 8.0 + index * 0.001,
            "sulfur_slope_60m": np.full(len(index), 0.001),
            "sulfur_60m": 8.1 + index * 0.001,
            "sulfur_120m": 8.2 + index * 0.001,
            "sulfur_180m": 8.3 + index * 0.001,
            "ht:P8": 0.15 + np.sin(index / 10.0) * 0.01,
            "ht:F19": 200.0 + np.cos(index / 10.0) * 2.0,
            "ht:T11": np.full(len(index), 100.0),
            "ht:F26": 350.0 + np.sin(index / 7.0),
            "delta_ht:P8": np.where(index % 15 == 0, 0.01, 0.0),
            "delta_ht:F19": np.where(index % 20 == 0, 5.0, 0.0),
        }
    )
    dataset = HistoricalActionDataset(
        frame,
        {},
        {"ht:P8": (-0.01, 0.01), "ht:F19": (-5.0, 5.0)},
        1.0,
    )

    report = evaluate_temporal_residualization(
        dataset, train_end="2024-01-01 08:00", validation_end="2024-01-01 20:00"
    )

    assert report["supports_actions"] is False
    assert report["promotion_eligible"] is False
    assert set(report["horizons"]) == {"60", "120", "180"}
    assert len(report["horizons"]["60"]["effect_sign_by_fold"]) == 3
    assert set(report["horizons"]["60"]["effect_sign_stable"]) == {
        "ht:P8",
        "ht:F19",
    }
