"""Tests for constraint-aware sulfur risk and joint applicability."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from source.agents.quality import predict_quality
from source.config import load_scenario
from source.contracts import ProcessState
from source.ml.artifacts import LastValueRegressor, ModelBundle, ModelMetadata, feature_schema_hash
from source.ml.safety import (
    CompositeSafetyPredictor,
    PlattCalibrator,
    binary_alarm_metrics,
    fit_joint_applicability,
    published_lims_features,
    select_alarm_threshold,
)


def test_threshold_minimizes_misses_inside_false_alarm_budget() -> None:
    actual = np.asarray([False, False, False, False, True, True])
    probability = np.asarray([0.05, 0.10, 0.20, 0.80, 0.40, 0.90])

    policy, metrics = select_alarm_threshold(actual, probability, false_alarm_budget=0.25)

    assert policy.threshold == pytest.approx(0.4)
    assert metrics["false_negative_rate"] == 0.0
    assert metrics["false_positive_rate"] == pytest.approx(0.25)


def test_alarm_metrics_reject_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="finite"):
        binary_alarm_metrics([True], [np.nan], 0.5)


def test_joint_applicability_does_not_reject_one_marginally_unusual_feature() -> None:
    rng = np.random.default_rng(42)
    training = pd.DataFrame(
        {
            "baseline": rng.normal(9.0, 0.5, 500),
            "x": rng.normal(0.0, 1.0, 500),
            "correlated": rng.normal(0.0, 1.0, 500),
        }
    )
    domain = fit_joint_applicability(training, required_features=("baseline",))

    assert domain.assess(
        pd.DataFrame({"baseline": [9.0], "x": [2.0], "correlated": [0.0]})
    ).available
    missing = domain.assess(pd.DataFrame({"baseline": [np.nan], "x": [0.0], "correlated": [0.0]}))
    assert missing.available is False
    assert missing.reason_code == "FEATURES_UNAVAILABLE"


def test_composite_predictor_exposes_point_upper_probability_and_alarm() -> None:
    features = pd.DataFrame({"baseline": [9.0, 11.0], "x": [0.0, 1.0]})
    point = LastValueRegressor("baseline").fit(features)

    class Upper:
        def predict_upper(self, frame: pd.DataFrame) -> np.ndarray:
            return frame["baseline"].to_numpy(dtype=float) + 1.0

    class Risk:
        def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
            p = np.asarray([0.2, 0.8])
            return np.column_stack((1.0 - p, p))

    calibrator_estimator = SimpleNamespace(
        predict_proba=lambda values: np.column_stack(
            (1.0 - np.asarray([0.2, 0.8]), np.asarray([0.2, 0.8]))
        )
    )
    domain = fit_joint_applicability(features, required_features=("baseline",), coverage=0.75)
    predictor = CompositeSafetyPredictor(
        point,
        Upper(),
        Risk(),
        PlattCalibrator(calibrator_estimator),  # type: ignore[arg-type]
        SimpleNamespace(threshold=0.5),
        domain,
    )

    assert predictor.predict(features).tolist() == [9.0, 11.0]
    assert predictor.predict_upper(features).tolist() == [10.0, 12.0]
    assert predictor.predict_exceedance_probability(features).tolist() == pytest.approx([0.2, 0.8])
    assert predictor.predict_alarm(features).tolist() == [False, True]


def test_schema_12_bundle_validates_capability_and_old_schemas_remain_valid() -> None:
    names = ("baseline",)
    common = {
        "model_id": "safety-test",
        "model_sha256": "0" * 64,
        "model_type": "test",
        "training_dataset_id": "dataset",
        "git_commit": "commit",
        "python_version": "3.12.0",
        "sklearn_version": "1.9.0",
        "target_signal": "ht:2:Mg.Sulfur",
        "target_source": "pak",
        "target_unit": "mg/kg",
        "horizon_minutes": 60,
        "feature_names": names,
        "baseline_feature": "baseline",
        "tag_dictionary_sha256": "a" * 64,
        "feature_schema_hash": feature_schema_hash(names),
        "processing": {"feature_definition": {}},
        "time_boundaries": {
            "train_end_local": "2025-01-01",
            "validation_end_local": "2026-01-01",
            "source_timezone": "UTC",
        },
        "seed": 42,
        "applicability": {},
        "reports": (),
    }
    old = ModelMetadata(
        **common,
        schema_version="1.0",
        capabilities={"supports_forecast": True, "supports_actions": False},
    )
    safety = ModelMetadata(
        **common,
        schema_version="1.1",
        capabilities={
            "supports_forecast": True,
            "supports_actions": False,
            "supports_uncertainty": True,
            "supports_exceedance_probability": True,
        },
    )
    multi_horizon = ModelMetadata(
        **common,
        schema_version="1.2",
        capabilities={
            "supports_forecast": True,
            "supports_actions": False,
            "supports_uncertainty": True,
            "supports_exceedance_probability": True,
            "supports_multi_horizon": True,
        },
    )

    assert old.capabilities.supports_exceedance_probability is False
    assert safety.capabilities.supports_exceedance_probability is True
    assert multi_horizon.capabilities.supports_multi_horizon is True


def test_bundle_refuses_probability_when_capability_is_absent() -> None:
    metadata = SimpleNamespace(
        feature_names=("baseline",),
        capabilities=SimpleNamespace(supports_exceedance_probability=False),
    )
    bundle = ModelBundle(LastValueRegressor(), metadata)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="does not declare"):
        bundle.predict_exceedance_probability(pd.DataFrame({"baseline": [9.0]}))


def test_bundle_returns_complete_safety_contract() -> None:
    features = pd.DataFrame({"baseline": [9.0]})

    class Predictor:
        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return np.asarray([9.2])

        def predict_upper(self, frame: pd.DataFrame) -> np.ndarray:
            return np.asarray([10.4])

        def predict_exceedance_probability(self, frame: pd.DataFrame) -> np.ndarray:
            return np.asarray([0.7])

        def predict_alarm(self, frame: pd.DataFrame) -> np.ndarray:
            return np.asarray([True])

        def check_applicability(self, frame: pd.DataFrame) -> object:
            return SimpleNamespace(available=False, reason_code="OUT_OF_DOMAIN")

    metadata = SimpleNamespace(
        feature_names=("baseline",),
        capabilities=SimpleNamespace(
            supports_uncertainty=True,
            supports_exceedance_probability=True,
        ),
    )
    bundle = ModelBundle(Predictor(), metadata)  # type: ignore[arg-type]

    assert bundle.predict_safety(features) == [
        {
            "point": 9.2,
            "upper": 10.4,
            "exceedance_probability": 0.7,
            "alarm": True,
            "applicable": False,
            "reason_codes": ("OUT_OF_DOMAIN",),
        }
    ]


def test_history_quality_exposes_risk_and_blocks_on_alarm() -> None:
    state = ProcessState.model_validate_json(
        Path("global_tests/fixtures/contracts/process_state.json").read_text(encoding="utf-8")
    )
    scenario = load_scenario("config/scenarios/history.json")

    class RiskFixture:
        metadata = SimpleNamespace(
            model_id="risk-fixture",
            target_signal="ht:2:Mg.Sulfur",
            target_source="pak",
            target_unit="mg/kg",
            horizon_minutes=60,
            capabilities=SimpleNamespace(
                supports_forecast=True,
                supports_actions=False,
                supports_uncertainty=True,
                supports_exceedance_probability=True,
            ),
        )

        def predict(self, features: pd.DataFrame) -> np.ndarray:
            return np.asarray([9.0])

        def predict_upper(self, features: pd.DataFrame) -> np.ndarray:
            return np.asarray([9.8])

        def predict_exceedance_probability(self, features: pd.DataFrame) -> np.ndarray:
            return np.asarray([0.7])

        def predict_alarm(self, features: pd.DataFrame) -> np.ndarray:
            return np.asarray([True])

        def check_applicability(self, features: pd.DataFrame) -> object:
            return SimpleNamespace(available=True, reason_code=None)

    assessment = predict_quality(state, pd.DataFrame({"baseline": [9.0]}), RiskFixture(), scenario)

    assert assessment.status.value == "degraded"
    assert assessment.metrics["sulfur_exceedance_probability"].value == pytest.approx(0.7)
    assert {issue.code for issue in assessment.issues} == {"SULFUR_EXCEEDANCE_RISK"}


def test_published_lims_features_never_see_unpublished_analysis() -> None:
    quality = pd.DataFrame(
        {
            "observation_id": ["old", "hidden"],
            "signal_id": ["ht:2:Mg.Sulfur"] * 2,
            "source": ["lims"] * 2,
            "validity": ["valid"] * 2,
            "unit": ["mg/kg"] * 2,
            "measured_at": pd.to_datetime(["2025-01-01 00:00Z", "2025-01-01 06:00Z"]),
            "available_at": pd.to_datetime(["2025-01-01 04:00Z", "2025-01-01 10:00Z"]),
            "value": [8.0, 999.0],
        }
    )
    data = SimpleNamespace(quality=quality)

    features = published_lims_features(  # type: ignore[arg-type]
        data, pd.Series(pd.to_datetime(["2025-01-01 09:00Z"]))
    )

    assert features.loc[0, "lims_published_lag_0"] == pytest.approx(8.0)
    assert features.loc[0, "lims_age_minutes_0"] == pytest.approx(9 * 60)
