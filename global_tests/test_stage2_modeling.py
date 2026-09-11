"""Fast synthetic checks for stage-2 training, evaluation and artifacts."""

from __future__ import annotations

import platform
from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn

import source.ml.train as training
from source.agents.quality import predict_quality
from source.config import load_scenario
from source.contracts import EstimateBasis, ProcessState
from source.ml.artifacts import (
    LastValueRegressor,
    ModelBundle,
    ModelMetadata,
    feature_schema_hash,
    load_model,
    save_model,
)
from source.ml.evaluate import (
    evaluate_model,
    expanding_purged_folds,
    make_outer_split,
    regression_metrics,
)


@dataclass(frozen=True)
class SyntheticDataset:
    """Minimal duck-typed counterpart of ``features.SupervisedDataset``."""

    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    target_signal_id: str = "ht:sulfur"
    target_unit: str = "mg/kg"
    target_source: str = "pak"
    feature_definition: dict[str, object] | None = None
    excluded_counts: dict[str, int] | None = None


def _metadata(model_id: str) -> dict[str, object]:
    names = ("baseline", "x")
    return {
        "model_id": model_id,
        "schema_version": "1.0",
        "model_sha256": "0" * 64,
        "model_type": "last_value",
        "training_dataset_id": "synthetic001",
        "git_commit": "test-commit",
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "target_signal": "ht:sulfur",
        "target_source": "pak",
        "target_unit": "mg/kg",
        "horizon_minutes": 60,
        "feature_names": names,
        "baseline_feature": "baseline",
        "tag_dictionary_sha256": "a" * 64,
        "feature_schema_hash": feature_schema_hash(names),
        "processing": {},
        "time_boundaries": {
            "train_end_local": "2025-01-01",
            "validation_end_local": "2026-01-01",
            "source_timezone": "UTC",
            "calibration_start": None,
            "calibration_end": None,
        },
        "seed": 42,
        "capabilities": {"supports_forecast": True, "supports_actions": False},
        "applicability": {"action_comparison": "forbidden"},
        "reports": ("metrics.json",),
    }


def _synthetic_dataset() -> SyntheticDataset:
    as_of = pd.date_range("2023-01-01", "2026-04-01", freq="14D", tz="UTC")
    phase = np.linspace(0.0, 8.0, len(as_of))
    baseline = 9.5 + np.sin(phase)
    frame = pd.DataFrame(
        {
            "observation_id": [f"target-{index}" for index in range(len(as_of))],
            "as_of": as_of,
            "target_at": as_of + pd.Timedelta(minutes=60),
            "target_available_at": as_of + pd.Timedelta(minutes=60),
            "target_source": "pak",
            "target_signal": "ht:sulfur",
            "target_unit": "mg/kg",
            "y": baseline,
            "baseline": baseline,
            "x": np.cos(phase),
        }
    )
    return SyntheticDataset(
        frame=frame,
        feature_names=("baseline", "x"),
        feature_definition={"horizon_minutes": 60, "lags_minutes": [10, 30, 60, 180]},
        excluded_counts={},
    )


def test_metrics_use_strict_threshold_and_null_denominators() -> None:
    metrics = regression_metrics(
        np.array([10.0, 10.0001, 8.0]),
        np.array([10.1, 10.0, 8.0]),
    )

    assert metrics["near_threshold"]["n"] == 3
    assert metrics["missed_exceedances"] == {
        "count": 1,
        "denominator": 1,
        "rate": 1.0,
    }
    assert metrics["false_alarms"] == {
        "count": 1,
        "denominator": 2,
        "rate": 0.5,
    }
    no_exceedance = regression_metrics(np.array([8.0, 9.0]), np.array([8.0, 9.0]))
    assert no_exceedance["missed_exceedances"]["rate"] is None
    all_exceed = regression_metrics(np.array([11.0]), np.array([11.0]))
    assert all_exceed["false_alarms"]["rate"] is None


def test_outer_split_purges_targets_crossing_calendar_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "id": ["train", "train-cross", "validation", "validation-cross", "test"],
            "as_of": pd.to_datetime(
                [
                    "2024-12-31T22:00:00Z",
                    "2024-12-31T23:30:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-12-31T23:30:00Z",
                    "2026-01-01T00:00:00Z",
                ],
                utc=True,
            ),
            "target_at": pd.to_datetime(
                [
                    "2024-12-31T23:00:00Z",
                    "2025-01-01T00:30:00Z",
                    "2025-01-01T01:00:00Z",
                    "2026-01-01T00:30:00Z",
                    "2026-01-01T01:00:00Z",
                ],
                utc=True,
            ),
            "target_available_at": pd.to_datetime(
                [
                    "2024-12-31T23:00:00Z",
                    "2025-01-01T00:30:00Z",
                    "2025-01-01T01:00:00Z",
                    "2026-01-01T00:30:00Z",
                    "2026-01-01T01:00:00Z",
                ],
                utc=True,
            ),
        }
    )

    split = make_outer_split(frame, source_timezone="UTC")

    assert frame.iloc[split.train]["id"].tolist() == ["train"]
    assert frame.iloc[split.validation]["id"].tolist() == ["validation"]
    assert frame.iloc[split.test]["id"].tolist() == ["test"]


def test_expanding_folds_purge_target_and_publication_time() -> None:
    as_of = pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "as_of": as_of,
            "target_at": as_of + pd.Timedelta(hours=1),
            "target_available_at": as_of + pd.Timedelta(hours=2),
        }
    )

    folds = expanding_purged_folds(frame, n_splits=3)

    assert len(folds) == 3
    previous_train_size = 0
    for train_indices, validation_indices in folds:
        validation_start = frame.iloc[validation_indices]["as_of"].min()
        train = frame.iloc[train_indices]
        assert train["as_of"].max() < validation_start
        assert train["target_at"].max() < validation_start
        assert train["target_available_at"].max() < validation_start
        assert len(train_indices) > previous_train_size
        previous_train_size = len(train_indices)


def test_selection_limits_misses_then_breaks_mae_ties_by_simplicity() -> None:
    comparison = {
        "last_value": {"mae": 1.0, "missed_exceedances": {"count": 1}},
        "ridge": {"mae": 1.0, "missed_exceedances": {"count": 1}},
        "hist_gradient_boosting": {
            "mae": 0.5,
            "missed_exceedances": {"count": 2},
        },
    }
    assert training.select_model(comparison) == "last_value"

    comparison["ridge"]["mae"] = 0.9
    assert training.select_model(comparison) == "ridge"


def test_artifact_round_trip_checks_trust_schema_order_and_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_id = "last-value-test"
    x = pd.DataFrame({"baseline": [8.0, 9.0], "x": [1.0, 2.0]})
    predictor = LastValueRegressor().fit(x)
    directory = tmp_path / model_id
    saved = save_model(directory, predictor, _metadata(model_id), {"selected_model": model_id})
    assert saved.predict(x).tolist() == [8.0, 9.0]

    with pytest.raises(ValueError, match="trusted"):
        load_model(directory)
    loaded = load_model(directory, trusted=True, expected_horizon_minutes=60)
    assert loaded.predict(x).tolist() == [8.0, 9.0]
    with pytest.raises(ValueError, match="feature order mismatch"):
        loaded.predict(x[["x", "baseline"]])

    monkeypatch.setattr(joblib, "load", lambda path: pytest.fail(f"loaded {path}"))
    with pytest.raises(ValueError, match="horizon_minutes"):
        load_model(directory, trusted=True, expected_horizon_minutes=30)
    model_path = directory / "model.joblib"
    model_path.write_bytes(model_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_model(directory, trusted=True)


def test_training_selects_before_test_and_evaluates_on_common_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(training, "RIDGE_ALPHAS", (1.0,))
    monkeypatch.setattr(
        training,
        "HGB_CONFIGS",
        (
            {
                "learning_rate": 0.1,
                "max_iter": 5,
                "max_leaf_nodes": 5,
                "min_samples_leaf": 2,
                "l2_regularization": 0.0,
            },
        ),
    )
    dataset = _synthetic_dataset()
    common = {
        "training_dataset_id": "synthetic001",
        "tag_dictionary_sha256": "a" * 64,
        "git_commit": "test-commit",
        "source_timezone": "UTC",
    }
    first = training.train_model(dataset, models_root=tmp_path / "one", **common)

    changed_frame = dataset.frame.copy()
    changed_frame.loc[changed_frame["as_of"] >= pd.Timestamp("2026-01-01", tz="UTC"), "y"] = 999
    changed = replace(dataset, frame=changed_frame)
    second = training.train_model(changed, models_root=tmp_path / "two", **common)

    assert first.metrics == second.metrics
    assert first.bundle.metadata.model_id == second.bundle.metadata.model_id
    assert first.bundle.metadata.model_type == "last_value"
    assert first.bundle.metadata.capabilities.supports_forecast is True
    assert first.bundle.metadata.capabilities.supports_actions is False
    evaluation = evaluate_model(dataset, first.bundle, split="test", source_timezone="UTC")
    assert evaluation.report["metrics"]["baseline"]["mae"] == pytest.approx(0.0)
    assert evaluation.report["metrics"]["model"]["mae"] == pytest.approx(0.0)


def test_evaluation_uses_artifact_boundaries_and_timezone() -> None:
    dataset = _synthetic_dataset()
    x = dataset.frame.loc[:, list(dataset.feature_names)]
    predictor = LastValueRegressor().fit(x)
    metadata_payload = _metadata("custom-boundaries")
    metadata_payload["time_boundaries"] = {
        "train_end_local": "2025-03-01",
        "validation_end_local": "2026-03-01",
        "source_timezone": "UTC",
        "calibration_start": None,
        "calibration_end": None,
    }
    model = ModelBundle(
        predictor=predictor,
        metadata=ModelMetadata(**metadata_payload),
    )

    result = evaluate_model(dataset, model, split="test")

    assert result.residuals["as_of"].min() >= pd.Timestamp("2026-03-01", tz="UTC")
    with pytest.raises(ValueError, match="timezone"):
        evaluate_model(dataset, model, split="test", source_timezone="Europe/Moscow")


def test_training_and_evaluation_reject_mislabeled_horizon(tmp_path: Path) -> None:
    dataset = _synthetic_dataset()
    bad_frame = dataset.frame.copy()
    bad_frame["target_at"] = bad_frame["as_of"] + pd.Timedelta(minutes=30)
    bad_frame["target_available_at"] = bad_frame["target_at"]
    mislabeled = replace(dataset, frame=bad_frame)

    with pytest.raises(ValueError, match="horizon"):
        training.train_model(
            mislabeled,
            models_root=tmp_path,
            training_dataset_id="synthetic001",
            tag_dictionary_sha256="a" * 64,
            git_commit="test-commit",
            source_timezone="UTC",
        )

    predictor = LastValueRegressor().fit(dataset.frame.loc[:, list(dataset.feature_names)])
    model = ModelBundle(
        predictor=predictor,
        metadata=ModelMetadata(**_metadata("horizon-check")),
    )
    with pytest.raises(ValueError, match="horizon"):
        evaluate_model(mislabeled, model, split="test")


def test_inner_tuning_keeps_safe_configuration_despite_lower_unsafe_mae(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    y = np.where(np.arange(len(as_of)) % 4 == 0, 11.0, 9.0)
    frame = pd.DataFrame(
        {
            "as_of": as_of,
            "target_at": as_of + pd.Timedelta(minutes=60),
            "target_available_at": as_of + pd.Timedelta(minutes=60),
            "y": y,
            "baseline": y,
            "unsafe": np.where(y > 10.0, 9.0, y),
            "safe": np.where(y > 10.0, y, 9.8),
        }
    )

    class FakeEstimator:
        def __init__(self, column: str) -> None:
            self.column = column

        def predict(self, features: pd.DataFrame) -> np.ndarray:
            return features[self.column].to_numpy(dtype=float)

    def fake_fit(
        family: str,
        params: dict[str, float],
        fit_frame: pd.DataFrame,
        feature_names: tuple[str, ...],
        baseline_feature: str,
        seed: int,
    ) -> FakeEstimator:
        del family, fit_frame, feature_names, baseline_feature, seed
        return FakeEstimator("unsafe" if params["alpha"] == 0.1 else "safe")

    monkeypatch.setattr(training, "_fit_estimator", fake_fit)

    params, report = training._tune_family(
        "ridge",
        frame,
        ("baseline", "unsafe", "safe"),
        "baseline",
        (0.1, 1.0),
        42,
    )

    assert params == {"alpha": 1.0}
    guards = {
        candidate["parameters"]["alpha"]: candidate["passes_baseline_miss_guard"]
        for candidate in report["candidates"]
    }
    assert guards == {0.1: False, 1.0: True}


def test_history_quality_exposes_point_forecast_without_action_or_interval() -> None:
    state = ProcessState.model_validate_json(
        Path("global_tests/fixtures/contracts/process_state.json").read_text(encoding="utf-8")
    )
    scenario = load_scenario("config/scenarios/history.json")
    feature_name = "ht:2:Mg.Sulfur__pak_last"
    features = pd.DataFrame({feature_name: [9.2]})
    predictor = LastValueRegressor(feature_name).fit(features)
    metadata = ModelMetadata(
        **{
            **_metadata("pak-point-forecast"),
            "model_id": "pak-point-forecast",
            "model_sha256": "0" * 64,
            "feature_names": (feature_name,),
            "baseline_feature": feature_name,
            "target_signal": "ht:2:Mg.Sulfur",
            "feature_schema_hash": feature_schema_hash((feature_name,)),
        }
    )
    model = ModelBundle(predictor=predictor, metadata=metadata)

    assessment = predict_quality(state, features, model, scenario)

    assert assessment.status.value == "degraded"
    assert assessment.metrics["sulfur"].value == pytest.approx(9.2)
    assert assessment.metrics["sulfur"].basis is EstimateBasis.FORECAST
    assert assessment.metrics["sulfur"].upper is None
    assert {issue.code for issue in assessment.issues} == {"UNCERTAINTY_UNAVAILABLE"}
    assert model.metadata.capabilities.supports_actions is False


def test_hgb_disables_internal_early_stopping() -> None:
    estimator = training._estimator(
        "hist_gradient_boosting",
        {
            "learning_rate": 0.1,
            "max_iter": 5,
            "max_leaf_nodes": 5,
            "l2_regularization": 0.0,
        },
        "baseline",
        42,
    )

    assert estimator.early_stopping is False


def test_training_keeps_source_specific_baseline_for_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(training, "RIDGE_ALPHAS", (1.0,))
    monkeypatch.setattr(
        training,
        "HGB_CONFIGS",
        (
            {
                "learning_rate": 0.1,
                "max_iter": 5,
                "max_leaf_nodes": 5,
                "min_samples_leaf": 2,
                "l2_regularization": 0.0,
            },
        ),
    )
    generic = _synthetic_dataset()
    feature_name = "ht:sulfur__pak_last"
    frame = generic.frame.copy()
    frame[feature_name] = frame["baseline"]
    dataset = replace(generic, frame=frame, feature_names=(feature_name, "x"))

    result = training.train_model(
        dataset,
        models_root=tmp_path,
        training_dataset_id="synthetic001",
        tag_dictionary_sha256="a" * 64,
        git_commit="test-commit",
        source_timezone="UTC",
    )

    assert result.bundle.metadata.baseline_feature == feature_name
    assert "baseline" not in result.bundle.metadata.feature_names
    prediction = result.bundle.predict(frame.loc[:1, list(result.bundle.feature_names)])
    assert np.isfinite(prediction).all()
