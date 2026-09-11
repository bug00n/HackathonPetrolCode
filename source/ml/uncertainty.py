"""Stage-5 uncertainty calibration and applicability checks.

The functions are deliberately independent from serving contracts: a caller
must explicitly turn an unavailable result into ``unknown`` rather than
silently treating it as a passing quality check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from source.ml.artifacts import ModelBundle, save_model
from source.ml.evaluate import make_outer_split


@dataclass(frozen=True)
class UpperBoundCalibration:
    """Selected raw quantile model and its validation-only correction."""

    model_id: str
    interval_level: float
    shift: float
    selection_size: int
    calibration_size: int

    def apply(self, raw_upper: Sequence[float] | np.ndarray) -> np.ndarray:
        """Apply the non-negative additive calibration shift."""
        values = _finite_vector(raw_upper, "raw_upper")
        return values + self.shift


@dataclass(frozen=True)
class UpperBoundMetrics:
    """Held-out quality of an upper estimate."""

    coverage: float
    average_width: float
    count: int


@dataclass(frozen=True)
class ApplicabilityResult:
    """Availability result which cannot be mistaken for a successful check."""

    available: bool
    reason_code: str | None
    violations: tuple[str, ...] = ()


@dataclass(frozen=True)
class RobustnessCaseResult:
    """Evaluation result for one deterministic stress case."""

    name: str
    available: bool
    metrics: UpperBoundMetrics | None
    reason_code: str | None


class UncertaintyDataset(Protocol):
    frame: pd.DataFrame
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class CalibratedPointUpperRegressor:
    """Persistable point predictor plus a calibrated quantile regressor."""

    point_predictor: Any
    upper_predictor: Any
    shift: float

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.point_predictor.predict(features), dtype=float)

    def predict_upper(self, features: pd.DataFrame) -> np.ndarray:
        point = self.predict(features)
        raw = np.asarray(self.upper_predictor.predict(features), dtype=float)
        return np.maximum(point, raw + self.shift)


@dataclass(frozen=True)
class Stage5FitResult:
    predictor: CalibratedPointUpperRegressor
    calibration: UpperBoundCalibration
    feature_bounds: dict[str, tuple[float, float]]
    report: dict[str, Any]


def _finite_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not len(vector):
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def pinball_loss(
    y_true: Sequence[float] | np.ndarray,
    prediction: Sequence[float] | np.ndarray,
    quantile: float,
) -> float:
    """Return mean quantile loss, used only on the selection half."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between zero and one")
    actual = _finite_vector(y_true, "y_true")
    predicted = _finite_vector(prediction, "prediction")
    if len(actual) != len(predicted):
        raise ValueError("y_true and prediction lengths differ")
    residual = actual - predicted
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))


def fit_upper_calibrator(
    y_validation: Sequence[float] | np.ndarray,
    raw_upper_by_model: Mapping[str, Sequence[float] | np.ndarray],
    *,
    interval_level: float = 0.95,
) -> UpperBoundCalibration:
    """Select on the first validation half and calibrate on the second.

    Test observations are intentionally absent from this API.  Models tie-break
    by identifier, making the outcome stable across mapping insertion order.
    The correction follows DESIGN.md exactly:
    ``max(0, quantile_0.95(y - upper_raw))``.
    """
    actual = _finite_vector(y_validation, "y_validation")
    if len(actual) < 4:
        raise ValueError("validation needs at least four chronological observations")
    if not raw_upper_by_model:
        raise ValueError("at least one raw upper model is required")
    split = len(actual) // 2
    predictions: dict[str, np.ndarray] = {}
    for model_id, values in raw_upper_by_model.items():
        vector = _finite_vector(values, f"raw_upper_by_model[{model_id!r}]")
        if len(vector) != len(actual):
            raise ValueError(f"raw upper length differs for model {model_id!r}")
        predictions[model_id] = vector

    selected = min(
        predictions,
        key=lambda model_id: (
            pinball_loss(actual[:split], predictions[model_id][:split], interval_level),
            model_id,
        ),
    )
    residuals = actual[split:] - predictions[selected][split:]
    shift = max(0.0, float(np.quantile(residuals, interval_level)))
    return UpperBoundCalibration(
        model_id=selected,
        interval_level=interval_level,
        shift=shift,
        selection_size=split,
        calibration_size=len(actual) - split,
    )


def evaluate_upper_bounds(
    y_true: Sequence[float] | np.ndarray,
    point_prediction: Sequence[float] | np.ndarray,
    upper_prediction: Sequence[float] | np.ndarray,
) -> UpperBoundMetrics:
    """Measure held-out coverage and mean upper-minus-point width."""
    actual = _finite_vector(y_true, "y_true")
    point = _finite_vector(point_prediction, "point_prediction")
    upper = _finite_vector(upper_prediction, "upper_prediction")
    if len({len(actual), len(point), len(upper)}) != 1:
        raise ValueError("held-out vectors must have equal lengths")
    if np.any(upper < point):
        raise ValueError("upper_prediction cannot be below point_prediction")
    return UpperBoundMetrics(
        coverage=float(np.mean(actual <= upper)),
        average_width=float(np.mean(upper - point)),
        count=len(actual),
    )


def check_applicability(
    features: Mapping[str, float | None],
    bounds: Mapping[str, tuple[float, float]],
) -> ApplicabilityResult:
    """Reject missing, invalid and out-of-training-domain feature values."""
    violations: list[str] = []
    missing: list[str] = []
    for name, (lower, upper) in bounds.items():
        if not np.isfinite((lower, upper)).all() or lower > upper:
            raise ValueError(f"invalid applicability bounds for {name!r}")
        value = features.get(name)
        if value is None or not np.isfinite(value):
            missing.append(name)
        elif value < lower or value > upper:
            violations.append(name)
    if missing:
        return ApplicabilityResult(False, "FEATURES_UNAVAILABLE", tuple(sorted(missing)))
    if violations:
        return ApplicabilityResult(False, "OUT_OF_DOMAIN", tuple(sorted(violations)))
    return ApplicabilityResult(True, None)


def evaluate_robustness_cases(
    cases: Mapping[str, tuple[Sequence[float], Sequence[float], Sequence[float]] | None],
) -> tuple[RobustnessCaseResult, ...]:
    """Evaluate nominal/error/shift cases without converting missing data to zero."""
    results: list[RobustnessCaseResult] = []
    for name in sorted(cases):
        values = cases[name]
        if values is None:
            results.append(RobustnessCaseResult(name, False, None, "UNCERTAINTY_UNAVAILABLE"))
            continue
        try:
            metrics = evaluate_upper_bounds(*values)
        except ValueError:
            results.append(RobustnessCaseResult(name, False, None, "UNCERTAINTY_UNAVAILABLE"))
        else:
            results.append(RobustnessCaseResult(name, True, metrics, None))
    return tuple(results)


def fit_stage5_uncertainty(
    dataset: UncertaintyDataset,
    point_model: ModelBundle,
    *,
    source_timezone: str = "Europe/Moscow",
    train_end: str = "2025-01-01",
    validation_end: str = "2026-01-01",
    interval_level: float = 0.95,
    seed: int = 42,
) -> Stage5FitResult:
    """Fit/select/calibrate a quantile model and evaluate once on held-out test."""
    boundaries = point_model.metadata.time_boundaries
    stored_timezone = boundaries.get("source_timezone")
    stored_train_end = boundaries.get("train_end_local")
    stored_validation_end = boundaries.get("validation_end_local")
    if (source_timezone, train_end, validation_end) != (
        stored_timezone,
        stored_train_end,
        stored_validation_end,
    ):
        raise ValueError("uncertainty split settings differ from the point artifact")
    frame = dataset.frame.reset_index(drop=True)
    if tuple(dataset.feature_names) != point_model.feature_names:
        raise ValueError("uncertainty dataset features differ from the point artifact")
    required = {"y", "as_of", *point_model.feature_names}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"uncertainty dataset is missing columns: {missing}")
    split = make_outer_split(
        frame,
        source_timezone=source_timezone,
        train_end=train_end,
        validation_end=validation_end,
    )
    if min(len(split.train), len(split.validation), len(split.test)) == 0:
        raise ValueError("train, validation and test must all be non-empty")
    feature_names = list(point_model.feature_names)
    train = frame.iloc[split.train]
    validation = frame.iloc[split.validation]
    test = frame.iloc[split.test]
    x_train = train.loc[:, feature_names]
    y_train = pd.to_numeric(train["y"], errors="coerce").to_numpy(dtype=float)
    train_valid = np.isfinite(y_train)
    x_train = x_train.loc[train_valid]
    y_train = y_train[train_valid]
    if len(y_train) < 20:
        raise ValueError("uncertainty training needs at least 20 finite targets")

    parameter_grid = (
        {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 20},
        {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 20},
    )
    estimators: dict[str, HistGradientBoostingRegressor] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    validation_target = pd.to_numeric(validation["y"], errors="coerce").to_numpy(dtype=float)
    validation_valid = np.isfinite(validation_target)
    if validation_valid.sum() < 4:
        raise ValueError("validation needs at least four finite targets")
    validation_features = validation.loc[validation_valid, feature_names]
    validation_target = validation_target[validation_valid]
    for index, parameters in enumerate(parameter_grid):
        model_id = f"hgb_quantile_{index}"
        estimator = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=interval_level,
            random_state=seed,
            early_stopping=False,
            **parameters,
        ).fit(x_train, y_train)
        estimators[model_id] = estimator
        validation_predictions[model_id] = np.asarray(
            estimator.predict(validation_features), dtype=float
        )
    calibration = fit_upper_calibrator(
        validation_target,
        validation_predictions,
        interval_level=interval_level,
    )
    predictor = CalibratedPointUpperRegressor(
        point_predictor=point_model.predictor,
        upper_predictor=estimators[calibration.model_id],
        shift=calibration.shift,
    )

    feature_bounds: dict[str, tuple[float, float]] = {}
    for name in feature_names:
        values = pd.to_numeric(train[name], errors="coerce").dropna().to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            raise ValueError(f"training feature {name!r} has no finite values")
        feature_bounds[name] = (
            float(np.quantile(values, 0.001)),
            float(np.quantile(values, 0.999)),
        )

    test_target = pd.to_numeric(test["y"], errors="coerce").to_numpy(dtype=float)
    test_baseline = pd.to_numeric(
        test[point_model.metadata.baseline_feature], errors="coerce"
    ).to_numpy(dtype=float)
    test_valid = np.isfinite(test_target) & np.isfinite(test_baseline)
    test_features = test.loc[test_valid, feature_names]
    test_target = test_target[test_valid]
    point = predictor.predict(test_features)
    upper = predictor.predict_upper(test_features)
    metrics = evaluate_upper_bounds(test_target, point, upper)
    in_domain = np.ones(len(test_features), dtype=bool)
    for name, (lower, upper_bound) in feature_bounds.items():
        values = pd.to_numeric(test_features[name], errors="coerce").to_numpy(dtype=float)
        in_domain &= np.isfinite(values) & (values >= lower) & (values <= upper_bound)
    in_domain_metrics = (
        evaluate_upper_bounds(test_target[in_domain], point[in_domain], upper[in_domain])
        if in_domain.any()
        else None
    )
    tail_start = max(0, len(test_target) * 3 // 4)
    tail_metrics = evaluate_upper_bounds(
        test_target[tail_start:], point[tail_start:], upper[tail_start:]
    )
    error_metrics = evaluate_upper_bounds(test_target + 0.5, point, upper)
    validation_times = pd.to_datetime(validation.loc[validation_valid, "as_of"], utc=True)
    calibration_start = calibration.selection_size
    report = {
        "schema_version": "1.0",
        "interval_level": interval_level,
        "selected_model": calibration.model_id,
        "calibration_shift": calibration.shift,
        "selection_rows": calibration.selection_size,
        "calibration_rows": calibration.calibration_size,
        "selection_end": validation_times.iloc[calibration_start - 1].isoformat(),
        "calibration_start": validation_times.iloc[calibration_start].isoformat(),
        "test_rows": metrics.count,
        "test_coverage": metrics.coverage,
        "test_average_width": metrics.average_width,
        "test_applicability_rate": float(np.mean(in_domain)),
        "test_in_domain_coverage": (
            in_domain_metrics.coverage if in_domain_metrics is not None else None
        ),
        "test_in_domain_average_width": (
            in_domain_metrics.average_width if in_domain_metrics is not None else None
        ),
        "test_used_for_tuning": False,
        "coverage_note": "Nominal 0.95 is an empirical target, not a safety guarantee.",
        "robustness": {
            "measurement_error_plus_0_5_mg_kg": {
                "coverage": error_metrics.coverage,
                "average_width": error_metrics.average_width,
            },
            "regime_shift_last_quarter": {
                "coverage": tail_metrics.coverage,
                "average_width": tail_metrics.average_width,
                "rows": tail_metrics.count,
            },
            "lims_delay": "inherited 4h publication delay; PAK target does not use future LIMS",
        },
    }
    return Stage5FitResult(predictor, calibration, feature_bounds, report)


def save_stage5_model(
    directory: Path,
    dataset: UncertaintyDataset,
    point_model: ModelBundle,
    *,
    source_timezone: str = "Europe/Moscow",
    train_end: str = "2025-01-01",
    validation_end: str = "2026-01-01",
    seed: int = 42,
) -> tuple[ModelBundle, Stage5FitResult]:
    """Fit and save an uncertainty-capable artifact using existing strict metadata."""
    fitted = fit_stage5_uncertainty(
        dataset,
        point_model,
        source_timezone=source_timezone,
        train_end=train_end,
        validation_end=validation_end,
        seed=seed,
    )
    metadata = point_model.metadata.model_dump(mode="python")
    metadata.update(
        {
            "model_id": directory.name,
            "model_type": f"{point_model.metadata.model_type}+hgb_quantile",
            "capabilities": {
                "supports_forecast": True,
                "supports_actions": False,
                "supports_uncertainty": True,
            },
            "applicability": {
                **point_model.metadata.applicability,
                "uncertainty_interval": "empirical_upper_0.95",
                "feature_bounds": {
                    name: list(bounds) for name, bounds in fitted.feature_bounds.items()
                },
                "out_of_domain_behavior": "unavailable",
            },
            "processing": {
                **point_model.metadata.processing,
                "uncertainty": {
                    "loss": "quantile",
                    "level": fitted.calibration.interval_level,
                    "selection_validation_half": "first",
                    "calibration_validation_half": "second",
                    "calibration_shift": fitted.calibration.shift,
                },
            },
            "reports": tuple(point_model.metadata.reports)
            + (f"artifacts/models/{directory.name}/metrics.json",),
        }
    )
    bundle = save_model(directory, fitted.predictor, metadata, fitted.report)
    return bundle, fitted


__all__ = [
    "ApplicabilityResult",
    "CalibratedPointUpperRegressor",
    "RobustnessCaseResult",
    "UpperBoundCalibration",
    "UpperBoundMetrics",
    "check_applicability",
    "evaluate_robustness_cases",
    "evaluate_upper_bounds",
    "fit_upper_calibrator",
    "fit_stage5_uncertainty",
    "pinball_loss",
    "save_stage5_model",
    "Stage5FitResult",
]
