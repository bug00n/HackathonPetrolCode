"""Research-only high-capacity ensemble for the episode sulfur forecast.

This module intentionally lives beside, rather than inside, the frozen v2 path.
It reuses the leakage-safe episode construction and evaluates a richer model:
HGB + LightGBM, several random seeds, nonlinear process interactions and a
slightly relaxed alarm budget.  The relaxed numbers are useful for research
and a hackathon demo only; the artifact never enables real controls.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from source.data.prepare import PreparedData
from source.ml.artifacts import ModelBundle, feature_schema_hash, save_model
from source.ml.features import SupervisedDataset
from source.ml.safety import (
    PlattCalibrator,
    _fit_calibrator,
    fit_joint_applicability,
)
from source.ml.v2 import (
    HORIZONS,
    WINDOWS,
    EpisodeDataset,
    EpisodeFitResult,
    EpisodeSafetyPredictor,
    _bootstrap_event_fnr,
    _cap_training_rows,
    _estimator,
    _fit,
    _lead_time_report,
    _monotone_probabilities,
    _monthly_report,
    _regime_report,
    _upper_limit_metrics,
    build_episode_dataset,
    episode_sample_weights,
    event_metrics,
    rolling_month_folds,
    select_event_threshold,
)

# The strict production budget is 20%.  This branch deliberately exposes a
# small 2-point research relaxation, while storing both strict and relaxed
# reports and keeping ``promotion_eligible`` false.
EXPERIMENTAL_FALSE_ALARM_BUDGET = 0.22
EXPERIMENTAL_UPPER_QUANTILE = 0.975
EXPERIMENTAL_SEEDS = (42, 202)
EXPERIMENTAL_SIGNALS = ("ht:P8", "ht:F19", "ht:T11")
# Keep the CPU experiment bounded while retaining every positive event row when
# possible.  The temporal folds and purge rules are unchanged.
EXPERIMENTAL_MAX_ROLLING_TRAIN_ROWS = 5_000
EXPERIMENTAL_MAX_FINAL_TRAIN_ROWS = 5_000


@dataclass(frozen=True)
class MeanEstimator:
    """Average a small collection of compatible sklearn estimators."""

    estimators: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.estimators:
            raise ValueError("ensemble needs at least one estimator")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        values = [
            np.asarray(estimator.predict(features), dtype=float) for estimator in self.estimators
        ]
        return np.asarray(np.mean(np.stack(values, axis=0), axis=0), dtype=float)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        values = [
            np.asarray(estimator.predict_proba(features), dtype=float)
            for estimator in self.estimators
        ]
        return np.asarray(np.mean(np.stack(values, axis=0), axis=0), dtype=float)


def _safe_ratio(left: pd.Series, right: pd.Series) -> pd.Series:
    left_values = left.to_numpy(dtype=float)
    right_values = right.to_numpy(dtype=float)
    values = np.full(len(left_values), np.nan, dtype=float)
    valid = np.isfinite(left_values) & np.isfinite(right_values) & (np.abs(right_values) > 1e-9)
    values[valid] = left_values[valid] / right_values[valid]
    return pd.Series(values, index=left.index)


def add_experimental_interactions(dataset: EpisodeDataset) -> EpisodeDataset:
    """Add only causal, as-of interactions to an already leakage-safe frame."""
    frame = dataset.frame.copy()
    names = list(dataset.feature_names)

    def column(name: str) -> pd.Series:
        if name not in frame:
            raise ValueError(f"experimental interaction needs feature {name}")
        return pd.to_numeric(frame[name], errors="coerce")

    # The values are all measured at or before as_of.  Products and ratios add
    # nonlinear capacity without importing unverified tags or future labels.
    current = column(dataset.baseline_feature)
    p8 = column("ht:P8__lag_0m")
    f19 = column("ht:F19__lag_0m")
    t11 = column("ht:T11__lag_0m")
    p8_slope = column("pak_slope_60m")
    f19_delta = column("ht:F19__delta_60m")
    t11_delta = column("ht:T11__delta_60m")
    interactions: dict[str, pd.Series] = {
        "exp_sulfur_sq": current.pow(2),
        "exp_p8_sq": p8.pow(2),
        "exp_f19_sq": f19.pow(2),
        "exp_t11_sq": t11.pow(2),
        "exp_p8_x_f19": p8 * f19,
        "exp_p8_x_t11": p8 * t11,
        "exp_f19_x_t11": f19 * t11,
        "exp_sulfur_x_slope": current * p8_slope,
        "exp_sulfur_x_f19_delta": current * f19_delta,
        "exp_p8_to_f19": _safe_ratio(p8, f19),
        "exp_f19_to_t11": _safe_ratio(f19, t11),
        "exp_slope_x_f19_delta": p8_slope * f19_delta,
        "exp_slope_x_t11_delta": p8_slope * t11_delta,
    }
    for name, values in interactions.items():
        frame[name] = values.replace([np.inf, -np.inf], np.nan)
        names.append(name)
    return EpisodeDataset(frame, tuple(dict.fromkeys(names)), dataset.baseline_feature)


def build_experimental_episode_dataset(
    data: PreparedData, base: SupervisedDataset
) -> EpisodeDataset:
    """Build the v2 episode dataset plus the experimental interactions."""
    raw_signals = base.feature_definition.get("telemetry_signals", ())
    if not isinstance(raw_signals, (list, tuple)):
        raise ValueError("experimental dataset has invalid telemetry signal definition")
    signals = tuple(str(signal) for signal in raw_signals)
    if tuple(EXPERIMENTAL_SIGNALS) != signals:
        raise ValueError("experimental dataset requires the confirmed P8/F19/T11 signals")
    return add_experimental_interactions(build_episode_dataset(data, base))


def _experimental_estimator(family: str, task: str, seed: int) -> Any:
    """Use a deeper tree budget than v2 while keeping the same estimator families."""
    estimator = _estimator(family, task, seed)
    if family == "hgb":
        estimator.set_params(
            learning_rate=0.04,
            max_iter=80,
            max_leaf_nodes=31,
            l2_regularization=2.0,
        )
    else:
        if not isinstance(estimator, Pipeline):
            raise TypeError("lightgbm estimator must be a pipeline")
        estimator.set_params(
            model__n_estimators=180,
            model__learning_rate=0.03,
            model__num_leaves=31,
            model__min_child_samples=40,
            model__reg_lambda=1.5,
        )
    return estimator


def _ensemble_rolling_predictions(
    dataset: EpisodeDataset,
    start: str,
    end: str,
    seed: int,
    *,
    include_regression: bool = True,
    include_upper: bool = True,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Return out-of-fold means of HGB and LightGBM on identical month folds."""
    frames: pd.DataFrame | None = None
    risk: dict[int, list[np.ndarray]] = {horizon: [] for horizon in HORIZONS}
    deltas: list[np.ndarray] = []
    uppers: list[np.ndarray] = []
    for family in ("hgb", "lightgbm"):
        current, raw, delta, upper = _rolling_predictions_for_family(
            dataset,
            family,
            start,
            end,
            seed,
            include_regression=include_regression,
            include_upper=include_upper,
        )
        if frames is None:
            frames = current
        elif not frames["as_of"].equals(current["as_of"]):
            raise ValueError("ensemble families produced different rolling folds")
        for horizon in HORIZONS:
            risk[horizon].append(raw[horizon])
        if include_regression:
            deltas.append(delta)
        if include_upper:
            uppers.append(upper)
    if frames is None:
        raise ValueError("ensemble rolling backtest produced no folds")
    return (
        frames,
        {horizon: np.mean(np.stack(values, axis=0), axis=0) for horizon, values in risk.items()},
        np.mean(np.stack(deltas, axis=0), axis=0) if deltas else np.asarray([], dtype=float),
        np.mean(np.stack(uppers, axis=0), axis=0) if uppers else np.asarray([], dtype=float),
    )


def _rolling_predictions_for_family(
    dataset: EpisodeDataset,
    family: str,
    start: str,
    end: str,
    seed: int,
    *,
    include_regression: bool,
    include_upper: bool,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Same rolling-origin protocol as v2, with the deeper estimator factory."""
    frame = dataset.frame
    pieces: list[pd.DataFrame] = []
    risk_parts: dict[int, list[np.ndarray]] = {horizon: [] for horizon in HORIZONS}
    delta_parts: list[np.ndarray] = []
    upper_parts: list[np.ndarray] = []
    features = list(dataset.feature_names)
    for fold_number, (train_indices, validation_indices) in enumerate(
        rolling_month_folds(frame, start, end)
    ):
        train_indices = _cap_training_rows(
            frame,
            train_indices,
            EXPERIMENTAL_MAX_ROLLING_TRAIN_ROWS,
            seed=seed,
            fold_number=fold_number,
        )
        train = frame.iloc[train_indices]
        validation = frame.iloc[validation_indices]
        weights = episode_sample_weights(train)
        for horizon in HORIZONS:
            estimator = _fit(
                _experimental_estimator(family, "classifier", seed),
                train.loc[:, features],
                train[f"crossing_{horizon}m"].to_numpy(dtype=bool),
                weights,
                family,
            )
            risk_parts[horizon].append(
                np.asarray(estimator.predict_proba(validation.loc[:, features])[:, 1], dtype=float)
            )
        if include_regression:
            estimator = _fit(
                _experimental_estimator(family, "delta", seed),
                train.loc[:, features],
                train["delta_60m"].to_numpy(dtype=float),
                weights,
                family,
            )
            delta_parts.append(
                np.asarray(estimator.predict(validation.loc[:, features]), dtype=float)
            )
        if include_upper:
            estimator = _fit(
                _experimental_estimator(family, "upper", seed),
                train.loc[:, features],
                train["delta_60m"].to_numpy(dtype=float),
                weights,
                family,
            )
            upper_parts.append(
                np.asarray(estimator.predict(validation.loc[:, features]), dtype=float)
            )
        pieces.append(validation)
    if not pieces:
        raise ValueError("rolling backtest produced no folds")
    return (
        pd.concat(pieces, ignore_index=True),
        {horizon: np.concatenate(parts) for horizon, parts in risk_parts.items()},
        np.concatenate(delta_parts) if delta_parts else np.asarray([], dtype=float),
        np.concatenate(upper_parts) if upper_parts else np.asarray([], dtype=float),
    )


def _fit_final_ensemble(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    task: str,
    target: np.ndarray,
    weights: np.ndarray,
) -> MeanEstimator:
    estimators: list[Any] = []
    for family in ("hgb", "lightgbm"):
        for seed in EXPERIMENTAL_SEEDS:
            estimators.append(
                _fit(
                    _experimental_estimator(family, task, seed),
                    frame.loc[:, list(features)],
                    target,
                    weights,
                    family,
                )
            )
    return MeanEstimator(tuple(estimators))


def _calibration_shift(
    actual: np.ndarray,
    baseline: np.ndarray,
    upper_delta: np.ndarray,
    quantile: float,
) -> float:
    finite = np.isfinite(actual) & np.isfinite(baseline) & np.isfinite(upper_delta)
    if not finite.any():
        return 0.0
    residual = actual[finite] - baseline[finite] - upper_delta[finite]
    return max(0.0, float(np.quantile(residual, quantile)))


def fit_experimental_episode_model(
    dataset: EpisodeDataset,
    *,
    seed: int = 42,
    relaxed_false_alarm_budget: float = EXPERIMENTAL_FALSE_ALARM_BUDGET,
) -> EpisodeFitResult:
    """Fit and evaluate the richer research-only ensemble."""
    if not 0.20 < relaxed_false_alarm_budget < 0.5:
        raise ValueError("experimental budget must be between 0.20 and 0.50")
    selection_frame, selection_raw, selection_delta, _ = _ensemble_rolling_predictions(
        dataset, "2024-01-01", "2025-01-01", seed, include_upper=False
    )
    selection_probabilities = _monotone_probabilities(selection_raw)[60]
    _strict_policy, strict_selection = select_event_threshold(
        selection_frame, selection_probabilities, 0.20
    )
    relaxed_policy, relaxed_selection = select_event_threshold(
        selection_frame, selection_probabilities, relaxed_false_alarm_budget
    )
    selection_point = selection_frame["baseline"].to_numpy(dtype=float) + selection_delta
    selection_actual = selection_frame["y_60m"].to_numpy(dtype=float)

    calibration_frame, calibration_raw, _, calibration_upper = _ensemble_rolling_predictions(
        dataset, "2025-01-01", "2025-07-01", seed, include_regression=False, include_upper=True
    )
    calibrators: dict[int, PlattCalibrator] = {}
    for horizon in HORIZONS:
        calibrators[horizon] = _fit_calibrator(
            calibration_raw[horizon],
            calibration_frame[f"crossing_{horizon}m"].to_numpy(dtype=bool),
            seed,
        )
    calibration_actual = calibration_frame["y_60m"].to_numpy(dtype=float)
    calibration_baseline = calibration_frame["baseline"].to_numpy(dtype=float)
    strict_upper_shift = _calibration_shift(
        calibration_actual, calibration_baseline, calibration_upper, 0.95
    )
    relaxed_upper_shift = _calibration_shift(
        calibration_actual, calibration_baseline, calibration_upper, EXPERIMENTAL_UPPER_QUANTILE
    )

    policy_frame, policy_raw, _, _ = _ensemble_rolling_predictions(
        dataset, "2025-07-01", "2026-01-01", seed, include_regression=False, include_upper=False
    )
    policy_probabilities = _monotone_probabilities(
        {horizon: calibrators[horizon].predict(policy_raw[horizon]) for horizon in HORIZONS}
    )
    strict_threshold_policy, strict_threshold_metrics = select_event_threshold(
        policy_frame, policy_probabilities[60], 0.20
    )
    relaxed_threshold_policy, relaxed_threshold_metrics = select_event_threshold(
        policy_frame, policy_probabilities[60], relaxed_false_alarm_budget
    )

    frame = dataset.frame
    as_of = pd.to_datetime(frame["as_of"], utc=True)
    available = pd.to_datetime(frame["target_available_at"], utc=True)
    fit_mask = (as_of < pd.Timestamp("2026-01-01", tz="UTC")) & (
        available < pd.Timestamp("2026-01-01", tz="UTC")
    )
    fit_indices = np.flatnonzero(fit_mask.to_numpy())
    fit_indices = _cap_training_rows(
        frame,
        fit_indices,
        EXPERIMENTAL_MAX_FINAL_TRAIN_ROWS,
        seed=seed,
        fold_number=99,
    )
    fit_frame = frame.iloc[fit_indices]
    features = tuple(dataset.feature_names)
    weights = episode_sample_weights(fit_frame)
    risk_estimators = {
        horizon: _fit_final_ensemble(
            fit_frame,
            features,
            task="classifier",
            target=fit_frame[f"crossing_{horizon}m"].to_numpy(dtype=bool),
            weights=weights,
        )
        for horizon in HORIZONS
    }
    delta_estimator = _fit_final_ensemble(
        fit_frame,
        features,
        task="delta",
        target=fit_frame["delta_60m"].to_numpy(dtype=float),
        weights=weights,
    )
    upper_estimator = _fit_final_ensemble(
        fit_frame,
        features,
        task="upper",
        target=fit_frame["delta_60m"].to_numpy(dtype=float),
        weights=weights,
    )
    applicability = fit_joint_applicability(
        fit_frame.loc[:, list(features)], required_features=(dataset.baseline_feature,)
    )
    predictor = EpisodeSafetyPredictor(
        dataset.baseline_feature,
        delta_estimator,
        upper_estimator,
        risk_estimators,
        calibrators,
        relaxed_threshold_policy,
        applicability,
        relaxed_upper_shift,
    )

    audit = frame.loc[as_of >= pd.Timestamp("2026-01-01", tz="UTC")].reset_index(drop=True)
    audit_features = audit.loc[:, list(features)]
    audit_probabilities = predictor.predict_horizon_probabilities(audit_features)
    audit_probability = audit_probabilities[60]
    audit_point = predictor.predict(audit_features)
    audit_upper = predictor.predict_upper(audit_features)
    audit_actual = audit["y_60m"].to_numpy(dtype=float)
    finite = np.isfinite(audit_actual) & np.isfinite(audit_point) & np.isfinite(audit_upper)
    audit_metrics = event_metrics(audit, audit_probability, relaxed_threshold_policy.threshold)
    audit_metrics.update(
        {
            "forecast_rows": int(finite.sum()),
            "mae": float(np.mean(np.abs(audit_actual[finite] - audit_point[finite])))
            if finite.any()
            else None,
            "upper_coverage": float(np.mean(audit_actual[finite] <= audit_upper[finite]))
            if finite.any()
            else None,
            "event_fnr_bootstrap_95": _bootstrap_event_fnr(
                audit, audit_probability, relaxed_threshold_policy.threshold, seed
            ),
        }
    )
    audit_metrics.update(_upper_limit_metrics(audit_actual, audit_upper))
    strict_audit_metrics = event_metrics(
        audit, audit_probability, strict_threshold_policy.threshold
    )
    strict_audit_metrics["upper_coverage"] = (
        float(np.mean(audit_actual[finite] <= audit_upper[finite])) if finite.any() else None
    )
    monthly = _monthly_report(
        audit,
        audit_probability,
        relaxed_threshold_policy.threshold,
        actual=audit_actual,
        upper=audit_upper,
    )
    valid_months = [
        item for item in monthly.values() if item["event_false_negative_rate"] is not None
    ]
    strict_gate = (
        strict_threshold_metrics["event_false_negative_rate"] <= 0.10
        and strict_threshold_metrics["event_false_positive_rate"] <= 0.20
        and strict_threshold_metrics["row_false_positive_rate"] <= 0.20
        and audit_metrics["upper_coverage"] >= 0.95
        and audit_metrics["upper_limit_miss_rate"] is not None
        and audit_metrics["upper_limit_miss_rate"] <= 0.05
    )
    report: dict[str, Any] = {
        "schema_version": "1.2",
        "variant": "experimental_complex_ensemble",
        "selected_family": "hgb+lightgbm_mean_ensemble",
        "ensemble_members": ["hgb", "lightgbm"],
        "ensemble_seeds": list(EXPERIMENTAL_SEEDS),
        "training_rows": int(len(fit_frame)),
        "rolling_train_row_cap": EXPERIMENTAL_MAX_ROLLING_TRAIN_ROWS,
        "final_train_row_cap": EXPERIMENTAL_MAX_FINAL_TRAIN_ROWS,
        "relaxations": {
            "false_alarm_budget": relaxed_false_alarm_budget,
            "upper_residual_quantile": EXPERIMENTAL_UPPER_QUANTILE,
            "purpose": "research_only; strict production gates remain unchanged",
        },
        "selection_2024": {
            "point_mae": float(np.mean(np.abs(selection_actual - selection_point))),
            "strict": strict_selection,
            "relaxed": relaxed_selection,
        },
        "calibration_period": "2025-01-01/2025-07-01",
        "threshold_period": "2025-07-01/2026-01-01",
        "threshold_strict": {
            "alarm_threshold": strict_threshold_policy.threshold,
            "metrics": strict_threshold_metrics,
        },
        "threshold_relaxed": {
            "alarm_threshold": relaxed_threshold_policy.threshold,
            "metrics": relaxed_threshold_metrics,
        },
        "alarm_threshold": relaxed_threshold_policy.threshold,
        "false_alarm_budget": relaxed_false_alarm_budget,
        "upper_delta_shift_strict_95": strict_upper_shift,
        "upper_delta_shift_experimental": relaxed_upper_shift,
        "audit_2026": audit_metrics,
        "audit_2026_strict_threshold": strict_audit_metrics,
        "audit_monthly": monthly,
        "audit_worst_month_event_fnr": max(
            float(item["event_false_negative_rate"]) for item in valid_months
        )
        if valid_months
        else None,
        "audit_regimes": _regime_report(
            audit, audit_probability, relaxed_threshold_policy.threshold
        ),
        "audit_lead_times": _lead_time_report(
            audit, audit_probabilities, relaxed_threshold_policy.threshold
        ),
        "upper_calibration": {
            "strict_95_shift": strict_upper_shift,
            "experimental_quantile": EXPERIMENTAL_UPPER_QUANTILE,
        },
        "promotion_eligible": False,
        "strict_gate_passed": bool(strict_gate),
        "supports_actions": False,
        "test_used_for_selection": False,
        "production_status": "experimental_shadow_only",
        "promotion_blockers": [
            "experimental relaxation exceeds the production false-alarm budget",
            "audit metrics are diagnostic and cannot authorize production",
            "action effects remain observational and supports_actions=false",
        ],
    }
    return EpisodeFitResult(predictor, report)


def save_experimental_episode_model(
    directory: Path,
    data: PreparedData,
    base: SupervisedDataset,
    *,
    git_commit: str,
    seed: int = 42,
    relaxed_false_alarm_budget: float = EXPERIMENTAL_FALSE_ALARM_BUDGET,
) -> tuple[ModelBundle, EpisodeFitResult]:
    """Persist the experimental artifact with an explicit non-production label."""
    dataset = build_experimental_episode_dataset(data, base)
    fitted = fit_experimental_episode_model(
        dataset, seed=seed, relaxed_false_alarm_budget=relaxed_false_alarm_budget
    )
    definition = {
        "horizon_minutes": 60,
        "horizons_minutes": list(HORIZONS),
        "windows_minutes": list(WINDOWS),
        "target": "delta_60m and any crossing within horizon",
        "episode_weighting": "inverse episode length and equal month mass",
        "telemetry_signals": list(EXPERIMENTAL_SIGNALS),
        "interaction_features": [name for name in dataset.feature_names if name.startswith("exp_")],
        "telemetry_semantics_version": "organizer-qa-2026-09-15",
        "lims_availability_rule": "available_at <= as_of; measured_at is never shifted",
        "experimental_relaxations": dict(fitted.report["relaxations"]),
    }
    metadata = {
        "model_id": directory.name,
        "schema_version": "1.2",
        "model_type": "experimental_complex_ensemble+episode_multi_horizon",
        "training_dataset_id": data.manifest.dataset_id,
        "git_commit": git_commit,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "target_signal": base.target_signal_id,
        "target_source": "pak",
        "target_unit": base.target_unit,
        "horizon_minutes": 60,
        "feature_names": dataset.feature_names,
        "baseline_feature": dataset.baseline_feature,
        "tag_dictionary_sha256": data.manifest.tag_dictionary_sha256,
        "feature_schema_hash": feature_schema_hash(dataset.feature_names, definition),
        "processing": {
            "feature_definition": definition,
            "selection_period": "2024",
            "calibration_period": fitted.report["calibration_period"],
            "threshold_period": fitted.report["threshold_period"],
            "alarm_threshold": fitted.report["alarm_threshold"],
            "production_status": "experimental_shadow_only",
            "experimental_variant": True,
            "telemetry_cleaning": "config/telemetry_rules.json:ht:Q21==307->missing",
            "telemetry_rules_sha256": getattr(data.manifest, "telemetry_rules_sha256", None),
        },
        "time_boundaries": {
            "train_end_local": "2026-01-01",
            "validation_end_local": "2026-01-01",
            "source_timezone": data.manifest.source_timezone,
            "calibration_start": "2025-01-01",
            "calibration_end": "2025-07-01",
        },
        "seed": seed,
        "alarm_threshold": fitted.report["alarm_threshold"],
        "false_alarm_budget": relaxed_false_alarm_budget,
        "calibration": {
            "method": "platt",
            "period": fitted.report["calibration_period"],
            "threshold_period": fitted.report["threshold_period"],
        },
        "metrics": {"pak": fitted.report, "lims": "separate_delayed_control_layer"},
        "capabilities": {
            "supports_forecast": True,
            "supports_actions": False,
            "supports_uncertainty": True,
            "supports_exceedance_probability": True,
            "supports_multi_horizon": True,
        },
        "applicability": {
            "method": "joint_pca_mahalanobis",
            "required_features": [dataset.baseline_feature],
            "coverage": 0.99,
            "max_distance_squared": fitted.predictor.applicability.max_distance_squared,
            "purpose": "experimental shadow forecast; out-of-domain is unavailable",
            "action_comparison": "forbidden",
        },
        "ood": {
            "method": "joint_pca_mahalanobis",
            "coverage": 0.99,
            "max_distance_squared": fitted.predictor.applicability.max_distance_squared,
            "out_of_domain_behavior": "unavailable",
        },
        "reports": ("metrics.json",),
    }
    return save_model(directory, fitted.predictor, metadata, fitted.report), fitted


__all__ = [
    "EXPERIMENTAL_FALSE_ALARM_BUDGET",
    "EXPERIMENTAL_MAX_FINAL_TRAIN_ROWS",
    "EXPERIMENTAL_MAX_ROLLING_TRAIN_ROWS",
    "EXPERIMENTAL_SEEDS",
    "EXPERIMENTAL_SIGNALS",
    "MeanEstimator",
    "add_experimental_interactions",
    "build_experimental_episode_dataset",
    "fit_experimental_episode_model",
    "save_experimental_episode_model",
]
