"""Offline stage-2 model comparison and artifact creation."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from source.ml.artifacts import (
    LastValueRegressor,
    ModelBundle,
    feature_schema_hash,
    save_model,
)
from source.ml.evaluate import (
    expanding_purged_folds,
    make_outer_split,
    regression_metrics,
    validate_supervised_horizon,
)

BASELINE_FEATURE = "baseline"
MODEL_COMPLEXITY = {"last_value": 0, "ridge": 1, "hist_gradient_boosting": 2}
RIDGE_ALPHAS = (0.1, 1.0, 10.0)
HGB_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "learning_rate": 0.05,
        "max_iter": 100,
        "max_leaf_nodes": 15,
        "l2_regularization": 1.0,
    },
    {
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
    },
)


class TrainingDatasetLike(Protocol):
    """Structural contract supplied by ``ml.features.SupervisedDataset``."""

    @property
    def frame(self) -> pd.DataFrame: ...

    @property
    def feature_names(self) -> tuple[str, ...]: ...

    @property
    def target_signal_id(self) -> str: ...

    @property
    def target_unit(self) -> str: ...

    @property
    def target_source(self) -> object: ...


@dataclass(frozen=True)
class TrainingResult:
    """The selected runtime model and its persisted validation evidence."""

    bundle: ModelBundle
    artifact_dir: Path
    metrics: dict[str, Any]


def train_model(
    dataset: TrainingDatasetLike,
    *,
    models_root: Path,
    training_dataset_id: str,
    tag_dictionary_sha256: str,
    git_commit: str,
    source_timezone: str = "Europe/Moscow",
    horizon_minutes: int = 60,
    seed: int = 42,
    train_end: str = "2025-01-01",
    validation_end: str = "2026-01-01",
) -> TrainingResult:
    """Tune on train, select on validation, then refit without reading test labels."""
    if horizon_minutes != 60:
        raise ValueError("stage 2 supports the declared 60-minute horizon only")
    frame = dataset.frame.copy().reset_index(drop=True)
    validate_supervised_horizon(frame, horizon_minutes)
    feature_definition = _feature_definition(dataset)
    definition_horizon = feature_definition.get("horizon_minutes")
    if definition_horizon is not None and definition_horizon != horizon_minutes:
        raise ValueError("feature recipe horizon does not match the training horizon")
    declared_features = tuple(str(name) for name in dataset.feature_names)
    feature_names = _model_feature_names(frame, declared_features)
    baseline_feature = _baseline_feature_name(dataset, frame, feature_names)
    required = {
        "as_of",
        "target_at",
        "target_available_at",
        "y",
        baseline_feature,
        *feature_names,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"supervised frame is missing columns: {missing}")

    split = make_outer_split(
        frame,
        source_timezone=source_timezone,
        train_end=train_end,
        validation_end=validation_end,
    )
    if len(split.train) == 0 or len(split.validation) == 0:
        raise ValueError("train and validation splits must both contain rows")
    train_frame = frame.iloc[split.train].reset_index(drop=True)
    validation_frame = frame.iloc[split.validation].reset_index(drop=True)
    feature_names = _drop_all_missing_train_features(train_frame, feature_names, baseline_feature)

    ridge_params, ridge_inner = _tune_family(
        "ridge", train_frame, feature_names, baseline_feature, RIDGE_ALPHAS, seed
    )
    hgb_params, hgb_inner = _tune_family(
        "hist_gradient_boosting",
        train_frame,
        feature_names,
        baseline_feature,
        HGB_CONFIGS,
        seed,
    )

    fitted = {
        "last_value": _fit_estimator(
            "last_value", {}, train_frame, feature_names, baseline_feature, seed
        ),
        "ridge": _fit_estimator(
            "ridge", ridge_params, train_frame, feature_names, baseline_feature, seed
        ),
        "hist_gradient_boosting": _fit_estimator(
            "hist_gradient_boosting",
            hgb_params,
            train_frame,
            feature_names,
            baseline_feature,
            seed,
        ),
    }
    comparison = _validation_comparison(fitted, validation_frame, feature_names, baseline_feature)
    selected_name = select_model(comparison)
    selected_params = {
        "last_value": {},
        "ridge": ridge_params,
        "hist_gradient_boosting": hgb_params,
    }[selected_name]

    fit_frame = pd.concat((train_frame, validation_frame), ignore_index=True)
    selected_predictor = _fit_estimator(
        selected_name,
        selected_params,
        fit_frame,
        feature_names,
        baseline_feature,
        seed,
    )
    schema_hash = _dataset_feature_schema_hash(dataset, feature_names, feature_definition)
    target_source = _enum_value(dataset.target_source)
    recipe = {
        "training_dataset_id": training_dataset_id,
        "git_commit": git_commit,
        "target_signal": dataset.target_signal_id,
        "target_source": target_source,
        "target_unit": dataset.target_unit,
        "horizon_minutes": horizon_minutes,
        "feature_schema_hash": schema_hash,
        "model_type": selected_name,
        "parameters": selected_params,
        "baseline_feature": baseline_feature,
        "train_end": train_end,
        "validation_end": validation_end,
        "seed": seed,
    }
    model_id = f"{selected_name}-{_stable_hash(recipe)[:12]}"
    metrics = _selection_report(
        selected_name,
        comparison,
        ridge_inner,
        hgb_inner,
        dataset,
    )
    metadata: dict[str, Any] = {
        "model_id": model_id,
        "schema_version": "1.0",
        "model_type": selected_name,
        "training_dataset_id": training_dataset_id,
        "git_commit": git_commit,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "target_signal": dataset.target_signal_id,
        "target_source": target_source,
        "target_unit": dataset.target_unit,
        "horizon_minutes": horizon_minutes,
        "feature_names": feature_names,
        "baseline_feature": baseline_feature,
        "tag_dictionary_sha256": tag_dictionary_sha256,
        "feature_schema_hash": schema_hash,
        "processing": {
            "feature_definition": feature_definition,
            "selected_parameters": selected_params,
            "ridge": "median imputation, standard scaling, L2 regression",
            "hist_gradient_boosting": "native missing values, early_stopping=false",
        },
        "time_boundaries": {
            "train_end_local": train_end,
            "validation_end_local": validation_end,
            "source_timezone": source_timezone,
            "calibration_start": None,
            "calibration_end": None,
        },
        "seed": seed,
        "capabilities": {
            "supports_forecast": True,
            "supports_actions": False,
        },
        "applicability": {
            "purpose": "60-minute point forecast of hydrotreatment sulfur",
            "target_source": target_source,
            "action_comparison": "forbidden",
            "uncertainty_interval": "unavailable_until_stage_5",
        },
        "reports": ("metrics.json",),
    }
    artifact_dir = Path(models_root) / model_id
    bundle = save_model(artifact_dir, selected_predictor, metadata, metrics)
    return TrainingResult(bundle=bundle, artifact_dir=artifact_dir, metrics=metrics)


def select_model(comparison: Mapping[str, Mapping[str, Any]]) -> str:
    """Apply the validation rule: no extra misses, then MAE, then simplicity."""
    if "last_value" not in comparison:
        raise ValueError("comparison must include last_value")
    baseline_missed = _metric_count(comparison["last_value"], "missed_exceedances")
    eligible: list[tuple[str, float]] = []
    for name, metrics in comparison.items():
        if name not in MODEL_COMPLEXITY:
            raise ValueError(f"unknown model family: {name}")
        mae = metrics.get("mae")
        if not isinstance(mae, (int, float)) or not np.isfinite(float(mae)):
            raise ValueError(f"model {name} has no finite validation MAE")
        if _metric_count(metrics, "missed_exceedances") <= baseline_missed:
            eligible.append((name, float(mae)))
    if not eligible:
        return "last_value"
    best_mae = min(mae for _, mae in eligible)
    tied = [name for name, mae in eligible if abs(mae - best_mae) <= 1e-12]
    return min(tied, key=MODEL_COMPLEXITY.__getitem__)


def _model_feature_names(
    frame: pd.DataFrame, declared_features: tuple[str, ...]
) -> tuple[str, ...]:
    if len(declared_features) != len(set(declared_features)):
        raise ValueError("feature_names must be unique")
    missing = set(declared_features).difference(frame.columns)
    if missing:
        raise ValueError(f"declared feature columns are missing: {sorted(missing)}")
    return declared_features


def _baseline_feature_name(
    dataset: TrainingDatasetLike,
    frame: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> str:
    configured = getattr(dataset, "baseline_feature", None)
    if isinstance(configured, str):
        baseline_feature = configured
    elif BASELINE_FEATURE in feature_names:
        baseline_feature = BASELINE_FEATURE
    else:
        suffix = f"__{_enum_value(dataset.target_source)}_last"
        matches = [name for name in feature_names if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError("cannot identify the source-specific last-value feature")
        baseline_feature = matches[0]
    if baseline_feature not in frame.columns or baseline_feature not in feature_names:
        raise ValueError("baseline_feature must be one of the declared model features")
    if BASELINE_FEATURE in frame.columns:
        left = pd.to_numeric(frame[BASELINE_FEATURE], errors="coerce")
        right = pd.to_numeric(frame[baseline_feature], errors="coerce")
        comparable = left.notna() & right.notna()
        if not np.allclose(left[comparable], right[comparable], rtol=0.0, atol=0.0):
            raise ValueError("baseline label and baseline feature disagree")
    return baseline_feature


def _drop_all_missing_train_features(
    train_frame: pd.DataFrame,
    feature_names: tuple[str, ...],
    baseline_feature: str,
) -> tuple[str, ...]:
    kept = tuple(name for name in feature_names if train_frame[name].notna().any())
    if baseline_feature not in kept:
        raise ValueError("the last-value baseline is unavailable throughout train")
    return kept


def _tune_family(
    family: str,
    train_frame: pd.DataFrame,
    feature_names: tuple[str, ...],
    baseline_feature: str,
    candidates: Sequence[object],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    folds = expanding_purged_folds(train_frame, n_splits=3)
    scored: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for candidate in candidates:
        if family == "ridge":
            params = {"alpha": float(candidate)}  # type: ignore[arg-type]
        elif isinstance(candidate, Mapping):
            params = dict(candidate)
        else:
            raise TypeError(f"parameters for {family} must be a mapping")
        fold_actual: list[np.ndarray] = []
        fold_baseline: list[np.ndarray] = []
        fold_prediction: list[np.ndarray] = []
        for fit_indices, validation_indices in folds:
            fit_part = train_frame.iloc[fit_indices]
            validation_part = train_frame.iloc[validation_indices]
            estimator = _fit_estimator(
                family, params, fit_part, feature_names, baseline_feature, seed
            )
            actual, baseline, prediction = _predict_common(
                estimator, validation_part, feature_names, baseline_feature
            )
            fold_actual.append(actual)
            fold_baseline.append(baseline)
            fold_prediction.append(prediction)
        actual = np.concatenate(fold_actual)
        baseline = np.concatenate(fold_baseline)
        prediction = np.concatenate(fold_prediction)
        scored.append(
            (
                params,
                regression_metrics(actual, prediction),
                regression_metrics(actual, baseline),
            )
        )
    if not scored:
        raise ValueError(f"no parameter candidates configured for {family}")
    eligible = [
        item
        for item in scored
        if _metric_count(item[1], "missed_exceedances")
        <= _metric_count(item[2], "missed_exceedances")
    ]
    pool = eligible or scored

    def ordering(
        item: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    ) -> tuple[float, float, str]:
        missed = 0.0 if eligible else float(_metric_count(item[1], "missed_exceedances"))
        return (
            missed,
            float(item[1]["mae"]),
            json.dumps(item[0], sort_keys=True, separators=(",", ":")),
        )

    best_params, best_metrics, _ = min(pool, key=ordering)
    return best_params, {
        "selected_parameters": best_params,
        "selected_metrics": best_metrics,
        "candidates": [
            {
                "parameters": params,
                "metrics": metrics,
                "baseline_metrics": baseline_metrics,
                "passes_baseline_miss_guard": _metric_count(metrics, "missed_exceedances")
                <= _metric_count(baseline_metrics, "missed_exceedances"),
            }
            for params, metrics, baseline_metrics in scored
        ],
        "folds": len(folds),
        "purge_minutes": 60,
    }


def _fit_estimator(
    family: str,
    params: Mapping[str, Any],
    frame: pd.DataFrame,
    feature_names: tuple[str, ...],
    baseline_feature: str,
    seed: int,
) -> Any:
    x = frame.loc[:, list(feature_names)]
    y = pd.to_numeric(frame["y"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y)
    if not valid.any():
        raise ValueError("fit period has no finite targets")
    estimator = _estimator(family, params, baseline_feature, seed)
    return estimator.fit(x.loc[valid], y[valid])


def _estimator(family: str, params: Mapping[str, Any], baseline_feature: str, seed: int) -> Any:
    if family == "last_value":
        return LastValueRegressor(baseline_feature)
    if family == "ridge":
        return Pipeline(
            (
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=float(params["alpha"]))),
            )
        )
    if family == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            early_stopping=False,
            random_state=seed,
            **dict(params),
        )
    raise ValueError(f"unknown model family: {family}")


def _predict_common(
    estimator: Any,
    frame: pd.DataFrame,
    feature_names: tuple[str, ...],
    baseline_feature: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual = pd.to_numeric(frame["y"], errors="coerce").to_numpy(dtype=float)
    baseline = pd.to_numeric(frame[baseline_feature], errors="coerce").to_numpy(dtype=float)
    common = np.isfinite(actual) & np.isfinite(baseline)
    if not common.any():
        raise ValueError("evaluation period has no rows shared with the baseline")
    actual = actual[common]
    baseline = baseline[common]
    prediction = np.asarray(estimator.predict(frame.loc[common, list(feature_names)]), dtype=float)
    if prediction.shape != actual.shape or not np.isfinite(prediction).all():
        raise ValueError("predictor returned invalid validation values")
    return actual, baseline, prediction


def _validation_comparison(
    estimators: Mapping[str, Any],
    validation_frame: pd.DataFrame,
    feature_names: tuple[str, ...],
    baseline_feature: str,
) -> dict[str, dict[str, Any]]:
    actual = pd.to_numeric(validation_frame["y"], errors="coerce").to_numpy(dtype=float)
    baseline = pd.to_numeric(validation_frame[baseline_feature], errors="coerce").to_numpy(
        dtype=float
    )
    common = np.isfinite(actual) & np.isfinite(baseline)
    if not common.any():
        raise ValueError("validation has no rows shared with the baseline")
    actual = actual[common]
    x = validation_frame.loc[common, list(feature_names)]
    result: dict[str, dict[str, Any]] = {}
    for name, estimator in estimators.items():
        prediction = np.asarray(estimator.predict(x), dtype=float)
        if prediction.shape != actual.shape or not np.isfinite(prediction).all():
            raise ValueError(f"model {name} returned invalid validation values")
        result[name] = regression_metrics(actual, prediction)
    return result


def _metric_count(metrics: Mapping[str, Any], name: str) -> int:
    value = metrics.get(name)
    if not isinstance(value, Mapping) or not isinstance(value.get("count"), int):
        raise ValueError(f"metrics are missing {name}.count")
    return int(value["count"])


def _feature_definition(dataset: TrainingDatasetLike) -> dict[str, Any]:
    value = getattr(dataset, "feature_definition", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _dataset_feature_schema_hash(
    dataset: TrainingDatasetLike,
    feature_names: tuple[str, ...],
    feature_definition: Mapping[str, Any],
) -> str:
    value = getattr(dataset, "feature_schema_hash", None)
    if isinstance(value, str) and len(value) == 64:
        return value
    return feature_schema_hash(feature_names, feature_definition)


def _enum_value(value: object) -> str:
    enum_value = getattr(value, "value", value)
    return str(enum_value)


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _selection_report(
    selected_name: str,
    comparison: dict[str, dict[str, Any]],
    ridge_inner: dict[str, Any],
    hgb_inner: dict[str, Any],
    dataset: TrainingDatasetLike,
) -> dict[str, Any]:
    limitations = [
        "point forecast only; no uncertainty interval before stage 5",
        "forecast accuracy does not establish action effects",
    ]
    if _metric_count(comparison["last_value"], "missed_exceedances") == 0:
        denominator = comparison["last_value"]["missed_exceedances"]["denominator"]
        if denominator == 0:
            limitations.append("validation contains no sulfur exceedances")
    excluded = getattr(dataset, "excluded_counts", {})
    return {
        "schema_version": "1.0",
        "selection_split": "validation",
        "test_used_for_selection": False,
        "selected_model": selected_name,
        "models": comparison,
        "inner_cv": {
            "ridge": ridge_inner,
            "hist_gradient_boosting": hgb_inner,
        },
        "excluded_counts": dict(excluded) if isinstance(excluded, Mapping) else {},
        "limitations": limitations,
    }


__all__ = [
    "BASELINE_FEATURE",
    "HGB_CONFIGS",
    "MODEL_COMPLEXITY",
    "RIDGE_ALPHAS",
    "TrainingResult",
    "select_model",
    "train_model",
]
