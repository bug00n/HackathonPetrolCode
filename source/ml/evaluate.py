"""Leakage-safe temporal splits and stage-2 forecast metrics."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from source.ml.artifacts import ModelBundle

SplitName = Literal["train", "validation", "test"]
_TIME_COLUMNS = ("as_of", "target_at", "target_available_at")


class SupervisedDatasetLike(Protocol):
    """Small structural contract shared with ``ml.features``."""

    @property
    def frame(self) -> pd.DataFrame: ...

    @property
    def feature_names(self) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class TemporalSplit:
    """Positional row indices for the fixed outer time split."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def indices(self, name: SplitName) -> np.ndarray:
        """Return one named split without stringly-typed ``getattr`` calls."""
        if name == "train":
            return self.train
        if name == "validation":
            return self.validation
        return self.test


@dataclass(frozen=True)
class EvaluationResult:
    """JSON-ready metrics plus row-level residuals for error analysis."""

    report: dict[str, Any]
    residuals: pd.DataFrame


def make_outer_split(
    frame: pd.DataFrame,
    *,
    source_timezone: str = "Europe/Moscow",
    train_end: str = "2025-01-01",
    validation_end: str = "2026-01-01",
) -> TemporalSplit:
    """Split by decision time and purge labels unavailable at either boundary."""
    times = _validated_times(frame)
    train_boundary = _local_boundary(train_end, source_timezone)
    validation_boundary = _local_boundary(validation_end, source_timezone)
    if train_boundary >= validation_boundary:
        raise ValueError("train_end must precede validation_end")

    decision = times["as_of"]
    target = times["target_at"]
    available = times["target_available_at"]
    train = (decision < train_boundary) & (target < train_boundary) & (available < train_boundary)
    validation = (
        (decision >= train_boundary)
        & (decision < validation_boundary)
        & (target < validation_boundary)
        & (available < validation_boundary)
    )
    test = decision >= validation_boundary
    return TemporalSplit(
        train=np.flatnonzero(train.to_numpy()),
        validation=np.flatnonzero(validation.to_numpy()),
        test=np.flatnonzero(test.to_numpy()),
    )


def validate_supervised_horizon(frame: pd.DataFrame, horizon_minutes: int) -> None:
    """Require every label timestamp to match the horizon declared by the model."""
    if horizon_minutes <= 0:
        raise ValueError("horizon_minutes must be positive")
    times = _validated_times(frame)
    actual = (times["target_at"] - times["as_of"]).dt.total_seconds() / 60.0
    if not np.allclose(actual.to_numpy(dtype=float), horizon_minutes, rtol=0.0, atol=1e-9):
        raise ValueError("supervised target horizon does not match the declared model horizon")


def expanding_purged_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int = 3,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Create expanding folds with a time-based target/publication purge."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    times = _validated_times(frame)
    unique_decisions = np.asarray(sorted(times["as_of"].unique()))
    if len(unique_decisions) < n_splits + 1:
        raise ValueError("not enough distinct timestamps for expanding folds")
    blocks = np.array_split(unique_decisions, n_splits + 1)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for block in blocks[1:]:
        validation_start = pd.Timestamp(block[0])
        validation_times = set(block.tolist())
        validation = times["as_of"].isin(validation_times)
        train = (
            (times["as_of"] < validation_start)
            & (times["target_at"] < validation_start)
            & (times["target_available_at"] < validation_start)
        )
        train_indices = np.flatnonzero(train.to_numpy())
        validation_indices = np.flatnonzero(validation.to_numpy())
        if len(train_indices) == 0 or len(validation_indices) == 0:
            raise ValueError("purge left an empty temporal fold")
        folds.append((train_indices, validation_indices))
    return tuple(folds)


def regression_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    *,
    threshold: float = 10.0,
    near_lower: float = 8.0,
    near_upper: float = 12.0,
) -> dict[str, Any]:
    """Compute the exact point-forecast metrics defined in DESIGN section 8."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.ndim != 1 or predicted.ndim != 1 or actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must be equal-length one-dimensional arrays")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("metrics do not silently discard non-finite values")
    if near_lower > near_upper:
        raise ValueError("near_lower cannot exceed near_upper")

    errors = np.abs(actual - predicted)
    near = (actual >= near_lower) & (actual <= near_upper)
    actual_exceedance = actual > threshold
    predicted_alarm = predicted > threshold
    missed = actual_exceedance & ~predicted_alarm
    false_alarm = ~actual_exceedance & predicted_alarm
    actual_exceedance_count = int(actual_exceedance.sum())
    actual_non_exceedance_count = int((~actual_exceedance).sum())
    return {
        "n": int(len(actual)),
        "mae": float(errors.mean()) if len(errors) else None,
        "near_threshold": {
            "lower": float(near_lower),
            "upper": float(near_upper),
            "n": int(near.sum()),
            "mae": float(errors[near].mean()) if near.any() else None,
        },
        "missed_exceedances": {
            "count": int(missed.sum()),
            "denominator": actual_exceedance_count,
            "rate": _ratio(int(missed.sum()), actual_exceedance_count),
        },
        "false_alarms": {
            "count": int(false_alarm.sum()),
            "denominator": actual_non_exceedance_count,
            "rate": _ratio(int(false_alarm.sum()), actual_non_exceedance_count),
        },
    }


def evaluate_model(
    dataset: SupervisedDatasetLike,
    model: ModelBundle,
    *,
    split: SplitName = "test",
    source_timezone: str | None = None,
) -> EvaluationResult:
    """Compare a saved model and persistence baseline on identical timestamps."""
    frame = dataset.frame.copy().reset_index(drop=True)
    validate_supervised_horizon(frame, model.metadata.horizon_minutes)
    stored_timezone, train_end, validation_end = _artifact_split_settings(model, source_timezone)
    dataset_unit = getattr(dataset, "target_unit", model.metadata.target_unit)
    if str(dataset_unit) != model.metadata.target_unit:
        raise ValueError("evaluation target unit is incompatible with the model")
    required = {"y", model.metadata.baseline_feature, *model.feature_names}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"supervised frame is missing columns: {missing}")
    split_indices = make_outer_split(
        frame,
        source_timezone=stored_timezone,
        train_end=train_end,
        validation_end=validation_end,
    ).indices(split)
    selected = frame.iloc[split_indices].copy()
    target = pd.to_numeric(selected["y"], errors="coerce").to_numpy(dtype=float)
    baseline = pd.to_numeric(selected[model.metadata.baseline_feature], errors="coerce").to_numpy(
        dtype=float
    )
    common = np.isfinite(target) & np.isfinite(baseline)
    target_count = int(np.isfinite(target).sum())
    selected = selected.loc[common].copy()
    target = target[common]
    baseline = baseline[common]
    if len(selected) == 0:
        raise ValueError(f"split {split!r} has no rows shared with the baseline")
    prediction = model.predict(selected.loc[:, list(model.feature_names)])

    residuals = _residual_frame(selected, target, baseline, prediction, model)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "model_id": model.metadata.model_id,
        "split": split,
        "target_signal": model.metadata.target_signal,
        "training_target_source": model.metadata.target_source,
        "evaluation_target_sources": sorted(
            str(value) for value in residuals["target_source"].dropna().unique()
        )
        if "target_source" in residuals
        else [],
        "feature_source": getattr(
            getattr(dataset, "feature_source", None),
            "value",
            getattr(dataset, "feature_source", model.metadata.target_source),
        ),
        "coverage": {
            "available_predictions": int(len(target)),
            "valid_targets": target_count,
            "rate": _ratio(len(target), target_count),
        },
        "metrics": {
            "baseline": regression_metrics(target, baseline),
            "model": regression_metrics(target, prediction),
        },
        "slices": {
            "month": _slice_metrics(residuals, "month"),
            "target_source": _slice_metrics(residuals, "target_source")
            if "target_source" in residuals
            else {},
            "near_threshold": _slice_metrics(residuals, "near_threshold"),
            "missing_feature_count": _slice_metrics(residuals, "missing_feature_count"),
            "baseline_age_bucket": _slice_metrics(residuals, "baseline_age_bucket")
            if "baseline_age_bucket" in residuals
            else {},
        },
    }
    return EvaluationResult(report=report, residuals=residuals)


def _artifact_split_settings(
    model: ModelBundle, requested_timezone: str | None
) -> tuple[str, str, str]:
    """Use the exact temporal boundaries persisted with the fitted artifact."""
    boundaries = model.metadata.time_boundaries
    values = {
        name: boundaries.get(name)
        for name in ("source_timezone", "train_end_local", "validation_end_local")
    }
    if any(not isinstance(value, str) or not value for value in values.values()):
        raise ValueError("model metadata is missing temporal split settings")
    stored_timezone = str(values["source_timezone"])
    if requested_timezone is not None and requested_timezone != stored_timezone:
        raise ValueError("evaluation timezone does not match the fitted artifact")
    return (
        stored_timezone,
        str(values["train_end_local"]),
        str(values["validation_end_local"]),
    )


def write_evaluation(directory: Path, result: EvaluationResult) -> Path:
    """Atomically write metrics and row-level residuals for one evaluation."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"evaluation directory already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        (temporary / "metrics.json").write_text(
            json.dumps(
                result.report,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        result.residuals.to_csv(temporary / "residuals.csv.gz", index=False, compression="gzip")
        temporary.replace(directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return directory


def _validated_times(frame: pd.DataFrame) -> dict[str, pd.Series]:
    missing = sorted(set(_TIME_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"supervised frame is missing time columns: {missing}")
    result = {name: _aware_utc(frame[name], name) for name in _TIME_COLUMNS}
    if (result["target_at"] < result["as_of"]).any():
        raise ValueError("target_at cannot precede as_of")
    if (result["target_available_at"] < result["target_at"]).any():
        raise ValueError("target_available_at cannot precede target_at")
    return result


def _aware_utc(values: pd.Series, name: str) -> pd.Series:
    parsed_values = [pd.Timestamp(value) for value in values]
    if any(value.tzinfo is None or value.utcoffset() is None for value in parsed_values):
        raise ValueError(f"{name} must contain timezone-aware timestamps")
    return pd.Series(pd.to_datetime(parsed_values, utc=True), index=values.index, name=name)


def _local_boundary(value: str, source_timezone: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        raise ValueError("split boundaries must be local calendar dates without a timezone")
    return timestamp.tz_localize(ZoneInfo(source_timezone)).tz_convert("UTC")


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _residual_frame(
    frame: pd.DataFrame,
    target: np.ndarray,
    baseline: np.ndarray,
    prediction: np.ndarray,
    model: ModelBundle,
) -> pd.DataFrame:
    label_columns = [
        name
        for name in (
            "observation_id",
            "as_of",
            "target_at",
            "target_available_at",
            "target_source",
            "target_signal",
            "target_unit",
        )
        if name in frame.columns
    ]
    residuals = frame.loc[:, label_columns].reset_index(drop=True)
    residuals["y_true"] = target
    residuals["baseline_prediction"] = baseline
    residuals["model_prediction"] = prediction
    residuals["residual"] = target - prediction
    residuals["absolute_error"] = np.abs(target - prediction)
    residuals["near_threshold"] = (target >= 8.0) & (target <= 12.0)
    residuals["actual_exceedance"] = target > 10.0
    residuals["predicted_alarm"] = prediction > 10.0
    residuals["missing_feature_count"] = (
        frame.loc[:, list(model.feature_names)].isna().sum(axis=1).to_numpy()
    )
    age_features = [name for name in model.feature_names if name.endswith("_age_minutes")]
    if len(age_features) == 1:
        age = pd.to_numeric(frame[age_features[0]], errors="coerce").reset_index(drop=True)
        residuals["baseline_age_minutes"] = age
        residuals["baseline_age_bucket"] = pd.cut(
            age,
            bins=[-np.inf, 30.0, 120.0, 360.0, 1_440.0, np.inf],
            labels=["<=30m", "30-120m", "2-6h", "6-24h", ">24h"],
        ).astype("string")
    residuals["month"] = pd.to_datetime(residuals["target_at"], utc=True).dt.strftime("%Y-%m")
    return residuals


def _slice_metrics(residuals: pd.DataFrame, column: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value, group in residuals.groupby(column, dropna=False, sort=True):
        key = "null" if pd.isna(value) else str(value)
        target = group["y_true"].to_numpy(dtype=float)
        result[key] = {
            "baseline": regression_metrics(
                target, group["baseline_prediction"].to_numpy(dtype=float)
            ),
            "model": regression_metrics(target, group["model_prediction"].to_numpy(dtype=float)),
        }
    return result


__all__ = [
    "EvaluationResult",
    "SplitName",
    "TemporalSplit",
    "evaluate_model",
    "expanding_purged_folds",
    "make_outer_split",
    "regression_metrics",
    "validate_supervised_horizon",
    "write_evaluation",
]
