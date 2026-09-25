"""Leakage-safe feature construction shared by stage-2 training and serving."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import numpy as np
import pandas as pd

from source.contracts import ProcessState, SourceKind, Unit, Validity
from source.data.prepare import PreparedData

FeatureFrame = pd.DataFrame

LAG_MINUTES = (0, 10, 30, 60, 180)
WINDOW_MINUTES = (60, 180)
TELEMETRY_MAX_AGE_MINUTES = 20
TARGET_UNIT = Unit.MG_KG.value
SUPERVISED_COLUMNS = (
    "observation_id",
    "as_of",
    "target_at",
    "target_available_at",
    "target_source",
    "target_signal",
    "target_unit",
    "y",
    "baseline",
)


@dataclass(frozen=True)
class SupervisedDataset:
    """A source-specific target table plus its exactly ordered feature columns."""

    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    target_signal_id: str
    target_unit: str
    target_source: SourceKind
    feature_source: SourceKind
    baseline_feature: str
    feature_definition: dict[str, object]
    excluded_counts: dict[str, int]


@dataclass(frozen=True, eq=False)
class FeatureBatch:
    """Features for one bounded history interval chunk and its exact inputs."""

    data: PreparedData
    model: object
    queries: pd.DatetimeIndex
    frame: FeatureFrame


@dataclass(frozen=True, eq=False)
class FeatureSources:
    """Validated and sorted source tables shared by history replay chunks."""

    data: PreparedData
    model: object
    telemetry: pd.DataFrame
    target_history: pd.DataFrame
    published_history: pd.DataFrame
    feature_spec: tuple[tuple[str, ...], tuple[str, ...], str, SourceKind, str]


def _enum_value(value: object) -> str:
    """Return a stable string for plain strings and string enums."""
    raw = getattr(value, "value", value)
    return str(raw)


def _as_utc(value: datetime | pd.Timestamp, name: str) -> pd.Timestamp:
    """Validate one application timestamp and normalize it to UTC."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _utc_series(values: pd.Series, name: str) -> pd.Series:
    """Parse a prepared timestamp column without accepting broken rows."""
    result = pd.to_datetime(values, utc=True, errors="coerce")
    if result.isna().any():
        raise ValueError(f"{name} contains an invalid timestamp")
    return result


def _quality_rows(
    data: PreparedData,
    target_signal_id: str,
    target_source: SourceKind,
    target_unit: str,
) -> pd.DataFrame:
    """Select finite, valid observations for one named target source and unit."""
    required = {
        "observation_id",
        "signal_id",
        "source",
        "measured_at",
        "available_at",
        "value",
        "unit",
        "validity",
    }
    missing = required.difference(data.quality.columns)
    if missing:
        raise ValueError(f"quality table is missing columns: {sorted(missing)}")

    frame = data.quality.copy()
    values = pd.to_numeric(frame["value"], errors="coerce")
    mask = (
        frame["signal_id"].astype(str).eq(target_signal_id)
        & frame["source"].map(_enum_value).eq(target_source.value)
        & frame["validity"].map(_enum_value).eq(Validity.VALID.value)
        & frame["unit"].astype(str).eq(target_unit)
        & values.notna()
        & np.isfinite(values)
    )
    selected = cast(pd.DataFrame, frame.loc[mask].copy())
    selected["value"] = values.loc[mask].astype(float)
    if selected.empty:
        return selected

    selected["measured_at"] = _utc_series(selected["measured_at"], "quality.measured_at")
    selected["available_at"] = _utc_series(selected["available_at"], "quality.available_at")
    if (selected["available_at"] < selected["measured_at"]).any():
        raise ValueError("quality available_at cannot precede measured_at")
    if selected["measured_at"].duplicated().any():
        raise ValueError("target observations must be unique by measured_at")
    selected.sort_values(["measured_at", "observation_id"], kind="stable", inplace=True)
    selected.reset_index(drop=True, inplace=True)
    return selected


def _telemetry(data: PreparedData, signals: tuple[str, ...]) -> pd.DataFrame:
    """Return sorted numeric telemetry with one row per UTC timestamp."""
    if "timestamp" not in data.telemetry.columns:
        raise ValueError("prepared telemetry needs a timestamp column")
    missing = set(signals).difference(data.telemetry.columns)
    if missing:
        raise ValueError(f"telemetry signals are missing: {sorted(missing)}")

    frame = data.telemetry.loc[:, ["timestamp", *signals]].copy()
    frame["timestamp"] = _utc_series(frame["timestamp"], "telemetry.timestamp")
    if frame["timestamp"].duplicated().any():
        raise ValueError("prepared telemetry timestamps must be unique")
    for signal in signals:
        frame[signal] = pd.to_numeric(frame[signal], errors="coerce")
        frame.loc[~np.isfinite(frame[signal]), signal] = np.nan
    frame.sort_values("timestamp", kind="stable", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def _resolve_telemetry_signals(
    data: PreparedData, telemetry_signals: Iterable[str] | None
) -> tuple[str, ...]:
    """Use autoregression alone unless a curated confirmed signal list is explicit."""
    available = set(data.telemetry.columns).difference({"timestamp"})
    if telemetry_signals is None:
        result: tuple[str, ...] = ()
    else:
        result = tuple(telemetry_signals)
        if len(result) != len(set(result)):
            raise ValueError("telemetry_signals must be unique")
        missing = set(result).difference(available)
        if missing:
            raise ValueError(f"telemetry signals are missing: {sorted(missing)}")
        unapproved = set(result).difference(data.feature_order)
        if unapproved:
            raise ValueError(
                f"telemetry signals are absent from prepared feature_order: {sorted(unapproved)}"
            )
    return result


def _lag_name(signal: str, minutes: int) -> str:
    return f"{signal}__lag_{minutes}m"


def _window_name(signal: str, statistic: str, minutes: int) -> str:
    return f"{signal}__{statistic}_{minutes}m"


def _delta_name(signal: str, minutes: int) -> str:
    return f"{signal}__delta_{minutes}m"


def baseline_feature_name(target_signal_id: str, target_source: SourceKind | str) -> str:
    """Name the source-specific last value used by the persistence baseline."""
    source = SourceKind(target_source)
    return f"{target_signal_id}__{source.value}_last"


def _age_feature_name(target_signal_id: str, target_source: SourceKind) -> str:
    return f"{target_signal_id}__{target_source.value}_age_minutes"


def _quality_lag_name(target_signal_id: str, target_source: SourceKind, minutes: int) -> str:
    return f"{target_signal_id}__{target_source.value}_lag_{minutes}m"


def _quality_delta_name(target_signal_id: str, target_source: SourceKind, minutes: int) -> str:
    return f"{target_signal_id}__{target_source.value}_delta_{minutes}m"


def _quality_window_name(
    target_signal_id: str,
    target_source: SourceKind,
    statistic: str,
    minutes: int,
) -> str:
    return f"{target_signal_id}__{target_source.value}_{statistic}_{minutes}m"


def _lag_values(
    telemetry: pd.DataFrame,
    queries: pd.DatetimeIndex,
    signals: tuple[str, ...],
) -> dict[tuple[str, int], np.ndarray[Any, np.dtype[np.float64]]]:
    """Find only observations at or before each requested lag cutoff."""
    result: dict[tuple[str, int], np.ndarray[Any, np.dtype[np.float64]]] = {}
    if not signals:
        return result
    right = telemetry.loc[:, ["timestamp", *signals]]
    for lag in LAG_MINUTES:
        left = pd.DataFrame(
            {
                "_position": np.arange(len(queries)),
                "_cutoff": queries - pd.Timedelta(minutes=lag),
            }
        ).sort_values("_cutoff", kind="stable")
        merged = pd.merge_asof(
            left,
            right,
            left_on="_cutoff",
            right_on="timestamp",
            direction="backward",
            tolerance=pd.Timedelta(minutes=TELEMETRY_MAX_AGE_MINUTES),
        ).sort_values("_position", kind="stable")
        for signal in signals:
            result[(signal, lag)] = merged[signal].to_numpy(dtype=float)
    return result


def _window_values(
    telemetry: pd.DataFrame,
    queries: pd.DatetimeIndex,
    signals: tuple[str, ...],
) -> dict[tuple[str, str, int], np.ndarray[Any, np.dtype[np.float64]]]:
    """Calculate windows ending exactly at as_of, never at a later sample."""
    result: dict[tuple[str, str, int], np.ndarray[Any, np.dtype[np.float64]]] = {}
    if not signals:
        return result
    base = telemetry.set_index("timestamp").loc[:, list(signals)]
    timeline = base.index.union(queries).sort_values()
    expanded = base.reindex(timeline)
    for window in WINDOW_MINUTES:
        rolling = expanded.rolling(f"{window}min", closed="both", min_periods=1)
        means = rolling.mean().reindex(queries)
        deviations = rolling.std(ddof=0).reindex(queries)
        for signal in signals:
            result[(signal, "mean", window)] = means[signal].to_numpy(dtype=float)
            result[(signal, "std", window)] = deviations[signal].to_numpy(dtype=float)
    return result


def _last_available_quality(
    observations: pd.DataFrame,
    queries: pd.DatetimeIndex,
    *,
    events_sorted: bool = False,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    """Select the latest measured observation among those published by each as_of."""
    values = np.full(len(queries), np.nan, dtype=float)
    ages = np.full(len(queries), np.nan, dtype=float)
    if observations.empty or queries.empty:
        return values, ages

    events = (
        observations
        if events_sorted
        else observations.sort_values(
            ["available_at", "measured_at", "observation_id"], kind="stable"
        ).reset_index(drop=True)
    )
    available = pd.DatetimeIndex(events["available_at"]).as_unit("ns").astype("int64").to_numpy()
    measured = pd.DatetimeIndex(events["measured_at"]).as_unit("ns").astype("int64").to_numpy()
    query_times = queries.as_unit("ns").astype("int64").to_numpy()
    # Prefix records preserve the latest *measured* reading, even when an older
    # lab sample is published later. Search publication time, never future data.
    latest_measured = np.maximum.accumulate(measured)
    new_record = np.r_[True, measured[1:] > latest_measured[:-1]]
    latest_index = np.maximum.accumulate(np.where(new_record, np.arange(len(events)), 0))
    event_positions = np.searchsorted(available, query_times, side="right") - 1
    visible = event_positions >= 0
    positions = np.flatnonzero(visible)
    selected = latest_index[event_positions[visible]]
    safe = measured[selected] <= query_times[visible]
    positions, selected = positions[safe], selected[safe]
    values[positions] = events["value"].to_numpy(dtype=float)[selected]
    ages[positions] = (query_times[positions] - measured[selected]) / 60_000_000_000.0
    return values, ages


def _quality_window_values(
    observations: pd.DataFrame,
    queries: pd.DatetimeIndex,
) -> dict[tuple[str, int], np.ndarray[Any, np.dtype[np.float64]]]:
    """Aggregate only measurements already published by each decision time."""
    result: dict[tuple[str, int], np.ndarray[Any, np.dtype[np.float64]]] = {}
    for window in WINDOW_MINUTES:
        result[("mean", window)] = np.full(len(queries), np.nan, dtype=float)
        result[("std", window)] = np.full(len(queries), np.nan, dtype=float)
    if observations.empty or queries.empty:
        return result

    measured = pd.DatetimeIndex(observations["measured_at"])
    available = pd.DatetimeIndex(observations["available_at"])
    values = observations["value"].to_numpy(dtype=float)
    if measured.equals(available):
        series = pd.Series(values, index=measured).sort_index()
        timeline = series.index.union(queries).sort_values()
        expanded = series.reindex(timeline)
        for window in WINDOW_MINUTES:
            rolling = expanded.rolling(f"{window}min", closed="both", min_periods=1)
            result[("mean", window)] = rolling.mean().reindex(queries).to_numpy(dtype=float)
            result[("std", window)] = rolling.std(ddof=0).reindex(queries).to_numpy(dtype=float)
        return result

    for position, query in enumerate(queries):
        visible = (available <= query) & (measured <= query)
        for window in WINDOW_MINUTES:
            in_window = visible & (measured >= query - pd.Timedelta(minutes=window))
            selected = values[in_window]
            if selected.size:
                result[("mean", window)][position] = float(selected.mean())
                result[("std", window)][position] = float(selected.std(ddof=0))
    return result


def _feature_matrix(
    data: PreparedData,
    queries: pd.DatetimeIndex,
    telemetry_signals: tuple[str, ...],
    target_signal_id: str,
    target_source: SourceKind,
    target_unit: str,
    *,
    sources: FeatureSources | None = None,
) -> pd.DataFrame:
    """Build the common training/serving matrix for already-normalized UTC times."""
    telemetry = sources.telemetry if sources is not None else _telemetry(data, telemetry_signals)
    lag_values = _lag_values(telemetry, queries, telemetry_signals)
    window_values = _window_values(telemetry, queries, telemetry_signals)
    target_history = (
        sources.target_history
        if sources is not None
        else _quality_rows(data, target_signal_id, target_source, target_unit)
    )
    published_history = (
        sources.published_history
        if sources is not None
        else target_history.sort_values(
            ["available_at", "measured_at", "observation_id"], kind="stable"
        ).reset_index(drop=True)
    )
    last_values, ages = _last_available_quality(published_history, queries, events_sorted=True)
    quality_lags = {
        lag: _last_available_quality(
            published_history, queries - pd.Timedelta(minutes=lag), events_sorted=True
        )[0]
        for lag in LAG_MINUTES
    }
    quality_windows = _quality_window_values(target_history, queries)

    columns: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
    for signal in telemetry_signals:
        for lag in LAG_MINUTES:
            columns[_lag_name(signal, lag)] = lag_values[(signal, lag)]
        for lag in LAG_MINUTES[1:]:
            columns[_delta_name(signal, lag)] = lag_values[(signal, 0)] - lag_values[(signal, lag)]
        for window in WINDOW_MINUTES:
            columns[_window_name(signal, "mean", window)] = window_values[(signal, "mean", window)]
            columns[_window_name(signal, "std", window)] = window_values[(signal, "std", window)]
    columns[baseline_feature_name(target_signal_id, target_source)] = last_values
    columns[_age_feature_name(target_signal_id, target_source)] = ages
    for lag in LAG_MINUTES:
        columns[_quality_lag_name(target_signal_id, target_source, lag)] = quality_lags[lag]
    for lag in LAG_MINUTES[1:]:
        columns[_quality_delta_name(target_signal_id, target_source, lag)] = (
            quality_lags[0] - quality_lags[lag]
        )
    for window in WINDOW_MINUTES:
        for statistic in ("mean", "std"):
            columns[_quality_window_name(target_signal_id, target_source, statistic, window)] = (
                quality_windows[(statistic, window)]
            )
    return pd.DataFrame(columns)


def build_supervised_dataset(
    data: PreparedData,
    target_signal_id: str,
    target_source: SourceKind | str,
    horizon_minutes: int = 60,
    telemetry_signals: Iterable[str] | None = None,
    feature_source: SourceKind | str | None = None,
) -> SupervisedDataset:
    """Build exact PAK-grid or one-row-per-analysis LIMS training examples."""
    source = SourceKind(target_source)
    history_source = source if feature_source is None else SourceKind(feature_source)
    if source not in {SourceKind.PAK, SourceKind.LIMS}:
        raise ValueError("stage-2 sulfur targets must come from PAK or LIMS")
    if history_source not in {SourceKind.PAK, SourceKind.LIMS}:
        raise ValueError("stage-2 autoregressive history must come from PAK or LIMS")
    if horizon_minutes <= 0:
        raise ValueError("horizon_minutes must be positive")

    targets = _quality_rows(data, target_signal_id, source, TARGET_UNIT)
    if targets.empty:
        raise ValueError(
            f"no valid {source.value} targets for {target_signal_id!r} in {TARGET_UNIT}"
        )
    signals = _resolve_telemetry_signals(data, telemetry_signals)
    target_at = pd.DatetimeIndex(targets["measured_at"])
    target_available_at = pd.DatetimeIndex(targets["available_at"])
    as_of = target_at - pd.Timedelta(minutes=horizon_minutes)
    features = _feature_matrix(data, as_of, signals, target_signal_id, history_source, TARGET_UNIT)
    baseline_name = baseline_feature_name(target_signal_id, history_source)
    metadata = pd.DataFrame(
        {
            "observation_id": targets["observation_id"].astype(str).to_numpy(),
            "as_of": as_of,
            "target_at": target_at,
            "target_available_at": target_available_at,
            "target_source": source.value,
            "target_signal": target_signal_id,
            "target_unit": TARGET_UNIT,
            "y": targets["value"].to_numpy(dtype=float),
            "baseline": features[baseline_name].to_numpy(dtype=float),
        }
    )
    frame = pd.concat([metadata, features], axis="columns")
    source_rows = data.quality[
        data.quality["signal_id"].astype(str).eq(target_signal_id)
        & data.quality["source"].map(_enum_value).eq(source.value)
    ]
    return SupervisedDataset(
        frame=frame,
        feature_names=tuple(features.columns),
        target_signal_id=target_signal_id,
        target_unit=TARGET_UNIT,
        target_source=source,
        feature_source=history_source,
        baseline_feature=baseline_name,
        feature_definition={
            "horizon_minutes": horizon_minutes,
            "lags_minutes": list(LAG_MINUTES),
            "rolling_windows_minutes": list(WINDOW_MINUTES),
            "telemetry_max_age_minutes": TELEMETRY_MAX_AGE_MINUTES,
            "telemetry_signals": list(signals),
            "quality_history_source": history_source.value,
            "autoregressive_target_signal": target_signal_id,
        },
        excluded_counts={
            "invalid_or_wrong_unit_targets": int(len(source_rows) - len(targets)),
            "missing_baseline": int(features[baseline_name].isna().sum()),
        },
    )


def _metadata_value(metadata: object, *names: str) -> object:
    """Read one required field from strict metadata or its serialized dict."""
    for name in names:
        if isinstance(metadata, dict) and name in metadata:
            return metadata[name]
        if hasattr(metadata, name):
            return getattr(metadata, name)
    raise ValueError(f"model metadata is missing {names[0]}")


def _model_feature_spec(
    data: PreparedData, model: object
) -> tuple[tuple[str, ...], tuple[str, ...], str, SourceKind, str]:
    """Resolve the frozen model's feature contract once per feature build."""
    metadata = getattr(model, "metadata", None)
    if metadata is None:
        raise ValueError("model needs metadata")
    raw_feature_names = _metadata_value(metadata, "feature_names")
    if isinstance(raw_feature_names, str) or not isinstance(raw_feature_names, (list, tuple)):
        raise ValueError("model feature_names must be a list or tuple")
    feature_names = tuple(str(item) for item in raw_feature_names)
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("model feature_names must be unique")
    target_signal_id = str(_metadata_value(metadata, "target_signal", "target_signal_id"))
    target_source = SourceKind(_enum_value(_metadata_value(metadata, "target_source")))
    target_unit = str(_metadata_value(metadata, "target_unit"))
    processing = (
        metadata.get("processing", {})
        if isinstance(metadata, dict)
        else getattr(metadata, "processing", {})
    )
    if isinstance(processing, dict):
        definition = processing.get("feature_definition", {})
        if isinstance(definition, dict) and "quality_history_source" in definition:
            target_source = SourceKind(_enum_value(definition["quality_history_source"]))

    telemetry_signals = tuple(
        signal
        for signal in data.feature_order
        if signal in data.telemetry.columns
        and any(name.startswith(f"{signal}__") for name in feature_names)
    )
    if isinstance(processing, dict) and isinstance(processing.get("feature_definition"), dict):
        definition = processing["feature_definition"]
        expected_recipe = {
            "horizon_minutes": _metadata_value(metadata, "horizon_minutes"),
            "lags_minutes": list(LAG_MINUTES),
            "rolling_windows_minutes": list(WINDOW_MINUTES),
            "telemetry_max_age_minutes": TELEMETRY_MAX_AGE_MINUTES,
            "telemetry_signals": list(telemetry_signals),
            "quality_history_source": target_source.value,
            "autoregressive_target_signal": target_signal_id,
        }
        mismatched = [
            key
            for key, value in expected_recipe.items()
            if key in definition
            and (
                sorted(definition[key]) != sorted(telemetry_signals)
                if key == "telemetry_signals" and isinstance(definition[key], list)
                else definition[key] != value
            )
        ]
        if mismatched:
            raise ValueError("model feature recipe is incompatible with serving code")
    return feature_names, telemetry_signals, target_signal_id, target_source, target_unit


def prepare_feature_sources(data: PreparedData, model: object) -> FeatureSources:
    """Parse the large source tables once for a historical interval."""
    spec = _model_feature_spec(data, model)
    _, telemetry_signals, target_signal_id, target_source, target_unit = spec
    target_history = _quality_rows(data, target_signal_id, target_source, target_unit)
    published_history = target_history.sort_values(
        ["available_at", "measured_at", "observation_id"], kind="stable"
    ).reset_index(drop=True)
    return FeatureSources(
        data,
        model,
        _telemetry(data, telemetry_signals),
        target_history,
        published_history,
        spec,
    )


def prepare_feature_batch(
    data: PreparedData,
    queries: pd.DatetimeIndex,
    model: object,
    *,
    sources: FeatureSources | None = None,
) -> FeatureBatch:
    """Build feature rows together for a bounded, UTC-aware replay chunk."""
    if queries.empty or queries.tz is None:
        raise ValueError("feature batch requires timezone-aware, nonempty queries")
    utc_queries = queries.tz_convert("UTC")
    if sources is not None and (sources.data is not data or sources.model is not model):
        raise ValueError("feature sources inputs do not match this replay")
    feature_names, telemetry_signals, target_signal_id, target_source, target_unit = (
        sources.feature_spec if sources is not None else _model_feature_spec(data, model)
    )
    generated = _feature_matrix(
        data,
        utc_queries,
        telemetry_signals,
        target_signal_id,
        target_source,
        target_unit,
        sources=sources,
    )
    missing = set(feature_names).difference(generated.columns)
    if missing:
        raise ValueError(f"model requests unknown features: {sorted(missing)}")
    return FeatureBatch(data, model, utc_queries, generated.loc[:, list(feature_names)])


def build_features(
    data: PreparedData,
    as_of: datetime,
    state: ProcessState,
    model: object,
    *,
    batch: FeatureBatch | None = None,
) -> FeatureFrame:
    """Build one serving row in the exact feature order stored by the model."""
    query = _as_utc(as_of, "as_of")
    if _as_utc(state.as_of, "state.as_of") != query:
        raise ValueError("state.as_of must match feature as_of")
    if state.dataset_id != data.manifest.dataset_id:
        raise ValueError("state and prepared data dataset_id mismatch")
    feature_names, telemetry_signals, target_signal_id, target_source, target_unit = (
        _model_feature_spec(data, model)
    )
    if batch is not None:
        if batch.data is not data or batch.model is not model:
            raise ValueError("feature batch inputs do not match this replay")
        if tuple(batch.frame.columns) != feature_names:
            raise ValueError("feature batch columns do not match model")
        position = batch.queries.get_indexer(pd.DatetimeIndex([query]))[0]
        if position < 0:
            raise ValueError("feature batch does not contain as_of")
        return batch.frame.iloc[[position]].reset_index(drop=True)
    generated = _feature_matrix(
        data,
        pd.DatetimeIndex([query]),
        telemetry_signals,
        target_signal_id,
        target_source,
        target_unit,
    )
    missing = set(feature_names).difference(generated.columns)
    if missing:
        raise ValueError(f"model requests unknown features: {sorted(missing)}")
    return generated.loc[:, list(feature_names)]


__all__ = [
    "FeatureBatch",
    "FeatureFrame",
    "FeatureSources",
    "LAG_MINUTES",
    "SUPERVISED_COLUMNS",
    "SupervisedDataset",
    "WINDOW_MINUTES",
    "baseline_feature_name",
    "build_features",
    "prepare_feature_batch",
    "prepare_feature_sources",
    "build_supervised_dataset",
]
