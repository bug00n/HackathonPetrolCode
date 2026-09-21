"""Reproducible diagnostics for sulfur alignment, drift and feature usefulness."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from source.contracts import SourceKind
from source.data.prepare import PreparedData
from source.ml.evaluate import make_outer_split
from source.ml.features import SupervisedDataset

SULFUR_SIGNAL = "ht:2:Mg.Sulfur"
SULFUR_LIMIT = 10.0
CONFIRMED_LIMS_OUTLIER_IDS = frozenset(
    {
        "c8a61777-9c04-5999-9a7c-ad46bc5e5d60",
        "8e60da42-5108-5e92-a1aa-8608455aa850",
        "26a758a9-3257-5e19-91da-efc9d04fc57d",
        "41e64ee3-d1c7-5553-b630-ed28bfc5f4dc",
        "74afa5b5-95e6-592c-80d2-ab41edffb64e",
        "4b4cb099-2147-5c9f-b632-c07bd0604cf8",
    }
)


def _source_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _sulfur_rows(data: PreparedData, source: SourceKind) -> pd.DataFrame:
    quality = data.quality
    rows = quality[
        quality["signal_id"].astype(str).eq(SULFUR_SIGNAL)
        & quality["source"].map(_source_value).eq(source.value)
    ].loc[:, ["observation_id", "measured_at", "value"]]
    rows = rows.copy()
    rows["measured_at"] = pd.to_datetime(rows["measured_at"], utc=True)
    rows["value"] = pd.to_numeric(rows["value"], errors="coerce")
    return rows.dropna().sort_values("measured_at", kind="stable")


def exclude_confirmed_lims_outliers(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Exclude only user-confirmed observation IDs while retaining raw provenance."""
    if "observation_id" not in frame.columns:
        raise ValueError("LIMS outlier policy requires observation_id")
    ids = frame["observation_id"].astype(str)
    excluded = tuple(sorted(set(ids).intersection(CONFIRMED_LIMS_OUTLIER_IDS)))
    return frame.loc[~ids.isin(CONFIRMED_LIMS_OUTLIER_IDS)].copy(), excluded


def pak_lims_alignment(
    data: PreparedData,
    *,
    offsets_minutes: Iterable[int] = range(-180, 181, 30),
) -> dict[str, Any]:
    """Compare timestamp shifts before and after confirmed observation exclusions."""
    pak = _sulfur_rows(data, SourceKind.PAK)
    lims = _sulfur_rows(data, SourceKind.LIMS)
    eligible_lims, excluded_ids = exclude_confirmed_lims_outliers(lims)

    def evaluate(selected: pd.DataFrame) -> list[dict[str, float | int | None]]:
        result: list[dict[str, float | int | None]] = []
        for offset in offsets_minutes:
            queries = selected.assign(
                query=selected["measured_at"] + pd.Timedelta(minutes=int(offset))
            )
            matched = pd.merge_asof(
                queries.sort_values("query", kind="stable"),
                pak,
                left_on="query",
                right_on="measured_at",
                direction="nearest",
                tolerance=pd.Timedelta(minutes=20),
                suffixes=("_lims", "_pak"),
            ).dropna(subset=["value_lims", "value_pak"])
            correlation = (
                matched["value_lims"].corr(matched["value_pak"])
                if len(matched) >= 2
                and matched["value_lims"].std() > 0.0
                and matched["value_pak"].std() > 0.0
                else np.nan
            )
            result.append(
                {
                    "offset_minutes": int(offset),
                    "n": int(len(matched)),
                    "mae": float(np.mean(np.abs(matched["value_lims"] - matched["value_pak"])))
                    if len(matched)
                    else None,
                    "bias_lims_minus_pak": float(
                        np.mean(matched["value_lims"] - matched["value_pak"])
                    )
                    if len(matched)
                    else None,
                    "correlation": float(correlation) if pd.notna(correlation) else None,
                }
            )
        return result

    return {
        "raw": evaluate(lims),
        "training_policy": {
            "interpretation": "confirmed outliers stay in prepared data but are excluded from ML",
            "excluded_count": len(excluded_ids),
            "excluded_observation_ids": excluded_ids,
            "excluded_values": sorted(
                float(value)
                for value in lims.loc[
                    lims["observation_id"].astype(str).isin(excluded_ids), "value"
                ]
            ),
            "results": evaluate(eligible_lims),
        },
    }


def temporal_drift(dataset: SupervisedDataset) -> dict[str, Any]:
    """Describe target drift and persistence failure on immutable temporal periods."""
    frame = dataset.frame.reset_index(drop=True)
    split = make_outer_split(frame)
    periods = {"train": split.train, "validation": split.validation, "test": split.test}
    result: dict[str, Any] = {}
    for name, indices in periods.items():
        selected = frame.iloc[indices]
        actual = pd.to_numeric(selected["y"], errors="coerce").to_numpy(dtype=float)
        baseline = pd.to_numeric(selected[dataset.baseline_feature], errors="coerce").to_numpy(
            dtype=float
        )
        valid = np.isfinite(actual) & np.isfinite(baseline)
        actual = actual[valid]
        baseline = baseline[valid]
        transition = (baseline <= SULFUR_LIMIT) & (actual > SULFUR_LIMIT)
        correlation = (
            float(np.corrcoef(baseline, actual)[0, 1])
            if baseline.std() > 0.0 and actual.std() > 0.0
            else None
        )
        result[name] = {
            "n": int(len(actual)),
            "target_mean": float(actual.mean()),
            "target_std": float(actual.std()),
            "exceedance_rate": float(np.mean(actual > SULFUR_LIMIT)),
            "transition_rate": float(transition.mean()),
            "transition_count": int(transition.sum()),
            "persistence_mae": float(np.mean(np.abs(actual - baseline))),
            "current_future_correlation": correlation,
        }
    return result


def regime_diagnostics(dataset: SupervisedDataset) -> dict[str, Any]:
    """Locate persistence errors in operationally meaningful current-sulfur bands."""
    frame = dataset.frame.reset_index(drop=True)
    split = make_outer_split(frame)
    periods = {"train": split.train, "validation": split.validation, "test": split.test}
    bins = (-np.inf, 8.0, 10.0, 12.0, np.inf)
    labels = ("below_8", "8_to_10", "10_to_12", "above_12")
    result: dict[str, Any] = {}
    for period, indices in periods.items():
        selected = frame.iloc[indices]
        actual = pd.to_numeric(selected["y"], errors="coerce")
        baseline = pd.to_numeric(selected[dataset.baseline_feature], errors="coerce")
        band = pd.cut(baseline, bins=bins, labels=labels, right=False)
        period_rows: dict[str, Any] = {}
        for label in labels:
            mask = band.eq(label) & actual.notna() & baseline.notna()
            y = actual[mask].to_numpy(dtype=float)
            current = baseline[mask].to_numpy(dtype=float)
            period_rows[label] = {
                "n": int(mask.sum()),
                "persistence_mae": float(np.mean(np.abs(y - current))) if len(y) else None,
                "future_exceedance_rate": float(np.mean(y > SULFUR_LIMIT)) if len(y) else None,
                "transition_count": int(((current <= SULFUR_LIMIT) & (y > SULFUR_LIMIT)).sum()),
            }
        result[period] = period_rows
    return result


def telemetry_residual_screen(
    data: PreparedData,
    dataset: SupervisedDataset,
    *,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """Rank train-only associations and expose whether their sign survives later periods."""
    frame = dataset.frame.reset_index(drop=True)
    split = make_outer_split(frame)
    queries = frame.loc[:, ["as_of", "y", dataset.baseline_feature]].copy()
    queries["as_of"] = pd.to_datetime(queries["as_of"], utc=True)
    telemetry = data.telemetry.copy()
    telemetry["timestamp"] = pd.to_datetime(telemetry["timestamp"], utc=True)
    signal_names = tuple(
        name for name in data.feature_order if name in telemetry.columns and name != "timestamp"
    )
    matched = pd.merge_asof(
        queries.sort_values("as_of", kind="stable"),
        telemetry.loc[:, ["timestamp", *signal_names]].sort_values("timestamp", kind="stable"),
        left_on="as_of",
        right_on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(minutes=20),
    ).sort_index(kind="stable")
    residual = pd.to_numeric(matched["y"], errors="coerce") - pd.to_numeric(
        matched[dataset.baseline_feature], errors="coerce"
    )

    def correlation(signal: str, indices: np.ndarray) -> tuple[int, float | None]:
        x = pd.to_numeric(matched.iloc[indices][signal], errors="coerce")
        y = residual.iloc[indices]
        valid = x.notna() & y.notna()
        value = (
            x[valid].corr(y[valid])
            if valid.sum() >= 100 and x[valid].std() > 0.0 and y[valid].std() > 0.0
            else np.nan
        )
        return int(valid.sum()), float(value) if pd.notna(value) else None

    rows: list[dict[str, Any]] = []
    for signal in signal_names:
        train_n, train_corr = correlation(signal, split.train)
        if train_corr is None:
            continue
        validation_n, validation_corr = correlation(signal, split.validation)
        test_n, test_corr = correlation(signal, split.test)
        rows.append(
            {
                "signal": signal,
                "train_n": train_n,
                "train_correlation": train_corr,
                "validation_n": validation_n,
                "validation_correlation": validation_corr,
                "test_n": test_n,
                "test_correlation": test_corr,
                "stable_sign": bool(
                    validation_corr is not None
                    and test_corr is not None
                    and np.sign(train_corr) == np.sign(validation_corr) == np.sign(test_corr)
                ),
            }
        )
    rows.sort(key=lambda row: abs(float(row["train_correlation"])), reverse=True)
    return rows[:top_n]


def engineered_feature_screen(
    dataset: SupervisedDataset, *, top_n: int = 20, min_samples: int = 100
) -> list[dict[str, Any]]:
    """Check whether existing causal lags explain change beyond persistence."""
    frame = dataset.frame.reset_index(drop=True)
    split = make_outer_split(frame)
    residual = pd.to_numeric(frame["y"], errors="coerce") - pd.to_numeric(
        frame[dataset.baseline_feature], errors="coerce"
    )

    def correlation(feature: str, indices: np.ndarray) -> tuple[int, float | None]:
        x = pd.to_numeric(frame.iloc[indices][feature], errors="coerce")
        y = residual.iloc[indices]
        valid = x.notna() & y.notna()
        value = (
            x[valid].corr(y[valid])
            if valid.sum() >= min_samples and x[valid].std() > 0.0 and y[valid].std() > 0.0
            else np.nan
        )
        return int(valid.sum()), float(value) if pd.notna(value) else None

    result: list[dict[str, Any]] = []
    for feature in dataset.feature_names:
        train_n, train_corr = correlation(feature, split.train)
        if train_corr is None:
            continue
        validation_n, validation_corr = correlation(feature, split.validation)
        test_n, test_corr = correlation(feature, split.test)
        result.append(
            {
                "feature": feature,
                "train_n": train_n,
                "train_correlation": train_corr,
                "validation_n": validation_n,
                "validation_correlation": validation_corr,
                "test_n": test_n,
                "test_correlation": test_corr,
                "stable_sign": bool(
                    validation_corr is not None
                    and test_corr is not None
                    and np.sign(train_corr) == np.sign(validation_corr) == np.sign(test_corr)
                ),
            }
        )
    result.sort(key=lambda row: abs(float(row["train_correlation"])), reverse=True)
    return result[:top_n]


def build_diagnostic_report(
    data: PreparedData, dataset: SupervisedDataset, *, top_n: int = 20
) -> dict[str, Any]:
    """Build one JSON-serializable research report without changing model capability."""
    return {
        "dataset_id": data.manifest.dataset_id,
        "target": SULFUR_SIGNAL,
        "pak_lims_alignment": pak_lims_alignment(data),
        "temporal_drift": temporal_drift(dataset),
        "regime_diagnostics": regime_diagnostics(dataset),
        "engineered_feature_residual_screen": engineered_feature_screen(dataset, top_n=top_n),
        "telemetry_residual_screen": telemetry_residual_screen(data, dataset, top_n=top_n),
        "limitations": [
            "confirmed LIMS outliers remain in prepared data and provenance",
            "univariate association is not causality and cannot enable actions",
            "test correlations are diagnostic only and are not model-selection inputs",
        ],
    }


__all__ = [
    "build_diagnostic_report",
    "CONFIRMED_LIMS_OUTLIER_IDS",
    "engineered_feature_screen",
    "exclude_confirmed_lims_outliers",
    "pak_lims_alignment",
    "regime_diagnostics",
    "telemetry_residual_screen",
    "temporal_drift",
]
