"""Episode-aware multi-horizon sulfur forecasting without action claims."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, cast

import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from source.contracts import SourceKind, Validity
from source.data.prepare import PreparedData
from source.ml.artifacts import ModelBundle, feature_schema_hash, save_model
from source.ml.features import SupervisedDataset, build_supervised_dataset
from source.ml.safety import (
    FALSE_ALARM_BUDGET,
    SULFUR_LIMIT,
    AlarmPolicy,
    JointApplicabilityModel,
    PlattCalibrator,
    _fit_calibrator,
    fit_joint_applicability,
)

HORIZONS = (10, 20, 30, 60)
WINDOWS = (30, 60, 180)
REGIME_LABELS = ("below_8", "8_to_10", "above_10")
ABLATED_TELEMETRY_GROUPS = {
    "pak_only": (),
    "ht_context": ("ht:P8", "ht:T11", "ht:F19"),
    "k2_state": ("avt:F65", "avt:T20", "avt:T33", "avt:P21", "avt:P22", "avt:P23", "avt:P67"),
    "k2_circulation": (
        "avt:F14",
        "avt:T13",
        "avt:T18",
        "avt:F12",
        "avt:T17",
        "avt:F64",
        "avt:T11",
        "avt:T15",
    ),
    "diesel_cut": ("avt:T66", "avt:F28", "avt:F32", "avt:T71", "avt:F30", "avt:W70"),
}


@dataclass(frozen=True)
class EpisodeDataset:
    """Leakage-safe v2 frame and its exact serving feature order."""

    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    baseline_feature: str


@dataclass(frozen=True)
class EpisodeSafetyPredictor:
    """Serializable point, upper and multi-horizon risk predictor."""

    baseline_feature: str
    delta_estimator: Any
    upper_delta_estimator: Any
    risk_estimators: Mapping[int, Any]
    calibrators: Mapping[int, PlattCalibrator]
    alarm_policy: AlarmPolicy
    applicability: JointApplicabilityModel
    upper_delta_shift: float = 0.0

    def predict_delta(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.delta_estimator.predict(features), dtype=float)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        current = pd.to_numeric(features[self.baseline_feature], errors="coerce").to_numpy(
            dtype=float
        )
        return cast(np.ndarray, current + self.predict_delta(features))

    def predict_upper(self, features: pd.DataFrame) -> np.ndarray:
        current = pd.to_numeric(features[self.baseline_feature], errors="coerce").to_numpy(
            dtype=float
        )
        upper = current + np.asarray(self.upper_delta_estimator.predict(features), dtype=float)
        upper += float(self.upper_delta_shift)
        return np.maximum(self.predict(features), upper)

    def predict_horizon_probabilities(self, features: pd.DataFrame) -> dict[int, np.ndarray]:
        result: dict[int, np.ndarray] = {}
        previous = np.zeros(len(features), dtype=float)
        for horizon in HORIZONS:
            raw = np.asarray(
                self.risk_estimators[horizon].predict_proba(features)[:, 1], dtype=float
            )
            calibrated = self.calibrators[horizon].predict(raw)
            previous = np.maximum(previous, calibrated)
            result[horizon] = previous.copy()
        return result

    def predict_exceedance_probability(self, features: pd.DataFrame) -> np.ndarray:
        return self.predict_horizon_probabilities(features)[60]

    def predict_alarm(self, features: pd.DataFrame) -> np.ndarray:
        current = pd.to_numeric(features[self.baseline_feature], errors="coerce").to_numpy(
            dtype=float
        )
        return (current > SULFUR_LIMIT) | (
            self.predict_exceedance_probability(features) >= self.alarm_policy.threshold
        )

    def predict_v2(self, features: pd.DataFrame) -> list[dict[str, object]]:
        point = self.predict(features)
        upper = self.predict_upper(features)
        delta = self.predict_delta(features)
        horizons = self.predict_horizon_probabilities(features)
        alarm = self.predict_alarm(features)
        current = pd.to_numeric(features[self.baseline_feature], errors="coerce").to_numpy(
            dtype=float
        )
        rows: list[dict[str, object]] = []
        for position in range(len(features)):
            applicable = self.applicability.assess(features.iloc[[position]])
            reasons: list[str] = []
            if current[position] > SULFUR_LIMIT:
                reasons.append("CURRENT_SULFUR_LIMIT")
            if alarm[position] and current[position] <= SULFUR_LIMIT:
                reasons.append("EXCEEDANCE_PROBABILITY_THRESHOLD")
            if applicable.reason_code is not None:
                reasons.append(applicable.reason_code)
            rows.append(
                {
                    "point": float(point[position]),
                    "upper": float(upper[position]),
                    "exceedance_probability": float(horizons[60][position]),
                    "alarm": bool(alarm[position]),
                    "horizon_probabilities": {
                        str(horizon): float(values[position])
                        for horizon, values in horizons.items()
                    },
                    "crossing_probability_60m": float(horizons[60][position]),
                    "event_alarm": bool(alarm[position]),
                    "predicted_delta": float(delta[position]),
                    "point_60m": float(point[position]),
                    "upper_60m": float(upper[position]),
                    "applicable": applicable.available,
                    "reason_codes": tuple(dict.fromkeys(reasons)),
                }
            )
        return rows

    def check_applicability(self, features: pd.DataFrame) -> object:
        return self.applicability.assess(features)


@dataclass(frozen=True)
class EpisodeFitResult:
    predictor: EpisodeSafetyPredictor
    report: dict[str, Any]


def ablate_episode_features(
    data: PreparedData, *, groups: tuple[str, ...] | None = None, seed: int = 42
) -> dict[str, Any]:
    """Compare fixed PAK/HT/AVT groups on 2024 folds without consulting audit 2026."""
    selected_groups = tuple(ABLATED_TELEMETRY_GROUPS) if groups is None else groups
    unknown = set(selected_groups).difference(ABLATED_TELEMETRY_GROUPS)
    if not selected_groups or unknown:
        raise ValueError(f"unknown or empty ablation groups: {sorted(unknown)}")
    all_signals = tuple(
        dict.fromkeys(
            signal for group in selected_groups for signal in ABLATED_TELEMETRY_GROUPS[group]
        )
    )
    base = build_supervised_dataset(
        data,
        target_signal_id="ht:2:Mg.Sulfur",
        target_source=SourceKind.PAK,
        feature_source=SourceKind.PAK,
        horizon_minutes=60,
        telemetry_signals=all_signals,
    )
    full_dataset = build_episode_dataset(data, base)
    telemetry_prefixes = tuple(f"{signal}__" for signal in all_signals)
    common_features = tuple(
        name for name in base.feature_names if not name.startswith(telemetry_prefixes)
    )
    candidates: dict[str, dict[str, Any]] = {}
    for group in selected_groups:
        signals = ABLATED_TELEMETRY_GROUPS[group]
        selected_features = tuple(
            name
            for name in base.feature_names
            if name in common_features or any(name.startswith(f"{signal}__") for signal in signals)
        )
        frame = full_dataset.frame.copy()
        frame["feature_missing_fraction"] = (
            frame.loc[:, list(selected_features)].isna().mean(axis=1)
        )
        added_features = tuple(
            name for name in full_dataset.feature_names if name.startswith("pak_")
        )
        dataset = EpisodeDataset(
            frame,
            tuple(dict.fromkeys([*selected_features, *added_features])),
            full_dataset.baseline_feature,
        )
        family_results: dict[str, dict[str, Any]] = {}
        for family in ("hgb", "lightgbm"):
            frame, raw, delta, _ = _rolling_predictions(
                dataset,
                family,
                "2024-01-01",
                "2025-01-01",
                seed,
                include_upper=False,
                estimator_factory=_ablation_estimator,
                max_train_rows=20_000,
                risk_horizons=(60,),
            )
            probabilities = raw[60]
            policy, metrics = select_event_threshold(frame, probabilities)
            point = frame["baseline"].to_numpy(dtype=float) + delta
            family_results[family] = {
                "event_fnr": metrics["event_false_negative_rate"],
                "event_fpr": metrics["event_false_positive_rate"],
                "row_fpr": metrics["row_false_positive_rate"],
                "brier": float(brier_score_loss(frame["crossing_60m"], probabilities)),
                "mae": float(np.mean(np.abs(frame["y_60m"].to_numpy(dtype=float) - point))),
                "threshold": policy.threshold,
            }
        eligible = [
            family
            for family, result in family_results.items()
            if result["event_fpr"] is not None
            and result["event_fpr"] <= FALSE_ALARM_BUDGET
            and result["row_fpr"] is not None
            and result["row_fpr"] <= FALSE_ALARM_BUDGET
        ]
        selected = (
            min(
                eligible,
                key=lambda family: (
                    float(family_results[family]["event_fnr"]),
                    float(family_results[family]["brier"]),
                    float(family_results[family]["mae"]),
                    0 if family == "hgb" else 1,
                ),
            )
            if eligible
            else None
        )
        candidates[group] = {
            "telemetry_signals": list(signals),
            "feature_count": len(dataset.feature_names),
            "families": family_results,
            "selected_family": selected,
            "selected_metrics": family_results.get(selected) if selected is not None else None,
        }
    return {
        "selection_period": "2024 rolling-origin monthly folds",
        "audit_2026_used": False,
        "false_alarm_budget": FALSE_ALARM_BUDGET,
        "training_budget": {
            "max_train_rows_per_fold": 20_000,
            "risk_horizons": [60],
            "hgb_max_iter": 8,
            "hgb_max_leaf_nodes": 7,
            "lightgbm_n_estimators": 20,
            "lightgbm_num_leaves": 7,
        },
        "candidates": candidates,
        "promotion_eligible": False,
        "next_gate": "repeat the chosen ablation on 2025 before any shadow artifact",
    }


def _pak_rows(data: PreparedData, signal_id: str) -> pd.DataFrame:
    quality = data.quality
    values = pd.to_numeric(quality["value"], errors="coerce")
    selected = quality[
        quality["signal_id"].astype(str).eq(signal_id)
        & quality["source"].map(lambda value: str(getattr(value, "value", value))).eq("pak")
        & quality["validity"]
        .map(lambda value: str(getattr(value, "value", value)))
        .eq(Validity.VALID.value)
        & values.notna()
        & np.isfinite(values)
    ].copy()
    selected["value"] = values.loc[selected.index].astype(float)
    selected["measured_at"] = pd.to_datetime(selected["measured_at"], utc=True)
    return cast(
        pd.DataFrame, selected.sort_values("measured_at", kind="stable").reset_index(drop=True)
    )


def _exact_future_values(
    queries: pd.Series, pak: pd.DataFrame, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    left = pd.DataFrame(
        {
            "position": np.arange(len(queries)),
            "query": pd.to_datetime(queries, utc=True) + pd.Timedelta(minutes=horizon),
        }
    ).sort_values("query", kind="stable")
    right = pak.loc[:, ["measured_at", "value", "episode_id"]]
    merged = pd.merge_asof(
        left,
        right,
        left_on="query",
        right_on="measured_at",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=1),
    ).sort_values("position", kind="stable")
    return merged["value"].to_numpy(dtype=float), merged["episode_id"].to_numpy(dtype=float)


def _rolling_features(queries: pd.Series, pak: pd.DataFrame) -> pd.DataFrame:
    query_index = pd.DatetimeIndex(pd.to_datetime(queries, utc=True))
    series = pd.Series(pak["value"].to_numpy(dtype=float), index=pak["measured_at"])
    timeline = series.index.union(query_index).sort_values()
    expanded = series.reindex(timeline)
    columns: dict[str, np.ndarray] = {}
    for window in WINDOWS:
        rolling = expanded.rolling(f"{window}min", closed="both", min_periods=1)
        minimum = rolling.min().reindex(query_index).to_numpy(dtype=float)
        maximum = rolling.max().reindex(query_index).to_numpy(dtype=float)
        columns[f"pak_min_{window}m"] = minimum
        columns[f"pak_max_{window}m"] = maximum
        columns[f"pak_range_{window}m"] = maximum - minimum
    return pd.DataFrame(columns)


def _state_age_features(queries: pd.Series, pak: pd.DataFrame) -> pd.DataFrame:
    query = pd.DatetimeIndex(pd.to_datetime(queries, utc=True))
    times = pd.DatetimeIndex(pak["measured_at"])
    values = pak["value"].to_numpy(dtype=float)
    result: dict[str, np.ndarray] = {}
    for threshold in (8.0, 9.0, 10.0):
        state = values >= threshold
        crossing = np.r_[True, state[1:] != state[:-1]]
        events = pd.DataFrame({"event_at": times[crossing]}).sort_values("event_at")
        merged = pd.merge_asof(
            pd.DataFrame({"position": np.arange(len(query)), "query": query}).sort_values("query"),
            events,
            left_on="query",
            right_on="event_at",
            direction="backward",
        ).sort_values("position")
        result[f"pak_minutes_since_crossing_{int(threshold)}"] = (
            (merged["query"] - merged["event_at"]).dt.total_seconds().div(60).to_numpy(dtype=float)
        )
    bands = pd.cut(values, [-np.inf, 8.0, 10.0, np.inf], labels=False, right=False)
    changed = np.r_[True, np.asarray(bands[1:]) != np.asarray(bands[:-1])]
    events = pd.DataFrame({"event_at": times[changed]}).sort_values("event_at")
    merged = pd.merge_asof(
        pd.DataFrame({"position": np.arange(len(query)), "query": query}).sort_values("query"),
        events,
        left_on="query",
        right_on="event_at",
        direction="backward",
    ).sort_values("position")
    result["pak_regime_duration_minutes"] = (
        (merged["query"] - merged["event_at"]).dt.total_seconds().div(60).to_numpy(dtype=float)
    )
    return pd.DataFrame(result)


def build_episode_dataset(data: PreparedData, base: SupervisedDataset) -> EpisodeDataset:
    """Add causal trajectory features and future labels to the 60-minute PAK table."""
    if base.target_source is not SourceKind.PAK or base.feature_source is not SourceKind.PAK:
        raise ValueError("episode dataset requires PAK targets and PAK history")
    frame = base.frame.reset_index(drop=True).copy()
    pak = _pak_rows(data, base.target_signal_id)
    exceedance = pak["value"].to_numpy(dtype=float) > SULFUR_LIMIT
    starts = exceedance & np.r_[True, ~exceedance[:-1]]
    pak["episode_id"] = np.where(exceedance, np.cumsum(starts), np.nan)

    exact_crossings: dict[int, np.ndarray] = {}
    future_episodes: dict[int, np.ndarray] = {}
    future_values: dict[int, np.ndarray] = {}
    future_steps = tuple(range(10, max(HORIZONS) + 1, 10))
    for step in future_steps:
        values, episode_ids = _exact_future_values(frame["as_of"], pak, step)
        future_values[step] = values
        exact_crossings[step] = values > SULFUR_LIMIT
        future_episodes[step] = episode_ids
    for horizon in HORIZONS:
        frame[f"y_{horizon}m"] = future_values[horizon]
        frame[f"crossing_{horizon}m"] = np.logical_or.reduce(
            [exact_crossings[step] for step in future_steps if step <= horizon]
        )
    frame["crossing_60m"] = frame["crossing_60m"].fillna(False)
    episode_id = np.full(len(frame), np.nan)
    for step in future_steps:
        use = np.isnan(episode_id) & exact_crossings[step]
        episode_id[use] = future_episodes[step][use]
    frame["event_id"] = episode_id
    episode_starts = pak.loc[starts, ["episode_id", "measured_at"]].set_index("episode_id")
    frame["event_start_at"] = frame["event_id"].map(episode_starts["measured_at"])
    frame["delta_60m"] = pd.to_numeric(frame["y_60m"], errors="coerce") - pd.to_numeric(
        frame[base.baseline_feature], errors="coerce"
    )

    derived = _rolling_features(frame["as_of"], pak)
    state_ages = _state_age_features(frame["as_of"], pak)
    current = pd.to_numeric(frame[base.baseline_feature], errors="coerce")
    for window in WINDOWS:
        lag_name = f"{base.target_signal_id}__pak_lag_{window}m"
        if lag_name in frame:
            frame[f"pak_slope_{window}m"] = (current - frame[lag_name]) / float(window)
    if "pak_slope_30m" in frame and "pak_slope_60m" in frame:
        frame["pak_acceleration_30_60m"] = frame["pak_slope_30m"] - frame["pak_slope_60m"]
    frame = pd.concat([frame, derived, state_ages], axis="columns")
    original_features = list(base.feature_names)
    frame["feature_missing_fraction"] = frame[original_features].isna().mean(axis=1)
    added = [
        name
        for name in frame.columns
        if name.startswith("pak_") or name == "feature_missing_fraction"
    ]
    feature_names = tuple(dict.fromkeys([*original_features, *added]))
    return EpisodeDataset(frame, feature_names, base.baseline_feature)


def episode_sample_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give every positive episode and calendar month comparable training mass."""
    positive = frame["crossing_60m"].fillna(False).to_numpy(dtype=bool)
    weights = np.ones(len(frame), dtype=float)
    episode_counts = frame.loc[positive, "event_id"].value_counts()
    if not episode_counts.empty:
        weights[positive] = frame.loc[positive, "event_id"].map(1.0 / episode_counts).to_numpy()
        weights[positive] *= positive.sum() / weights[positive].sum()
    months = pd.to_datetime(frame["as_of"], utc=True).dt.strftime("%Y-%m")
    month_mass = (
        pd.Series(weights).groupby(months.to_numpy()).transform("sum").to_numpy(dtype=float)
    )
    weights /= month_mass
    weights *= len(weights) / weights.sum()
    return weights


def rolling_month_folds(
    frame: pd.DataFrame, validation_start: str, validation_end: str
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return monthly expanding folds purged by target publication time."""
    as_of = pd.to_datetime(frame["as_of"], utc=True)
    available = pd.to_datetime(frame["target_available_at"], utc=True)
    starts = pd.date_range(validation_start, validation_end, freq="MS", inclusive="left", tz="UTC")
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for start in starts:
        end = start + pd.offsets.MonthBegin(1)
        train = (as_of < start) & (available < start)
        validation = (as_of >= start) & (as_of < end)
        if train.any() and validation.any():
            folds.append((np.flatnonzero(train), np.flatnonzero(validation)))
    return tuple(folds)


def _event_lead_window(frame: pd.DataFrame) -> np.ndarray:
    """Return rows 10–60 minutes before the first crossing of their episode."""
    if "event_start_at" not in frame:
        # Keep small synthetic fixtures/backward-compatible callers usable. Real v2
        # datasets always carry event_start_at from build_episode_dataset().
        return np.ones(len(frame), dtype=bool)
    as_of = pd.to_datetime(frame["as_of"], utc=True)
    event_start = pd.to_datetime(frame["event_start_at"], utc=True, errors="coerce")
    return (
        event_start.notna()
        & (as_of >= event_start - pd.Timedelta(minutes=60))
        & (as_of <= event_start - pd.Timedelta(minutes=10))
    ).to_numpy(dtype=bool)


def event_metrics(frame: pd.DataFrame, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    """Evaluate unique exceedance episodes and ordinary false-alarm opportunities."""
    actual = frame["crossing_60m"].fillna(False).to_numpy(dtype=bool)
    current = pd.to_numeric(frame["baseline"], errors="coerce").to_numpy(dtype=float)
    alarm = (probability >= threshold) | (current > SULFUR_LIMIT)
    event_alarm = alarm & _event_lead_window(frame)
    event_ids = frame.loc[actual, "event_id"].dropna().unique()
    detected = sum(
        bool(event_alarm[frame["event_id"].eq(event_id).to_numpy()].any()) for event_id in event_ids
    )
    eligible_negative = (~actual) & (current <= SULFUR_LIMIT) & np.isfinite(current)
    false_positive_rate = (
        float(alarm[eligible_negative].mean()) if eligible_negative.any() else None
    )
    opportunity_hour = pd.to_datetime(frame["as_of"], utc=True).dt.floor("60min")
    negative_opportunities = pd.DataFrame(
        {
            "opportunity": opportunity_hour[eligible_negative].to_numpy(),
            "alarm": alarm[eligible_negative],
        }
    )
    opportunity_alarm = (
        negative_opportunities.groupby("opportunity", sort=False)["alarm"].any()
        if len(negative_opportunities)
        else pd.Series(dtype=bool)
    )
    return {
        "event_count": int(len(event_ids)),
        "detected_events": int(detected),
        "event_false_negative_rate": 1.0 - detected / len(event_ids) if len(event_ids) else None,
        "event_false_positive_rate": float(opportunity_alarm.mean())
        if len(opportunity_alarm)
        else None,
        "false_alarm_opportunities": int(opportunity_alarm.sum()),
        "negative_opportunities": int(len(opportunity_alarm)),
        "row_false_positive_rate": false_positive_rate,
        "row_false_negatives": int((actual & ~alarm).sum()),
        "row_false_negative_rate": float((actual & ~alarm).sum() / actual.sum())
        if actual.any()
        else None,
        "brier": float(brier_score_loss(actual, probability)),
        "average_precision": float(average_precision_score(actual, probability))
        if actual.any() and (~actual).any()
        else None,
    }


def select_event_threshold(
    frame: pd.DataFrame,
    probability: np.ndarray,
    false_alarm_budget: float = FALSE_ALARM_BUDGET,
) -> tuple[AlarmPolicy, dict[str, Any]]:
    """Minimize missed episodes under the row-level false-alarm budget."""
    actual = frame["crossing_60m"].fillna(False).to_numpy(dtype=bool)
    current = pd.to_numeric(frame["baseline"], errors="coerce").to_numpy(dtype=float)
    negatives = probability[(~actual) & (current <= SULFUR_LIMIT) & np.isfinite(current)]
    if not len(negatives):
        raise ValueError("threshold selection needs eligible negative rows")
    allowed = int(np.floor(false_alarm_budget * len(negatives)))
    if allowed >= len(negatives):
        threshold = 0.0
    else:
        boundary = np.sort(negatives)[len(negatives) - allowed - 1]
        threshold = float(np.nextafter(boundary, np.inf))
    if threshold > 1.0:
        raise ValueError("no event threshold satisfies false-alarm budget")
    metrics = event_metrics(frame, probability, threshold)
    if (
        metrics["event_false_positive_rate"] is not None
        and metrics["event_false_positive_rate"] > false_alarm_budget
    ):
        candidates = np.unique(probability[np.isfinite(probability)])
        low = int(np.searchsorted(candidates, threshold, side="left"))
        high = len(candidates)
        while low < high:
            middle = (low + high) // 2
            candidate = float(candidates[middle])
            candidate_metrics = event_metrics(frame, probability, candidate)
            event_fpr = candidate_metrics["event_false_positive_rate"]
            if event_fpr is not None and event_fpr <= false_alarm_budget:
                high = middle
            else:
                low = middle + 1
        if low == len(candidates):
            threshold = 1.0
        else:
            threshold = float(candidates[low])
        metrics = event_metrics(frame, probability, threshold)
    return AlarmPolicy(threshold, false_alarm_budget, 0.10), metrics


def _estimator(family: str, task: str, seed: int) -> Any:
    if family == "hgb":
        if task == "classifier":
            return HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=100,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=seed,
            )
        loss = "quantile" if task == "upper" else "squared_error"
        kwargs = {"quantile": 0.95} if task == "upper" else {}
        return HistGradientBoostingRegressor(
            loss=loss, max_iter=100, max_leaf_nodes=15, random_state=seed, **kwargs
        )
    if family == "lightgbm":
        if task == "classifier":
            model: Any = LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=15, random_state=seed, verbosity=-1
            )
        else:
            objective = "quantile" if task == "upper" else "regression_l1"
            model = LGBMRegressor(
                objective=objective,
                alpha=0.95 if task == "upper" else 0.9,
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=15,
                random_state=seed,
                verbosity=-1,
            )
        return Pipeline(
            (("to_array", FunctionTransformer(np.asarray, validate=False)), ("model", model))
        )
    raise ValueError(f"unknown v2 family: {family}")


def _ablation_estimator(family: str, task: str, seed: int) -> Any:
    """Use a fixed smaller budget for exploratory feature screening only."""
    estimator = _estimator(family, task, seed)
    if family == "hgb":
        estimator.set_params(max_iter=8, max_leaf_nodes=7)
    else:
        estimator.set_params(model__n_estimators=20, model__num_leaves=7)
    return estimator


def _fit(estimator: Any, x: pd.DataFrame, y: np.ndarray, weights: np.ndarray, family: str) -> Any:
    valid = np.isfinite(y)
    if not valid.any():
        raise ValueError("v2 fit period has no finite targets")
    argument = (
        {"model__sample_weight": weights[valid]}
        if family == "lightgbm"
        else {"sample_weight": weights[valid]}
    )
    return estimator.fit(x.loc[valid], y[valid], **argument)


def _rolling_predictions(
    dataset: EpisodeDataset,
    family: str,
    start: str,
    end: str,
    seed: int,
    *,
    include_regression: bool = True,
    include_upper: bool = True,
    estimator_factory: Callable[[str, str, int], Any] = _estimator,
    max_train_rows: int | None = None,
    risk_horizons: tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], np.ndarray, np.ndarray]:
    frame = dataset.frame
    pieces: list[pd.DataFrame] = []
    horizons = HORIZONS if risk_horizons is None else tuple(risk_horizons)
    if not horizons or any(horizon not in HORIZONS for horizon in horizons):
        raise ValueError(f"risk_horizons must be a non-empty subset of {HORIZONS}")
    risk_parts: dict[int, list[np.ndarray]] = {horizon: [] for horizon in horizons}
    point_parts: list[np.ndarray] = []
    upper_parts: list[np.ndarray] = []
    for fold_number, (train_indices, validation_indices) in enumerate(
        rolling_month_folds(frame, start, end)
    ):
        if max_train_rows is not None:
            train_indices = _cap_training_rows(
                frame, train_indices, max_train_rows, seed=seed, fold_number=fold_number
            )
        train = frame.iloc[train_indices]
        validation = frame.iloc[validation_indices]
        features = list(dataset.feature_names)
        weights = episode_sample_weights(train)
        for horizon in horizons:
            estimator = _fit(
                estimator_factory(family, "classifier", seed),
                train.loc[:, features],
                train[f"crossing_{horizon}m"].to_numpy(dtype=bool),
                weights,
                family,
            )
            risk_parts[horizon].append(
                np.asarray(estimator.predict_proba(validation.loc[:, features])[:, 1], dtype=float)
            )
        if include_regression:
            delta = _fit(
                estimator_factory(family, "delta", seed),
                train.loc[:, features],
                train["delta_60m"].to_numpy(dtype=float),
                weights,
                family,
            )
            point_parts.append(np.asarray(delta.predict(validation.loc[:, features]), dtype=float))
        if include_upper:
            upper = _fit(
                estimator_factory(family, "upper", seed),
                train.loc[:, features],
                train["delta_60m"].to_numpy(dtype=float),
                weights,
                family,
            )
            upper_parts.append(np.asarray(upper.predict(validation.loc[:, features]), dtype=float))
        pieces.append(validation)
    if not pieces:
        raise ValueError("rolling backtest produced no folds")
    return (
        pd.concat(pieces, ignore_index=True),
        {horizon: np.concatenate(parts) for horizon, parts in risk_parts.items()},
        np.concatenate(point_parts) if point_parts else np.asarray([], dtype=float),
        np.concatenate(upper_parts) if upper_parts else np.asarray([], dtype=float),
    )


def _cap_training_rows(
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    max_rows: int,
    *,
    seed: int,
    fold_number: int,
) -> np.ndarray:
    """Bound exploratory fit cost while retaining all positive event rows when possible."""
    if max_rows <= 0 or len(train_indices) <= max_rows:
        return train_indices
    train = frame.iloc[train_indices]
    positive = train.loc[:, [f"crossing_{horizon}m" for horizon in HORIZONS]].any(axis=1)
    positive_indices = train_indices[positive.to_numpy()]
    negative_indices = train_indices[~positive.to_numpy()]
    rng = np.random.default_rng(seed + fold_number)
    if len(positive_indices) >= max_rows:
        selected = rng.choice(positive_indices, size=max_rows, replace=False)
    else:
        negative_count = max_rows - len(positive_indices)
        sampled_negative = rng.choice(
            negative_indices, size=min(negative_count, len(negative_indices)), replace=False
        )
        selected = np.concatenate([positive_indices, sampled_negative])
    return np.sort(selected)


def _monotone_probabilities(probabilities: Mapping[int, np.ndarray]) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    previous = np.zeros_like(next(iter(probabilities.values())))
    for horizon in HORIZONS:
        previous = np.maximum(previous, probabilities[horizon])
        result[horizon] = previous.copy()
    return result


def _regime_report(
    frame: pd.DataFrame, probability: np.ndarray, threshold: float
) -> dict[str, Any]:
    current = pd.to_numeric(frame["baseline"], errors="coerce")
    bands = pd.cut(current, [-np.inf, 8.0, 10.0, np.inf], labels=REGIME_LABELS, right=False)
    return {
        label: event_metrics(
            frame.loc[bands.eq(label)].reset_index(drop=True),
            probability[bands.eq(label)],
            threshold,
        )
        for label in REGIME_LABELS
        if bands.eq(label).any()
    }


def _monthly_report(
    frame: pd.DataFrame,
    probability: np.ndarray,
    threshold: float,
    *,
    actual: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> dict[str, Any]:
    month = pd.to_datetime(frame["as_of"], utc=True).dt.strftime("%Y-%m")
    result: dict[str, Any] = {}
    for value in sorted(month.unique()):
        selected = month.eq(value).to_numpy()
        metrics = event_metrics(
            frame.loc[month.eq(value)].reset_index(drop=True),
            probability[month.eq(value)],
            threshold,
        )
        if actual is not None and upper is not None:
            finite = np.isfinite(actual[selected]) & np.isfinite(upper[selected])
            metrics["upper_coverage"] = (
                float(np.mean(actual[selected][finite] <= upper[selected][finite]))
                if finite.any()
                else None
            )
            metrics.update(_upper_limit_metrics(actual[selected], upper[selected]))
        result[value] = metrics
    return result


def _upper_limit_metrics(actual: np.ndarray, upper: np.ndarray) -> dict[str, float | int | None]:
    """Measure misses and false alarms of the conservative ``upper > 10`` rule."""
    finite = np.isfinite(actual) & np.isfinite(upper)
    if not finite.any():
        return {
            "upper_limit_misses": 0,
            "upper_limit_miss_rate": None,
            "upper_limit_false_alarm_rate": None,
        }
    actual = actual[finite]
    upper = upper[finite]
    violation = actual > SULFUR_LIMIT
    upper_alarm = upper > SULFUR_LIMIT
    return {
        "upper_limit_misses": int((violation & ~upper_alarm).sum()),
        "upper_limit_miss_rate": float((violation & ~upper_alarm).sum() / violation.sum())
        if violation.any()
        else None,
        "upper_limit_false_alarm_rate": float((~violation & upper_alarm).sum() / (~violation).sum())
        if (~violation).any()
        else None,
    }


def _lead_time_report(
    frame: pd.DataFrame, probabilities: Mapping[int, np.ndarray], threshold: float
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in HORIZONS:
        view = frame.copy()
        view["crossing_60m"] = view[f"crossing_{horizon}m"]
        result[f"{horizon}m"] = event_metrics(view, probabilities[horizon], threshold)
    return result


def _bootstrap_event_fnr(
    frame: pd.DataFrame, probability: np.ndarray, threshold: float, seed: int
) -> dict[str, float] | None:
    event_ids = frame.loc[frame["crossing_60m"], "event_id"].dropna().unique()
    if not len(event_ids):
        return None
    current = pd.to_numeric(frame["baseline"], errors="coerce").to_numpy(dtype=float)
    alarm = (probability >= threshold) | (current > SULFUR_LIMIT)
    event_alarm = alarm & _event_lead_window(frame)
    detected = np.asarray(
        [event_alarm[frame["event_id"].eq(event_id).to_numpy()].any() for event_id in event_ids],
        dtype=bool,
    )
    rng = np.random.default_rng(seed)
    samples = np.asarray(
        [1.0 - rng.choice(detected, size=len(detected), replace=True).mean() for _ in range(1000)]
    )
    return {
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
    }


def fit_episode_safety_model(dataset: EpisodeDataset, *, seed: int = 42) -> EpisodeFitResult:
    """Select on 2024, calibrate/threshold in 2025, and audit 2026 once fixed."""
    family_reports: dict[str, Any] = {}
    for family in ("hgb", "lightgbm"):
        frame_2024, raw, delta, _ = _rolling_predictions(
            dataset, family, "2024-01-01", "2025-01-01", seed, include_upper=False
        )
        probabilities = _monotone_probabilities(raw)[60]
        policy, metrics = select_event_threshold(frame_2024, probabilities)
        point = pd.to_numeric(frame_2024["baseline"], errors="coerce").to_numpy(dtype=float) + delta
        actual = frame_2024["y_60m"].to_numpy(dtype=float)
        family_reports[family] = {
            "policy": policy,
            "metrics": metrics,
            "mae": float(np.mean(np.abs(actual - point))),
            "brier": float(brier_score_loss(frame_2024["crossing_60m"], probabilities)),
        }
    eligible_families = tuple(
        family
        for family, values in family_reports.items()
        if values["metrics"]["event_false_positive_rate"] is not None
        and values["metrics"]["event_false_positive_rate"] <= FALSE_ALARM_BUDGET
        and values["metrics"]["row_false_positive_rate"] is not None
        and values["metrics"]["row_false_positive_rate"] <= FALSE_ALARM_BUDGET
    )
    selection_pool = eligible_families or tuple(family_reports)
    selected_family = min(
        selection_pool,
        key=lambda family: (
            0 if family in eligible_families else 1,
            float(family_reports[family]["metrics"]["event_false_negative_rate"]),
            float(family_reports[family]["brier"]),
            float(family_reports[family]["mae"]),
            0 if family == "hgb" else 1,
        ),
    )

    calibration_frame, calibration_raw, _, calibration_upper_delta = _rolling_predictions(
        dataset,
        selected_family,
        "2025-01-01",
        "2025-07-01",
        seed,
        include_regression=False,
        include_upper=True,
    )
    calibrators: dict[int, PlattCalibrator] = {}
    for horizon in HORIZONS:
        labels = calibration_frame[f"crossing_{horizon}m"].to_numpy(dtype=bool)
        calibrators[horizon] = _fit_calibrator(calibration_raw[horizon], labels, seed)
    calibration_actual = calibration_frame["y_60m"].to_numpy(dtype=float)
    calibration_baseline = calibration_frame["baseline"].to_numpy(dtype=float)
    finite_upper = (
        np.isfinite(calibration_actual)
        & np.isfinite(calibration_baseline)
        & np.isfinite(calibration_upper_delta)
    )
    upper_delta_shift = (
        max(
            0.0,
            float(
                np.quantile(
                    calibration_actual[finite_upper]
                    - calibration_baseline[finite_upper]
                    - calibration_upper_delta[finite_upper],
                    0.95,
                )
            ),
        )
        if finite_upper.any()
        else 0.0
    )

    policy_frame, policy_raw, _, _ = _rolling_predictions(
        dataset,
        selected_family,
        "2025-07-01",
        "2026-01-01",
        seed,
        include_regression=False,
        include_upper=False,
    )
    calibrated = _monotone_probabilities(
        {horizon: calibrators[horizon].predict(policy_raw[horizon]) for horizon in HORIZONS}
    )
    policy, policy_metrics = select_event_threshold(policy_frame, calibrated[60])

    frame = dataset.frame
    as_of = pd.to_datetime(frame["as_of"], utc=True)
    available = pd.to_datetime(frame["target_available_at"], utc=True)
    fit_mask = (as_of < pd.Timestamp("2026-01-01", tz="UTC")) & (
        available < pd.Timestamp("2026-01-01", tz="UTC")
    )
    fit_frame = frame.loc[fit_mask]
    features = list(dataset.feature_names)
    weights = episode_sample_weights(fit_frame)
    risk_estimators: dict[int, Any] = {}
    for horizon in HORIZONS:
        risk_estimators[horizon] = _fit(
            _estimator(selected_family, "classifier", seed),
            fit_frame.loc[:, features],
            fit_frame[f"crossing_{horizon}m"].to_numpy(dtype=bool),
            weights,
            selected_family,
        )
    delta_estimator = _fit(
        _estimator(selected_family, "delta", seed),
        fit_frame.loc[:, features],
        fit_frame["delta_60m"].to_numpy(dtype=float),
        weights,
        selected_family,
    )
    upper_estimator = _fit(
        _estimator(selected_family, "upper", seed),
        fit_frame.loc[:, features],
        fit_frame["delta_60m"].to_numpy(dtype=float),
        weights,
        selected_family,
    )
    applicability = fit_joint_applicability(
        fit_frame.loc[:, features], required_features=(dataset.baseline_feature,)
    )
    predictor = EpisodeSafetyPredictor(
        dataset.baseline_feature,
        delta_estimator,
        upper_estimator,
        risk_estimators,
        calibrators,
        policy,
        applicability,
        upper_delta_shift,
    )

    audit = frame.loc[as_of >= pd.Timestamp("2026-01-01", tz="UTC")].reset_index(drop=True)
    audit_features = audit.loc[:, features]
    audit_probabilities = predictor.predict_horizon_probabilities(audit_features)
    audit_probability = audit_probabilities[60]
    audit_point = predictor.predict(audit_features)
    audit_upper = predictor.predict_upper(audit_features)
    actual = audit["y_60m"].to_numpy(dtype=float)
    finite_forecast = np.isfinite(actual) & np.isfinite(audit_point) & np.isfinite(audit_upper)
    audit_metrics = event_metrics(audit, audit_probability, policy.threshold)
    audit_metrics.update(
        {
            "forecast_rows": int(finite_forecast.sum()),
            "mae": float(np.mean(np.abs(actual[finite_forecast] - audit_point[finite_forecast])))
            if finite_forecast.any()
            else None,
            "upper_coverage": float(
                np.mean(actual[finite_forecast] <= audit_upper[finite_forecast])
            )
            if finite_forecast.any()
            else None,
            "event_fnr_bootstrap_95": _bootstrap_event_fnr(
                audit, audit_probability, policy.threshold, seed
            ),
        }
    )
    audit_metrics.update(_upper_limit_metrics(actual, audit_upper))
    monthly = _monthly_report(
        audit,
        audit_probability,
        policy.threshold,
        actual=actual,
        upper=audit_upper,
    )
    valid_months = [
        metrics for metrics in monthly.values() if metrics["event_false_negative_rate"] is not None
    ]
    report = {
        "schema_version": "1.2",
        "selected_family": selected_family,
        "selection_gate": {
            "eligible_families": list(eligible_families),
            "false_alarm_budget": FALSE_ALARM_BUDGET,
            "passed": bool(eligible_families),
        },
        "selection_2024": {
            family: {key: value for key, value in values.items() if key != "policy"}
            for family, values in family_reports.items()
        },
        "calibration_period": "2025-01-01/2025-07-01",
        "threshold_period": "2025-07-01/2026-01-01",
        "threshold_metrics": policy_metrics,
        "alarm_threshold": policy.threshold,
        "upper_delta_shift": upper_delta_shift,
        "audit_2026": audit_metrics,
        "audit_monthly": monthly,
        "audit_worst_month_event_fnr": max(
            float(metrics["event_false_negative_rate"]) for metrics in valid_months
        )
        if valid_months
        else None,
        "audit_regimes": _regime_report(audit, audit_probability, policy.threshold),
        "audit_lead_times": _lead_time_report(audit, audit_probabilities, policy.threshold),
        "statistical_gate_passed": bool(
            policy_metrics["event_false_negative_rate"] is not None
            and policy_metrics["event_false_negative_rate"] <= 0.10
            and policy_metrics["event_false_positive_rate"] is not None
            and policy_metrics["event_false_positive_rate"] <= FALSE_ALARM_BUDGET
            and policy_metrics["row_false_positive_rate"] is not None
            and policy_metrics["row_false_positive_rate"] <= FALSE_ALARM_BUDGET
        ),
        "audit_gate_passed": bool(
            audit_metrics["event_false_negative_rate"] is not None
            and audit_metrics["event_false_negative_rate"] <= 0.10
            and audit_metrics["event_false_positive_rate"] is not None
            and audit_metrics["event_false_positive_rate"] <= FALSE_ALARM_BUDGET
            and audit_metrics["row_false_positive_rate"] is not None
            and audit_metrics["row_false_positive_rate"] <= FALSE_ALARM_BUDGET
            and audit_metrics["upper_coverage"] is not None
            and audit_metrics["upper_coverage"] >= 0.95
            and audit_metrics["upper_limit_miss_rate"] is not None
            and audit_metrics["upper_limit_miss_rate"] <= 0.05
        ),
        "test_used_for_selection": False,
        "shadow_required": True,
        "shadow_minimum": "100 independent episodes or 3 complete months",
        "pak_only_ablation_completed": False,
        "promotion_blockers": [
            "new shadow period is required",
            "PAK-only ablation is required for disputed P8/T11/F19 semantics",
        ],
        "promotion_eligible": False,
        "supports_actions": False,
    }
    return EpisodeFitResult(predictor, report)


def save_episode_safety_model(
    directory: Path,
    data: PreparedData,
    base: SupervisedDataset,
    *,
    git_commit: str,
    seed: int = 42,
) -> tuple[ModelBundle, EpisodeFitResult]:
    """Persist a schema-1.2 shadow artifact; production eligibility stays false."""
    dataset = build_episode_dataset(data, base)
    fitted = fit_episode_safety_model(dataset, seed=seed)
    definition = {
        "horizon_minutes": 60,
        "horizons_minutes": list(HORIZONS),
        "windows_minutes": list(WINDOWS),
        "target": "delta_60m and any crossing within horizon",
        "episode_weighting": "inverse episode length and equal month mass",
        "upper_calibration": "nonnegative 0.95 residual shift on 2025-01/2025-07",
        "telemetry_signals": ["ht:P8", "ht:T11", "ht:F19"],
    }
    metadata = {
        "model_id": directory.name,
        "schema_version": "1.2",
        "model_type": f"{fitted.report['selected_family']}+episode_multi_horizon",
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
            "upper_delta_shift": fitted.report["upper_delta_shift"],
            "production_status": "shadow_only",
            "telemetry_semantics_status": "disputed_by_qa_2026_09_11",
            "pak_only_ablation_required": True,
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
        "false_alarm_budget": FALSE_ALARM_BUDGET,
        "calibration": {
            "method": "platt",
            "period": fitted.report["calibration_period"],
            "threshold_period": fitted.report["threshold_period"],
        },
        "metrics": {
            "pak": {
                "selection_2024": fitted.report["selection_2024"],
                "threshold": fitted.report["threshold_metrics"],
                "audit_2026": fitted.report["audit_2026"],
            },
            "lims": "separate_delayed_control_layer",
        },
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
            "purpose": "shadow-only PAK episode forecast",
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
    "EpisodeDataset",
    "EpisodeFitResult",
    "EpisodeSafetyPredictor",
    "HORIZONS",
    "build_episode_dataset",
    "episode_sample_weights",
    "event_metrics",
    "fit_episode_safety_model",
    "rolling_month_folds",
    "save_episode_safety_model",
    "select_event_threshold",
]
