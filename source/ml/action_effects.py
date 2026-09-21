"""Transparent setpoint consequences for Stage-3 model-only scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from source.contracts import CandidateAction, CandidateKind, ControlSpec, Validity
from source.data.prepare import PreparedData
from source.ml.controls import ActionEffectEvidence, JointControlDomain
from source.ml.safety import JointApplicabilityModel, fit_joint_applicability

HISTORICAL_ACTION_CONTROLS = ("ht:P8", "ht:F19")
HISTORICAL_CONTEXT_SIGNALS = ("ht:F26",)
ACTION_HORIZONS = (60, 120, 180)
ACTION_STATE_FEATURES = (
    "baseline_sulfur",
    "sulfur_slope_60m",
    "ht:P8",
    "ht:F19",
    "ht:F26",
)
ACTION_MODEL_FEATURES = (*ACTION_STATE_FEATURES, "delta_ht:P8", "delta_ht:F19")


@dataclass(frozen=True)
class ActionModelBundle:
    """Action model kept separate from an ordinary forecast artifact."""

    predictor: object
    control_ids: tuple[str, ...]
    outcome_names: tuple[str, ...]
    horizons_minutes: tuple[int, ...]
    joint_domain: JointControlDomain
    evidence: ActionEffectEvidence
    supports_actions: bool = False

    def __post_init__(self) -> None:
        if not self.control_ids or not self.outcome_names or not self.horizons_minutes:
            raise ValueError("action bundle needs controls, outcomes and horizons")
        if self.supports_actions and not (
            self.evidence.shadow_replay_passed and self.evidence.pilot_approved
        ):
            raise ValueError("action capability requires shadow replay and technologist pilot")


@dataclass(frozen=True)
class HistoricalActionDataset:
    """Matched observational action episodes; never an authorization to act."""

    frame: pd.DataFrame
    thresholds: dict[str, float]
    observed_delta_bounds: dict[str, tuple[float, float]]
    match_distance_limit: float


@dataclass(frozen=True)
class HistoricalActionEstimate:
    """UI-ready historical effect estimate explicitly separated from advice."""

    control_id: str
    proposed_delta: float
    sulfur_by_horizon: dict[int, float]
    sulfur_upper_by_horizon: dict[int, float]
    effect_by_horizon: dict[int, float]
    applicable: bool
    reason_codes: tuple[str, ...]

    def as_ui_payload(self) -> dict[str, object]:
        return {
            "title": "Модельный эффект по историческим эпизодам",
            "disclaimer": "Оценка не является советом по изменению уставки.",
            "control_id": self.control_id,
            "proposed_delta": self.proposed_delta,
            "sulfur_change": {
                str(horizon): self.effect_by_horizon[horizon] for horizon in ACTION_HORIZONS
            },
            "predicted_sulfur": {
                str(horizon): self.sulfur_by_horizon[horizon] for horizon in ACTION_HORIZONS
            },
            "sulfur_upper": {
                str(horizon): self.sulfur_upper_by_horizon[horizon] for horizon in ACTION_HORIZONS
            },
            "applicable": self.applicable,
            "advisory": False,
            "reason_codes": self.reason_codes,
        }


@dataclass(frozen=True)
class HistoricalActionEffectModel:
    """Matched-episode sulfur model for shadow display, not control advice."""

    point_estimators: Mapping[int, Any]
    upper_estimators: Mapping[int, Any]
    upper_shifts: Mapping[int, float]
    applicability: JointApplicabilityModel
    observed_delta_bounds: Mapping[str, tuple[float, float]]
    report: Mapping[str, Any]
    supports_actions: bool = False

    def estimate(
        self,
        state: Mapping[str, float],
        control_id: str,
        proposed_delta: float,
        *,
        sulfur_limit: float = 10.0,
    ) -> HistoricalActionEstimate:
        if control_id not in HISTORICAL_ACTION_CONTROLS:
            raise ValueError("historical action model supports only ht:P8 or ht:F19")
        if not np.isfinite(proposed_delta):
            raise ValueError("proposed action delta must be finite")
        missing = set(ACTION_STATE_FEATURES).difference(state)
        if missing:
            raise ValueError(f"historical action state is missing: {sorted(missing)}")
        action = {"delta_ht:P8": 0.0, "delta_ht:F19": 0.0}
        action[f"delta_{control_id}"] = float(proposed_delta)
        row = pd.DataFrame([{**state, **action}], columns=ACTION_MODEL_FEATURES)
        reasons: list[str] = []
        if not bool(self.report.get("evidence_gate_passed", False)):
            reasons.append("ACTION_EFFECT_VALIDATION_FAILED")
        lower, upper = self.observed_delta_bounds[control_id]
        if not lower <= proposed_delta <= upper:
            reasons.append("ACTION_DELTA_OUT_OF_OBSERVED_RANGE")
        applicability = self.applicability.assess(row.loc[:, list(ACTION_STATE_FEATURES)])
        if not applicability.available:
            reasons.append(applicability.reason_code or "OUT_OF_DOMAIN")
        hold = row.copy()
        hold.loc[:, ["delta_ht:P8", "delta_ht:F19"]] = 0.0
        sulfur: dict[int, float] = {}
        sulfur_upper: dict[int, float] = {}
        effect: dict[int, float] = {}
        for horizon in ACTION_HORIZONS:
            point = float(self.point_estimators[horizon].predict(row)[0])
            hold_point = float(self.point_estimators[horizon].predict(hold)[0])
            upper_point = float(self.upper_estimators[horizon].predict(row)[0]) + float(
                self.upper_shifts[horizon]
            )
            sulfur[horizon] = point
            sulfur_upper[horizon] = max(point, upper_point)
            effect[horizon] = point - hold_point
            if sulfur_upper[horizon] > sulfur_limit:
                reasons.append(f"SULFUR_UPPER_LIMIT_{horizon}M")
        return HistoricalActionEstimate(
            control_id,
            proposed_delta,
            sulfur,
            sulfur_upper,
            effect,
            not reasons,
            tuple(dict.fromkeys(reasons)),
        )


def _pak_sulfur(data: PreparedData) -> pd.DataFrame:
    quality = data.quality
    value = pd.to_numeric(quality["value"], errors="coerce")
    selected = quality[
        quality["signal_id"].astype(str).eq("ht:2:Mg.Sulfur")
        & quality["source"].map(lambda item: str(getattr(item, "value", item))).eq("pak")
        & quality["validity"]
        .map(lambda item: str(getattr(item, "value", item)))
        .eq(Validity.VALID.value)
        & quality["unit"].astype(str).eq("mg/kg")
        & value.notna()
        & np.isfinite(value)
    ].copy()
    selected["baseline_sulfur"] = value.loc[selected.index].astype(float)
    selected["timestamp"] = pd.to_datetime(selected["measured_at"], utc=True)
    return cast(
        pd.DataFrame,
        selected.loc[:, ["timestamp", "baseline_sulfur"]].sort_values("timestamp"),
    )


def _action_timeline(data: PreparedData) -> pd.DataFrame:
    signals = (*HISTORICAL_ACTION_CONTROLS, *HISTORICAL_CONTEXT_SIGNALS)
    missing = set(signals).difference(data.telemetry.columns)
    if missing:
        raise ValueError(f"telemetry is missing historical action signals: {sorted(missing)}")
    telemetry = data.telemetry.loc[:, ["timestamp", *signals]].copy()
    telemetry["timestamp"] = pd.to_datetime(telemetry["timestamp"], utc=True)
    for signal in signals:
        telemetry[signal] = pd.to_numeric(telemetry[signal], errors="coerce")
    timeline = pd.merge_asof(
        telemetry.sort_values("timestamp"),
        _pak_sulfur(data),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(minutes=10),
    )
    timeline["sulfur_slope_60m"] = (
        timeline["baseline_sulfur"] - timeline["baseline_sulfur"].shift(6)
    ) / 60.0
    indexed = timeline.set_index("timestamp")
    for horizon in ACTION_HORIZONS:
        future = indexed["baseline_sulfur"].reindex(indexed.index + pd.Timedelta(minutes=horizon))
        timeline[f"sulfur_{horizon}m"] = future.to_numpy(dtype=float)
    return timeline


def build_historical_action_dataset(
    data: PreparedData,
    *,
    train_end: str = "2025-01-01",
    threshold_quantile: float = 0.95,
) -> HistoricalActionDataset:
    """Extract isolated P8/F19 changes and pair them with similar calm states."""
    if not 0.5 < threshold_quantile < 1.0:
        raise ValueError("action threshold quantile must be between 0.5 and 1")
    frame = _action_timeline(data)
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    changes = frame.loc[:, list(HISTORICAL_ACTION_CONTROLS)].diff()
    train = timestamp < pd.Timestamp(train_end, tz="UTC")
    thresholds: dict[str, float] = {}
    for signal in HISTORICAL_ACTION_CONTROLS:
        positive_steps = changes.loc[train, signal].abs()
        positive_steps = positive_steps[positive_steps > 0].dropna()
        if positive_steps.empty:
            raise ValueError(f"historical action signal {signal} has no observed changes")
        thresholds[signal] = float(positive_steps.quantile(threshold_quantile))
    notable = pd.DataFrame(
        {signal: changes[signal].abs().ge(threshold) for signal, threshold in thresholds.items()}
    )
    stable_before = (
        changes.abs().rolling(6, min_periods=6).max().shift(1).lt(pd.Series(thresholds)).all(axis=1)
    )
    quiet_after = (
        notable.iloc[::-1].rolling(18, min_periods=18).sum().iloc[::-1].shift(-1).fillna(1).eq(0)
    ).all(axis=1)
    outcomes = [f"sulfur_{horizon}m" for horizon in ACTION_HORIZONS]
    complete = frame[[*ACTION_STATE_FEATURES, *outcomes]].notna().all(axis=1)
    treated_mask = notable.sum(axis=1).eq(1) & stable_before & quiet_after & complete
    calm_mask = notable.rolling(18, min_periods=18).sum().eq(0).all(axis=1) & complete
    treated = frame.loc[treated_mask].copy()
    calm = frame.loc[calm_mask].copy()
    if treated.empty or calm.empty:
        raise ValueError("no complete isolated action/control episodes were found")

    scaler = StandardScaler().fit(frame.loc[train & complete, list(ACTION_STATE_FEATURES)])
    treated_parts: list[pd.DataFrame] = []
    matched_parts: list[pd.DataFrame] = []
    distance_parts: list[np.ndarray] = []
    for year in sorted(pd.to_datetime(treated["timestamp"], utc=True).dt.year.unique()):
        treated_year = treated.loc[pd.to_datetime(treated["timestamp"], utc=True).dt.year.eq(year)]
        calm_year = calm.loc[pd.to_datetime(calm["timestamp"], utc=True).dt.year.eq(year)]
        if calm_year.empty:
            continue
        treated_state = scaler.transform(treated_year.loc[:, list(ACTION_STATE_FEATURES)])
        calm_state = scaler.transform(calm_year.loc[:, list(ACTION_STATE_FEATURES)])
        distances, indices = (
            NearestNeighbors(n_neighbors=1).fit(calm_state).kneighbors(treated_state)
        )
        treated_parts.append(treated_year)
        matched_parts.append(calm_year.iloc[indices[:, 0]].copy())
        distance_parts.append(distances[:, 0])
    if not treated_parts:
        raise ValueError("no same-year matched action/control episodes were found")
    treated = pd.concat(treated_parts).copy()
    matched = pd.concat(matched_parts).copy()
    matched_distances = np.concatenate(distance_parts)
    treated["pair_id"] = np.arange(len(treated))
    matched["pair_id"] = np.arange(len(treated))
    treated["is_action_episode"] = True
    matched["is_action_episode"] = False
    treated["control_id"] = np.where(
        notable.loc[treated.index, "ht:P8"].to_numpy(), "ht:P8", "ht:F19"
    )
    matched["control_id"] = treated["control_id"].to_numpy()
    for signal in HISTORICAL_ACTION_CONTROLS:
        treated[f"delta_{signal}"] = np.where(
            treated["control_id"].eq(signal),
            changes.loc[treated.index, signal].to_numpy(dtype=float),
            0.0,
        )
        matched[f"delta_{signal}"] = 0.0
    treated["match_distance"] = matched_distances
    matched["match_distance"] = matched_distances
    result = pd.concat([treated, matched], ignore_index=True).sort_values(
        ["timestamp", "pair_id"], kind="stable"
    )
    bounds = {
        signal: (
            float(treated.loc[treated[f"delta_{signal}"].ne(0), f"delta_{signal}"].min()),
            float(treated.loc[treated[f"delta_{signal}"].ne(0), f"delta_{signal}"].max()),
        )
        for signal in HISTORICAL_ACTION_CONTROLS
    }
    return HistoricalActionDataset(
        result.reset_index(drop=True),
        thresholds,
        bounds,
        float(np.quantile(matched_distances, 0.99)),
    )


def fit_historical_action_model(
    dataset: HistoricalActionDataset,
    *,
    train_end: str = "2025-01-01",
    validation_end: str = "2026-01-01",
    seed: int = 42,
) -> HistoricalActionEffectModel:
    """Fit shadow-only matched action models and evaluate 2026 without selection."""
    frame = dataset.frame
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    train = timestamp < pd.Timestamp(train_end, tz="UTC")
    validation = (timestamp >= pd.Timestamp(train_end, tz="UTC")) & (
        timestamp < pd.Timestamp(validation_end, tz="UTC")
    )
    audit = timestamp >= pd.Timestamp(validation_end, tz="UTC")
    if min(int(train.sum()), int(validation.sum()), int(audit.sum())) == 0:
        raise ValueError("historical action model needs train, validation and audit episodes")
    features = list(ACTION_MODEL_FEATURES)
    point_estimators: dict[int, Any] = {}
    upper_estimators: dict[int, Any] = {}
    upper_shifts: dict[int, float] = {}
    validation_metrics: dict[str, Any] = {}
    audit_metrics: dict[str, Any] = {}
    for horizon in ACTION_HORIZONS:
        target = f"sulfur_{horizon}m"
        point = Pipeline(
            (
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            )
        ).fit(frame.loc[train, features], frame.loc[train, target])
        upper = Pipeline(
            (
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        loss="quantile", alpha=0.95, max_depth=2, random_state=seed
                    ),
                ),
            )
        ).fit(frame.loc[train, features], frame.loc[train, target])
        validation_upper = upper.predict(frame.loc[validation, features])
        shift = max(
            0.0,
            float(np.quantile(frame.loc[validation, target] - validation_upper, 0.95)),
        )
        point_estimators[horizon] = point
        upper_estimators[horizon] = upper
        upper_shifts[horizon] = shift
        for label, selected, destination in (
            ("validation", validation, validation_metrics),
            ("audit", audit, audit_metrics),
        ):
            predicted = point.predict(frame.loc[selected, features])
            conservative = np.maximum(
                predicted, upper.predict(frame.loc[selected, features]) + shift
            )
            actual = frame.loc[selected, target].to_numpy(dtype=float)
            destination[str(horizon)] = {
                "period": label,
                "rows": int(selected.sum()),
                "mae": float(mean_absolute_error(actual, predicted)),
                "hold_mae": float(
                    mean_absolute_error(actual, frame.loc[selected, "baseline_sulfur"])
                ),
                "upper_coverage": float(np.mean(actual <= conservative)),
            }
    applicability = fit_joint_applicability(
        frame.loc[train, list(ACTION_STATE_FEATURES)],
        required_features=ACTION_STATE_FEATURES,
    )
    evidence_gate_passed = all(
        values["mae"] <= 0.90 * values["hold_mae"] and values["upper_coverage"] >= 0.95
        for values in validation_metrics.values()
    )
    report = {
        "basis": "matched_historical_episodes_not_causal_guarantee",
        "controls": list(HISTORICAL_ACTION_CONTROLS),
        "context_only": list(HISTORICAL_CONTEXT_SIGNALS),
        "thresholds": dataset.thresholds,
        "episode_rows": int(len(frame)),
        "independent_pairs": int(frame["pair_id"].nunique()),
        "per_control_pairs": {
            signal: int(frame.loc[frame["is_action_episode"], "control_id"].eq(signal).sum())
            for signal in HISTORICAL_ACTION_CONTROLS
        },
        "validation_2025": validation_metrics,
        "audit_2026": audit_metrics,
        "evidence_gate_passed": evidence_gate_passed,
        "supports_actions": False,
        "ui_label": "Модельный эффект по историческим эпизодам",
        "disclaimer": "Не является советом по изменению уставки.",
    }
    return HistoricalActionEffectModel(
        point_estimators,
        upper_estimators,
        upper_shifts,
        applicability,
        dataset.observed_delta_bounds,
        report,
    )


@dataclass(frozen=True)
class LinearActionEffectModel:
    """Small scenario model; coefficients are explicit assumptions, not causality proof."""

    current_setpoints: dict[str, float]
    baseline_sulfur: float
    baseline_sulfur_upper: float | None
    baseline_risk_index: float
    baseline_throughput: float
    baseline_cost_proxy: float
    sulfur_coefficients: dict[str, float]
    risk_coefficients: dict[str, float]
    throughput_coefficients: dict[str, float]
    cost_coefficients: dict[str, float]
    evidence_ref: str


@dataclass(frozen=True)
class ActionOutcome:
    candidate_id: str
    sulfur: float
    sulfur_upper: float | None
    risk_index: float
    throughput: float
    cost_proxy: float
    change_size: float
    feasible: bool
    reason_codes: tuple[str, ...]

    @property
    def rank_key(self) -> tuple[float, float, float, float, str] | None:
        if not self.feasible:
            return None
        return (
            self.risk_index,
            -self.throughput,
            self.cost_proxy,
            self.change_size,
            self.candidate_id,
        )


def _metric_delta(
    changes: dict[str, float], coefficients: dict[str, float], *, metric: str
) -> float:
    unknown = set(coefficients).difference(changes)
    if unknown:
        raise ValueError(f"{metric} coefficients reference unknown controls: {sorted(unknown)}")
    return sum(coefficients.get(signal_id, 0.0) * delta for signal_id, delta in changes.items())


def evaluate_linear_action(
    candidate: CandidateAction,
    model: LinearActionEffectModel,
    controls: tuple[ControlSpec, ...],
    domain: JointControlDomain,
    *,
    sulfur_upper_limit: float = 10.0,
) -> ActionOutcome:
    """Evaluate and hard-filter one hold/setpoint candidate."""
    if candidate.kind is CandidateKind.HOLD:
        setpoints = dict(model.current_setpoints)
    elif candidate.kind is CandidateKind.SETPOINTS:
        setpoints = dict(candidate.setpoints)
    else:
        raise ValueError("linear setpoint model cannot evaluate blending")
    if set(setpoints) != set(model.current_setpoints):
        raise ValueError("candidate setpoints must match the action model controls")

    specs = {control.signal_id: control for control in controls}
    if set(specs) != set(setpoints):
        raise ValueError("control specifications must match model setpoints")
    reasons: list[str] = []
    for signal_id, value in setpoints.items():
        spec = specs[signal_id]
        if not spec.enabled or spec.lower is None or spec.upper is None or spec.max_step is None:
            reasons.append("CONTROL_LIMITS_UNCONFIRMED")
            continue
        if not spec.lower <= value <= spec.upper:
            reasons.append("CONTROL_BOUND_VIOLATION")
        if abs(value - model.current_setpoints[signal_id]) > spec.max_step + 1e-12:
            reasons.append("CONTROL_STEP_VIOLATION")
    if not domain.contains(setpoints):
        reasons.append("OUT_OF_DOMAIN")

    changes = {
        signal_id: value - model.current_setpoints[signal_id]
        for signal_id, value in setpoints.items()
    }
    sulfur_delta = _metric_delta(changes, model.sulfur_coefficients, metric="sulfur")
    sulfur = model.baseline_sulfur + sulfur_delta
    sulfur_upper = (
        None if model.baseline_sulfur_upper is None else model.baseline_sulfur_upper + sulfur_delta
    )
    risk = max(
        0.0,
        min(
            1.0,
            model.baseline_risk_index
            + _metric_delta(changes, model.risk_coefficients, metric="risk"),
        ),
    )
    throughput = model.baseline_throughput + _metric_delta(
        changes, model.throughput_coefficients, metric="throughput"
    )
    cost = model.baseline_cost_proxy + _metric_delta(
        changes, model.cost_coefficients, metric="cost"
    )
    change_size = 0.0
    for signal_id, delta in changes.items():
        step = specs[signal_id].step
        if step is None:
            reasons.append("CONTROL_LIMITS_UNCONFIRMED")
        else:
            change_size += abs(delta) / step
    if not all(isfinite(value) for value in (sulfur, risk, throughput, cost, change_size)) or (
        sulfur_upper is not None and not isfinite(sulfur_upper)
    ):
        reasons.append("NON_FINITE_ACTION_FORECAST")
    if sulfur_upper is None:
        reasons.append("UNCERTAINTY_UNAVAILABLE")
    elif sulfur_upper > sulfur_upper_limit:
        reasons.append("QUALITY_LIMIT_VIOLATION")
    return ActionOutcome(
        candidate_id=candidate.id,
        sulfur=sulfur,
        sulfur_upper=sulfur_upper,
        risk_index=risk,
        throughput=throughput,
        cost_proxy=cost,
        change_size=change_size,
        feasible=not reasons,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def rank_linear_actions(
    candidates: tuple[CandidateAction, ...],
    model: LinearActionEffectModel,
    controls: tuple[ControlSpec, ...],
    domain: JointControlDomain,
    *,
    sulfur_upper_limit: float = 10.0,
) -> tuple[ActionOutcome, ...]:
    """Filter infeasible actions before applying the documented rank key."""
    outcomes = tuple(
        evaluate_linear_action(
            candidate,
            model,
            controls,
            domain,
            sulfur_upper_limit=sulfur_upper_limit,
        )
        for candidate in candidates
    )
    feasible = (outcome for outcome in outcomes if outcome.feasible)
    return tuple(sorted(feasible, key=lambda outcome: outcome.rank_key or ()))


__all__ = [
    "ACTION_HORIZONS",
    "ACTION_MODEL_FEATURES",
    "ActionModelBundle",
    "ActionOutcome",
    "HISTORICAL_ACTION_CONTROLS",
    "HISTORICAL_CONTEXT_SIGNALS",
    "HistoricalActionDataset",
    "HistoricalActionEffectModel",
    "HistoricalActionEstimate",
    "LinearActionEffectModel",
    "build_historical_action_dataset",
    "evaluate_linear_action",
    "fit_historical_action_model",
    "rank_linear_actions",
]
