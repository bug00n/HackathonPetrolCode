"""Stage-3 control selection, applicability and deterministic action grids.

Observed telemetry describes where the plant used to operate; it is not an
engineering operating limit.  This module therefore keeps observational
domains separate from :class:`ControlSpec` and enables actions only after a
separate temporal action-effect validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from source.contracts import (
    CandidateAction,
    CandidateKind,
    ConstraintBasis,
    ControlSpec,
    ProcessState,
    Unit,
)

CONTROL_IDS = ("ht:P8", "ht:T11", "ht:F19")
CONTROL_MEANINGS = {
    "ht:P8": "Polisep reactor R-202 inlet gas/feed temperature",
    "ht:T11": "Hydrotreatment unit mass feed rate",
    "ht:F19": "Polisep reactor R-202 inlet pressure",
}
CONTROL_EVIDENCE = (
    "materials/Теги_хакатон.xlsx#КИП;DESIGN.md#16-подтверждения-экспертов-от-10092026"
)
ACTION_HORIZONS_MINUTES = (15, 30, 60, 120, 180)


@dataclass(frozen=True)
class ObservedControlStats:
    """Training-period telemetry summary, never an allowable control range."""

    signal_id: str
    count: int
    missing: int
    q05: float
    median: float
    q95: float
    median_abs_step: float
    q95_abs_step: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "count": self.count,
            "missing": self.missing,
            "q05": self.q05,
            "median": self.median,
            "q95": self.q95,
            "median_abs_step": self.median_abs_step,
            "q95_abs_step": self.q95_abs_step,
            "interpretation": "observed_training_distribution_not_an_engineering_limit",
        }


@dataclass(frozen=True)
class JointControlDomain:
    """Robust Mahalanobis domain fitted only on a training-period sample."""

    signal_ids: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    inverse_correlation: tuple[tuple[float, ...], ...]
    max_distance_squared: float

    def distance_squared(self, values: dict[str, float]) -> float:
        if any(signal_id not in values for signal_id in self.signal_ids):
            return float("inf")
        vector = np.asarray([values[signal_id] for signal_id in self.signal_ids], dtype=float)
        if not np.isfinite(vector).all():
            return float("inf")
        center = np.asarray(self.center)
        scale = np.asarray(self.scale)
        inverse = np.asarray(self.inverse_correlation)
        standardized = (vector - center) / scale
        return float(standardized @ inverse @ standardized)

    def contains(self, values: dict[str, float]) -> bool:
        return self.distance_squared(values) <= self.max_distance_squared


@dataclass(frozen=True)
class ActionEffectEvidence:
    """Separate temporal evaluation of an action-effect model."""

    validation_start: str
    validation_end: str
    change_episode_count: int
    best_lag_minutes: int
    baseline_mae: float
    action_model_mae: float
    per_control_episode_counts: dict[str, int] | None = None
    conservative_coverage: float | None = None
    sign_stable_folds: int | None = None
    shadow_replay_passed: bool = False
    pilot_approved: bool = False


@dataclass(frozen=True)
class ActionCapabilityReport:
    supports_actions: bool
    reason_codes: tuple[str, ...]
    control_ids: tuple[str, ...]
    evidence: ActionEffectEvidence | None


def extract_change_episodes(
    telemetry: pd.DataFrame,
    change_thresholds: dict[str, float],
    *,
    target_signal: str = "ht:2:Mg.Sulfur",
    horizons_minutes: tuple[int, ...] = ACTION_HORIZONS_MINUTES,
    stable_window_minutes: int = 60,
) -> pd.DataFrame:
    """Extract isolated natural control changes for offline action research.

    The result is observational evidence only. It cannot enable controls by
    itself because operator feedback and unobserved feed changes may confound it.
    """
    required = {"timestamp", target_signal, *CONTROL_IDS}
    missing = sorted(required.difference(telemetry.columns))
    if missing:
        raise ValueError(f"telemetry is missing action-research columns: {missing}")
    if set(change_thresholds) != set(CONTROL_IDS) or any(
        not np.isfinite(value) or value <= 0 for value in change_thresholds.values()
    ):
        raise ValueError("positive finite change thresholds are required for every control")
    if (
        stable_window_minutes <= 0
        or not horizons_minutes
        or any(horizon <= 0 for horizon in horizons_minutes)
    ):
        raise ValueError("stable window and action horizons must be positive")

    frame = telemetry.loc[:, ["timestamp", *CONTROL_IDS, target_signal]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep=False)
    frame = frame.set_index("timestamp")
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    changes = numeric.loc[:, list(CONTROL_IDS)].diff()
    rows: list[dict[str, Any]] = []
    for timestamp, deltas in changes.iterrows():
        active = [
            signal_id
            for signal_id in CONTROL_IDS
            if pd.notna(deltas[signal_id])
            and abs(float(deltas[signal_id])) >= change_thresholds[signal_id]
        ]
        if len(active) != 1:
            continue
        signal_id = active[0]
        start = timestamp - pd.Timedelta(minutes=stable_window_minutes)
        before = numeric.loc[(numeric.index >= start) & (numeric.index < timestamp)]
        if len(before) < 3 or before[[*CONTROL_IDS, target_signal]].isna().any().any():
            continue
        if any(
            before[control].diff().abs().max() >= change_thresholds[control]
            for control in CONTROL_IDS
        ):
            continue
        baseline = float(before[target_signal].iloc[-1])
        row: dict[str, Any] = {
            "changed_at": timestamp,
            "control_id": signal_id,
            "control_delta": float(deltas[signal_id]),
            "baseline_quality": baseline,
        }
        complete = True
        for horizon in horizons_minutes:
            target_at = timestamp + pd.Timedelta(minutes=horizon)
            position = numeric.index.searchsorted(target_at)
            if position >= len(numeric.index):
                complete = False
                break
            observed_at = numeric.index[position]
            if observed_at - target_at > pd.Timedelta(minutes=10):
                complete = False
                break
            value = numeric.iloc[position][target_signal]
            if pd.isna(value):
                complete = False
                break
            row[f"delta_quality_{horizon}m"] = float(value) - baseline
        if complete:
            rows.append(row)
    return pd.DataFrame(rows)


def unconfirmed_real_controls() -> tuple[ControlSpec, ...]:
    """Return confirmed meanings while keeping unsupported real limits disabled."""
    return tuple(
        ControlSpec(
            signal_id=signal_id,
            unit=Unit.UNKNOWN.value,
            lower=None,
            upper=None,
            max_step=None,
            step=None,
            evidence_ref=CONTROL_EVIDENCE,
            basis=ConstraintBasis.CONFIRMED,
            enabled=False,
        )
        for signal_id in CONTROL_IDS
    )


def summarize_observed_controls(frame: pd.DataFrame) -> tuple[ObservedControlStats, ...]:
    """Summarize the selected controls without converting statistics into limits."""
    missing_columns = [signal_id for signal_id in CONTROL_IDS if signal_id not in frame]
    if missing_columns:
        raise ValueError(f"missing control columns: {missing_columns}")
    result: list[ObservedControlStats] = []
    for signal_id in CONTROL_IDS:
        values = pd.to_numeric(frame[signal_id], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            raise ValueError(f"control {signal_id} has no finite observations")
        steps = valid.diff().abs().dropna()
        result.append(
            ObservedControlStats(
                signal_id=signal_id,
                count=int(valid.size),
                missing=int(values.isna().sum()),
                q05=float(valid.quantile(0.05)),
                median=float(valid.median()),
                q95=float(valid.quantile(0.95)),
                median_abs_step=float(steps.median()) if not steps.empty else 0.0,
                q95_abs_step=float(steps.quantile(0.95)) if not steps.empty else 0.0,
            )
        )
    return tuple(result)


def fit_joint_control_domain(
    frame: pd.DataFrame,
    *,
    coverage: float = 0.99,
) -> JointControlDomain:
    """Fit a robust joint domain; callers must pass training-period rows only."""
    if not 0.5 < coverage < 1.0:
        raise ValueError("coverage must be between 0.5 and 1")
    data = frame.loc[:, list(CONTROL_IDS)].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 20:
        raise ValueError("at least 20 complete training rows are required")
    center = data.median(axis=0).to_numpy(dtype=float)
    q25 = data.quantile(0.25).to_numpy(dtype=float)
    q75 = data.quantile(0.75).to_numpy(dtype=float)
    scale = q75 - q25
    if (scale <= 0).any():
        raise ValueError("every control needs a positive training IQR")
    standardized = (data.to_numpy(dtype=float) - center) / scale
    correlation = np.cov(standardized, rowvar=False)
    inverse = np.linalg.pinv(correlation)
    distances = np.einsum("ij,jk,ik->i", standardized, inverse, standardized)
    threshold = float(np.quantile(distances, coverage))
    return JointControlDomain(
        signal_ids=CONTROL_IDS,
        center=tuple(float(value) for value in center),
        scale=tuple(float(value) for value in scale),
        inverse_correlation=tuple(tuple(float(value) for value in row) for row in inverse),
        max_distance_squared=threshold,
    )


def assess_action_capability(
    controls: tuple[ControlSpec, ...],
    evidence: ActionEffectEvidence | None,
    *,
    minimum_episodes: int = 100,
) -> ActionCapabilityReport:
    """Gate action support independently from ordinary forecast performance."""
    reasons: list[str] = []
    if not controls:
        reasons.append("CONTROL_SET_EMPTY")
    if any(not control.enabled for control in controls):
        reasons.append("CONTROL_LIMITS_UNCONFIRMED")
    if any(control.unit == Unit.UNKNOWN.value for control in controls):
        reasons.append("CONTROL_UNITS_UNKNOWN")
    if any(control.basis is ConstraintBasis.MODEL_ASSUMPTION for control in controls):
        reasons.append("REAL_CONTROL_LIMITS_MODEL_ASSUMPTION")
    if evidence is None:
        reasons.append("ACTION_EFFECT_VALIDATION_MISSING")
    else:
        if evidence.change_episode_count < minimum_episodes:
            reasons.append("INSUFFICIENT_CHANGE_EPISODES")
        counts = evidence.per_control_episode_counts
        if counts is None or any(
            counts.get(control.signal_id, 0) < minimum_episodes for control in controls
        ):
            reasons.append("INSUFFICIENT_PER_CONTROL_EPISODES")
        if not np.isfinite(evidence.change_episode_count) or evidence.change_episode_count < 0:
            reasons.append("ACTION_EPISODE_COUNT_INVALID")
        if not all(
            np.isfinite(value)
            for value in (
                evidence.best_lag_minutes,
                evidence.baseline_mae,
                evidence.action_model_mae,
            )
        ):
            reasons.append("ACTION_EFFECT_METRICS_NON_FINITE")
        if evidence.baseline_mae < 0 or evidence.action_model_mae < 0:
            reasons.append("ACTION_EFFECT_METRICS_INVALID")
        if not 0 <= evidence.best_lag_minutes <= 180:
            reasons.append("ACTION_LAG_OUT_OF_RANGE")
        if (
            np.isfinite(evidence.baseline_mae)
            and np.isfinite(evidence.action_model_mae)
            and evidence.action_model_mae >= evidence.baseline_mae
        ):
            reasons.append("ACTION_MODEL_NO_TEMPORAL_GAIN")
        elif evidence.action_model_mae > 0.9 * evidence.baseline_mae:
            reasons.append("ACTION_MODEL_GAIN_BELOW_10_PERCENT")
        if evidence.conservative_coverage is None or evidence.conservative_coverage < 0.95:
            reasons.append("ACTION_INTERVAL_COVERAGE_INSUFFICIENT")
        if evidence.sign_stable_folds != 3:
            reasons.append("ACTION_EFFECT_SIGN_UNSTABLE")
        if not evidence.shadow_replay_passed:
            reasons.append("SHADOW_REPLAY_MISSING")
        if not evidence.pilot_approved:
            reasons.append("TECHNOLOGIST_PILOT_MISSING")
    return ActionCapabilityReport(
        supports_actions=not reasons,
        reason_codes=tuple(reasons),
        control_ids=tuple(control.signal_id for control in controls),
        evidence=evidence,
    )


def _selected_value(state: ProcessState, signal_id: str) -> float:
    snapshot = state.signals.get(signal_id)
    if snapshot is None or snapshot.selected is None or snapshot.selected.value is None:
        raise ValueError(f"current value unavailable for {signal_id}")
    return float(snapshot.selected.value)


def _grid_values(current: float, control: ControlSpec) -> tuple[float, ...]:
    limits = (control.lower, control.upper, control.max_step, control.step)
    if not control.enabled or None in limits:
        raise ValueError(f"control {control.signal_id} is not fully enabled")
    assert control.lower is not None
    assert control.upper is not None
    assert control.max_step is not None
    assert control.step is not None
    values = {current}
    for multiplier in (-2, -1, 1, 2):
        delta = multiplier * control.step
        if abs(delta) <= control.max_step + 1e-12:
            value = min(control.upper, max(control.lower, current + delta))
            values.add(float(value))
    return tuple(sorted(values))


def generate_setpoint_candidates(
    state: ProcessState,
    controls: tuple[ControlSpec, ...],
    *,
    supports_actions: bool,
    joint_domain: JointControlDomain | None,
    horizon_minutes: int = 60,
    max_action_combinations: int = 125,
) -> tuple[CandidateAction, ...]:
    """Generate at most 5^3 deterministic combinations plus an explicit hold."""
    if horizon_minutes <= 0:
        raise ValueError("horizon_minutes must be positive")
    hold = CandidateAction(
        id="hold",
        kind=CandidateKind.HOLD,
        horizon_minutes=horizon_minutes,
        is_model_scenario=state.mode.value != "history",
    )
    if not supports_actions:
        return (hold,)
    if not 1 <= len(controls) <= 3:
        raise ValueError("setpoint search requires one to three controls")
    if joint_domain is None:
        raise ValueError("joint control domain is required for action generation")

    ordered = tuple(sorted(controls, key=lambda item: item.signal_id))
    grids = [
        _grid_values(_selected_value(state, control.signal_id), control) for control in ordered
    ]
    combination_count = int(np.prod([len(values) for values in grids])) - 1
    if combination_count > max_action_combinations:
        raise ValueError(
            f"candidate grid requires {combination_count} action combinations, "
            f"but max_action_combinations={max_action_combinations}"
        )

    current = {control.signal_id: _selected_value(state, control.signal_id) for control in ordered}
    candidates = [hold]
    for vector in product(*grids):
        setpoints = {
            control.signal_id: float(value) for control, value in zip(ordered, vector, strict=True)
        }
        if all(abs(setpoints[key] - current[key]) <= 1e-12 for key in setpoints):
            continue
        if not joint_domain.contains(setpoints):
            continue
        payload = ",".join(f"{key}={setpoints[key]:.8g}" for key in sorted(setpoints))
        candidates.append(
            CandidateAction(
                id=f"setpoints:{payload}",
                kind=CandidateKind.SETPOINTS,
                setpoints=setpoints,
                horizon_minutes=horizon_minutes,
                is_model_scenario=state.mode.value != "history",
            )
        )
    return tuple(candidates)


__all__ = [
    "ActionCapabilityReport",
    "ACTION_HORIZONS_MINUTES",
    "ActionEffectEvidence",
    "CONTROL_EVIDENCE",
    "CONTROL_IDS",
    "CONTROL_MEANINGS",
    "JointControlDomain",
    "ObservedControlStats",
    "assess_action_capability",
    "extract_change_episodes",
    "fit_joint_control_domain",
    "generate_setpoint_candidates",
    "summarize_observed_controls",
    "unconfirmed_real_controls",
]
