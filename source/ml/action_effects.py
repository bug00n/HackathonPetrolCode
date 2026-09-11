"""Transparent setpoint consequences for Stage-3 model-only scenarios."""

from __future__ import annotations

from dataclasses import dataclass

from source.contracts import CandidateAction, CandidateKind, ControlSpec
from source.ml.controls import JointControlDomain


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
    "ActionOutcome",
    "LinearActionEffectModel",
    "evaluate_linear_action",
    "rank_linear_actions",
]
