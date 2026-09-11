"""Pure Stage-5 materiality and cooldown policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PolicyParameters:
    """Thresholds are absolute for risk and relative for throughput/cost."""

    risk: float = 0.02
    throughput: float = 0.02
    cost_proxy: float = 0.02
    cooldown_minutes: int = 60
    zero_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        values = (self.risk, self.throughput, self.cost_proxy, self.zero_tolerance)
        if any(value < 0 for value in values) or self.cooldown_minutes < 0:
            raise ValueError("policy thresholds and cooldown must be non-negative")


@dataclass(frozen=True)
class PolicyDecision:
    """Whether an already feasible candidate may replace hold."""

    allow_change: bool
    reason_code: str
    first_differing_criterion: str | None = None


@dataclass(frozen=True)
class PolicyReplayCase:
    """One chronological validation case used to choose policy settings."""

    now: datetime
    hold: Mapping[str, float | None]
    candidate: Mapping[str, float | None]
    active_criteria: tuple[str, ...]
    hold_feasible: bool
    change_required: bool
    last_recommended_at: datetime | None = None


DEFAULT_POLICY = PolicyParameters()


def _relative_improvement(current: float, proposed: float, *, higher_is_better: bool) -> float:
    scale = max(abs(current), 1e-9)
    delta = proposed - current if higher_is_better else current - proposed
    return delta / scale


def assess_change_policy(
    hold: Mapping[str, float | None],
    candidate: Mapping[str, float | None],
    active_criteria: Sequence[str],
    *,
    hold_feasible: bool,
    now: datetime,
    last_recommended_at: datetime | None,
    parameters: PolicyParameters = DEFAULT_POLICY,
) -> PolicyDecision:
    """Apply first-differing-criterion materiality and conditional cooldown."""
    if not hold_feasible:
        return PolicyDecision(True, "HOLD_INFEASIBLE")

    first: str | None = None
    current = proposed = 0.0
    for criterion in active_criteria:
        hold_value = hold.get(criterion)
        candidate_value = candidate.get(criterion)
        if hold_value is None or candidate_value is None:
            return PolicyDecision(False, "CRITERION_UNAVAILABLE", criterion)
        if abs(candidate_value - hold_value) > parameters.zero_tolerance:
            first, current, proposed = criterion, hold_value, candidate_value
            break
    if first is None or first == "change_size":
        return PolicyDecision(False, "NO_MATERIAL_IMPROVEMENT", first)

    if first == "risk_index":
        material = current - proposed >= parameters.risk
    elif first == "throughput":
        material = (
            _relative_improvement(current, proposed, higher_is_better=True) >= parameters.throughput
        )
    elif first == "cost_proxy":
        material = (
            _relative_improvement(current, proposed, higher_is_better=False)
            >= parameters.cost_proxy
        )
    else:
        return PolicyDecision(False, "CRITERION_UNAVAILABLE", first)
    if not material:
        return PolicyDecision(False, "NO_MATERIAL_IMPROVEMENT", first)

    if last_recommended_at is not None:
        if now.tzinfo is None or last_recommended_at.tzinfo is None:
            raise ValueError("policy timestamps must be timezone-aware")
        elapsed_minutes = (now - last_recommended_at).total_seconds() / 60.0
        if elapsed_minutes < 0:
            raise ValueError("last_recommended_at cannot be in the future")
        if elapsed_minutes < parameters.cooldown_minutes:
            return PolicyDecision(False, "ACTION_COOLDOWN", first)
    return PolicyDecision(True, "MATERIAL_IMPROVEMENT", first)


def tune_policy(
    validation_cases: Sequence[PolicyReplayCase],
    candidates: Sequence[PolicyParameters],
) -> PolicyParameters:
    """Choose settings on validation by missed, unnecessary, then total actions."""
    if not validation_cases or not candidates:
        raise ValueError("validation cases and candidate policies are required")

    def score(parameters: PolicyParameters) -> tuple[int, int, int, float, float, float, int]:
        missed = unnecessary = actions = 0
        for case in validation_cases:
            decision = assess_change_policy(
                case.hold,
                case.candidate,
                case.active_criteria,
                hold_feasible=case.hold_feasible,
                now=case.now,
                last_recommended_at=case.last_recommended_at,
                parameters=parameters,
            )
            actions += decision.allow_change
            missed += case.change_required and not decision.allow_change
            unnecessary += not case.change_required and decision.allow_change
        return (
            missed,
            unnecessary,
            actions,
            parameters.risk,
            parameters.throughput,
            parameters.cost_proxy,
            parameters.cooldown_minutes,
        )

    return min(candidates, key=score)


__all__ = [
    "PolicyDecision",
    "PolicyParameters",
    "PolicyReplayCase",
    "DEFAULT_POLICY",
    "assess_change_policy",
    "tune_policy",
]
