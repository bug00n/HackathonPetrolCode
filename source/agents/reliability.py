"""Transparent severity index; it is explicitly not a failure probability."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from source.agents.effects import assess_blend_candidate
from source.contracts import (
    AgentAssessment,
    AssessmentAgent,
    AssessmentStatus,
    CandidateAction,
    CandidateKind,
    EstimateBasis,
    IntervalKind,
    Issue,
    MetricEstimate,
    OperationMode,
    ProcessState,
    ScenarioConfig,
    Severity,
    Unit,
    Validity,
)


@dataclass(frozen=True)
class SeverityFactor:
    """One evidenced factor with a normal edge and a model boundary."""

    signal_id: str
    unit: str
    normal_edge: float
    model_boundary: float
    adverse_direction: Literal["higher", "lower"]
    evidence_ref: str

    def __post_init__(self) -> None:
        if not self.evidence_ref:
            raise ValueError("severity factor needs evidence_ref")
        if not isfinite(self.normal_edge) or not isfinite(self.model_boundary):
            raise ValueError("severity factor boundaries must be finite")
        if self.adverse_direction == "higher" and self.model_boundary <= self.normal_edge:
            raise ValueError("higher factor boundary must exceed the normal edge")
        if self.adverse_direction == "lower" and self.model_boundary >= self.normal_edge:
            raise ValueError("lower factor boundary must be below the normal edge")


def factor_contribution(value: float, factor: SeverityFactor) -> float:
    """Linearly scale one factor from zero at normal to one at the model boundary."""
    if factor.adverse_direction == "higher":
        raw = (value - factor.normal_edge) / (factor.model_boundary - factor.normal_edge)
    else:
        raw = (factor.normal_edge - value) / (factor.normal_edge - factor.model_boundary)
    return min(1.0, max(0.0, raw))


def assess_confirmed_factors(
    state: ProcessState, factors: tuple[SeverityFactor, ...]
) -> AgentAssessment:
    """Assess evidenced factors and expose every contribution behind the maximum."""
    contributions: dict[str, MetricEstimate] = {}
    issues: list[Issue] = []
    for factor in factors:
        snapshot = state.signals.get(factor.signal_id)
        if snapshot is None or snapshot.selected is None:
            issues.append(
                Issue(
                    code="MISSING_REQUIRED_SIGNAL",
                    severity=Severity.BLOCKING,
                    signal_id=factor.signal_id,
                    detail="Severity factor has no valid available observation.",
                    source_ref=factor.evidence_ref,
                )
            )
            continue
        observation = snapshot.selected
        if observation.validity is not Validity.VALID or observation.value is None:
            issues.append(
                Issue(
                    code="MISSING_REQUIRED_SIGNAL",
                    severity=Severity.BLOCKING,
                    signal_id=factor.signal_id,
                    detail="Severity factor has no valid available observation.",
                    source_ref=factor.evidence_ref,
                )
            )
            continue
        if observation.unit != factor.unit:
            issues.append(
                Issue(
                    code="UNIT_UNCONFIRMED",
                    severity=Severity.BLOCKING,
                    signal_id=factor.signal_id,
                    detail=f"Expected {factor.unit}, received {observation.unit}.",
                    source_ref=factor.evidence_ref,
                )
            )
            continue
        if not snapshot.fresh:
            issues.append(
                Issue(
                    code="STALE_REQUIRED_SIGNAL",
                    severity=Severity.BLOCKING,
                    signal_id=factor.signal_id,
                    detail="Severity factor is stale.",
                    source_ref=observation.source_ref,
                )
            )
            continue
        contributions[f"severity:{factor.signal_id}"] = MetricEstimate(
            value=factor_contribution(observation.value, factor),
            lower=None,
            upper=None,
            unit=Unit.RISK_INDEX.value,
            basis=EstimateBasis.PROXY,
            interval_kind=IntervalKind.NONE,
            interval_level=None,
            reference=factor.evidence_ref,
            assumptions=(
                "Linear severity contribution; not a probability of failure.",
                f"Normal edge={factor.normal_edge}; model boundary={factor.model_boundary}.",
            ),
        )

    if not contributions:
        if not issues:
            issues.append(
                Issue(
                    code="RELIABILITY_UNAVAILABLE",
                    severity=Severity.BLOCKING,
                    signal_id=None,
                    detail="No confirmed severity factors are configured.",
                    source_ref=None,
                )
            )
        return AgentAssessment(
            agent=AssessmentAgent.RELIABILITY,
            state_id=state.state_id,
            candidate_id="hold",
            evaluated_for=state.as_of,
            status=AssessmentStatus.UNAVAILABLE,
            metrics={},
            issues=tuple(issues),
        )

    risk_index = max(metric.value or 0.0 for metric in contributions.values())
    contributions["risk_index"] = MetricEstimate(
        value=risk_index,
        lower=None,
        upper=None,
        unit=Unit.RISK_INDEX.value,
        basis=EstimateBasis.PROXY,
        interval_kind=IntervalKind.NONE,
        interval_level=None,
        reference="DESIGN.md#8-reliability",
        assumptions=("Maximum of evidenced factor contributions; not a failure probability.",),
    )
    return AgentAssessment(
        agent=AssessmentAgent.RELIABILITY,
        state_id=state.state_id,
        candidate_id="hold",
        evaluated_for=state.as_of,
        status=AssessmentStatus.DEGRADED if issues else AssessmentStatus.OK,
        metrics=contributions,
        issues=tuple(issues),
    )


def assess_reliability(state: ProcessState, scenario: ScenarioConfig) -> AgentAssessment:
    """Use scenario risk in model_demo; refuse to invent historical factor limits."""
    if state.mode is not scenario.mode:
        raise ValueError("state and scenario modes must match")
    if state.mode is OperationMode.MODEL_DEMO and scenario.blend_components:
        hold = CandidateAction(
            id="hold",
            kind=CandidateKind.HOLD,
            horizon_minutes=60,
            is_model_scenario=state.mode is OperationMode.MODEL_DEMO,
        )
        return assess_blend_candidate(state, hold, scenario)[1]
    return assess_confirmed_factors(state, ())


__all__ = [
    "SeverityFactor",
    "assess_confirmed_factors",
    "assess_reliability",
    "factor_contribution",
]
