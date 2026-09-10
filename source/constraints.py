"""Single stage-1 implementation of hard decision constraints."""

from __future__ import annotations

from source.contracts import (
    AgentAssessment,
    CandidateAction,
    CandidateKind,
    ConstraintResult,
    ConstraintSpec,
    ConstraintStatus,
    MetricEstimate,
    ProcessState,
    ScenarioConfig,
)


def _metric(assessments: tuple[AgentAssessment, ...], metric_name: str) -> MetricEstimate | None:
    for assessment in assessments:
        if metric_name in assessment.metrics:
            return assessment.metrics[metric_name]
    return None


def _quality_check(
    constraint: ConstraintSpec,
    candidate: CandidateAction,
    assessments: tuple[AgentAssessment, ...],
) -> ConstraintResult:
    metric = _metric(assessments, constraint.metric)
    if metric is None:
        return ConstraintResult(
            constraint_id=constraint.id,
            candidate_id=candidate.id,
            status=ConstraintStatus.UNKNOWN,
            actual=None,
            lower=constraint.lower,
            upper=constraint.upper,
            unit=constraint.unit,
            basis=constraint.basis,
            evidence_ref=constraint.evidence_ref,
            reason_code="UNCERTAINTY_UNAVAILABLE",
        )
    actual = metric.upper if constraint.use_upper_estimate else metric.value
    status = ConstraintStatus.PASS
    reason = "OK"
    if actual is None:
        status = ConstraintStatus.UNKNOWN
        reason = "UNCERTAINTY_UNAVAILABLE"
    elif constraint.lower is not None and actual < constraint.lower:
        status = ConstraintStatus.FAIL
        reason = "QUALITY_LIMIT"
    elif constraint.upper is not None and actual > constraint.upper:
        status = ConstraintStatus.FAIL
        reason = "QUALITY_LIMIT"
    return ConstraintResult(
        constraint_id=constraint.id,
        candidate_id=candidate.id,
        status=status,
        actual=actual,
        lower=constraint.lower,
        upper=constraint.upper,
        unit=metric.unit,
        basis=constraint.basis,
        evidence_ref=constraint.evidence_ref,
        reason_code=reason,
    )


def _stock_checks(
    scenario: ScenarioConfig, candidate: CandidateAction
) -> tuple[ConstraintResult, ...]:
    if candidate.kind is not CandidateKind.BLEND or scenario.total_mass_t is None:
        return ()
    by_id = {item.id: item for item in scenario.blend_components}
    checks: list[ConstraintResult] = []
    for component_id, fraction in candidate.blend_mass_fractions.items():
        component = by_id[component_id]
        requested = fraction * scenario.total_mass_t
        status = (
            ConstraintStatus.PASS
            if requested <= component.available_mass_t
            else ConstraintStatus.FAIL
        )
        checks.append(
            ConstraintResult(
                constraint_id=f"component_stock:{component_id}",
                candidate_id=candidate.id,
                status=status,
                actual=requested,
                lower=None,
                upper=component.available_mass_t,
                unit="t",
                basis="model_assumption",
                evidence_ref="ScenarioConfig.blend_components",
                reason_code="OK" if status is ConstraintStatus.PASS else "COMPONENT_STOCK",
            )
        )
    return tuple(checks)


def check_constraints(
    state: ProcessState,
    candidate: CandidateAction,
    assessments: tuple[AgentAssessment, ...],
    scenario: ScenarioConfig,
) -> tuple[ConstraintResult, ...]:
    """Check all hard constraints for one evaluated candidate."""
    _ = state
    checks = [_quality_check(item, candidate, assessments) for item in scenario.constraints]
    checks.extend(_stock_checks(scenario, candidate))
    return tuple(checks)


__all__ = ["check_constraints"]
