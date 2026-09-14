"""Deterministic consequences for the stage-1 blending demonstration."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta

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
)

FRACTION_TOLERANCE = 1e-9


def _recipe(candidate: CandidateAction, scenario: ScenarioConfig) -> dict[str, float]:
    component_ids = {component.id for component in scenario.blend_components}
    if len(component_ids) != 2:
        raise ValueError("stage-1 blending requires exactly two components")
    current = scenario.current_blend_mass_fractions
    if set(current) != component_ids or abs(sum(current.values()) - 1.0) > FRACTION_TOLERANCE:
        raise ValueError("current blend must contain every component and sum to one")

    if candidate.kind is CandidateKind.HOLD:
        recipe = dict(current)
    elif candidate.kind is CandidateKind.BLEND:
        recipe = dict(candidate.blend_mass_fractions)
    else:
        raise ValueError("stage-1 effects cannot evaluate setpoint actions")

    if set(recipe) != component_ids:
        raise ValueError("blend recipe must contain every scenario component exactly once")
    if abs(sum(recipe.values()) - 1.0) > FRACTION_TOLERANCE:
        raise ValueError("blend mass fractions must sum to one")
    return recipe


def _weighted(recipe: Mapping[str, float], values: Mapping[str, float | None]) -> float | None:
    if any(values[component_id] is None for component_id, weight in recipe.items() if weight > 0):
        return None
    return sum(weight * (values[component_id] or 0.0) for component_id, weight in recipe.items())


def _weighted_metric(
    recipe: Mapping[str, float],
    values: Mapping[str, MetricEstimate | None],
    metric_name: str,
) -> tuple[float | None, float | None, str, EstimateBasis, IntervalKind, str]:
    positive = [component_id for component_id, weight in recipe.items() if weight > 0]
    present_by_id: dict[str, MetricEstimate] = {}
    for component_id in positive:
        estimate = values[component_id]
        if estimate is None:
            return (
                None,
                None,
                Unit.UNKNOWN.value,
                EstimateBasis.FORMULA,
                IntervalKind.NONE,
                "missing",
            )
        present_by_id[component_id] = estimate
    present = tuple(present_by_id.values())
    units = {metric.unit for metric in present}
    if len(units) != 1:
        raise ValueError(f"all {metric_name} components must use one unit")
    value = sum(recipe[key] * (present_by_id[key].value or 0.0) for key in positive)
    upper = (
        None
        if any(present_by_id[key].upper is None for key in positive)
        else sum(recipe[key] * (present_by_id[key].upper or 0.0) for key in positive)
    )
    interval = IntervalKind.SCENARIO_BOUND if upper is not None else IntervalKind.NONE
    basis = EstimateBasis.FORMULA
    if any(estimate.basis is EstimateBasis.PROXY for estimate in present):
        basis = EstimateBasis.PROXY
    return value, upper, present[0].unit, basis, interval, "DESIGN.md#9"


def calculate_blend_metrics(
    candidate: CandidateAction, scenario: ScenarioConfig
) -> dict[str, MetricEstimate]:
    """Calculate product properties and transparent ranking proxies for one recipe."""
    if scenario.mode is not OperationMode.MODEL_DEMO:
        raise ValueError("stage-1 blending effects are only valid in model_demo mode")
    if scenario.total_mass_t is None:
        raise ValueError("blend scenario needs total_mass_t")

    recipe = _recipe(candidate, scenario)
    components = {component.id: component for component in scenario.blend_components}
    sulfur_units = {component.sulfur.unit for component in components.values()}
    if sulfur_units != {Unit.MG_KG.value}:
        raise ValueError("all sulfur components must use mg/kg")

    sulfur_value = _weighted(
        recipe, {key: component.sulfur.value for key, component in components.items()}
    )
    sulfur_upper = _weighted(
        recipe, {key: component.sulfur.upper for key, component in components.items()}
    )
    t95_value, t95_upper, t95_unit, t95_basis, t95_interval, t95_ref = _weighted_metric(
        recipe, {key: component.t95 for key, component in components.items()}, "t95"
    )
    cetane_value, cetane_upper, cetane_unit, cetane_basis, cetane_interval, cetane_ref = (
        _weighted_metric(
            recipe,
            {key: component.cetane_number for key, component in components.items()},
            "cetane_number",
        )
    )
    risk_index = sum(recipe[key] * component.risk_index for key, component in components.items())
    cost_proxy = sum(
        recipe[key] * component.cost_proxy_per_t for key, component in components.items()
    )
    change_size = sum(
        abs(recipe[key] - scenario.current_blend_mass_fractions.get(key, 0.0)) for key in recipe
    )
    assumptions = tuple(scenario.assumptions) + (
        "Sulfur, T95 and cetane number are mixed by mass fraction for model-demo only.",
        "Risk and cost are scenario proxies, not plant safety or currency estimates.",
    )

    def estimate(
        value: float | None,
        unit: str,
        basis: EstimateBasis,
        reference: str,
        *,
        upper: float | None = None,
        interval_kind: IntervalKind = IntervalKind.NONE,
    ) -> MetricEstimate:
        return MetricEstimate(
            value=value,
            lower=None,
            upper=upper,
            unit=unit,
            basis=basis,
            interval_kind=interval_kind,
            interval_level=None,
            reference=reference,
            assumptions=assumptions,
        )

    sulfur_interval = IntervalKind.SCENARIO_BOUND if sulfur_upper is not None else IntervalKind.NONE
    return {
        "sulfur": estimate(
            sulfur_value,
            Unit.MG_KG.value,
            EstimateBasis.FORMULA,
            "DESIGN.md#9",
            upper=sulfur_upper,
            interval_kind=sulfur_interval,
        ),
        "t95": estimate(
            t95_value,
            t95_unit,
            t95_basis,
            t95_ref,
            upper=t95_upper,
            interval_kind=t95_interval,
        ),
        "cetane_number": estimate(
            cetane_value,
            cetane_unit,
            cetane_basis,
            cetane_ref,
            upper=cetane_upper,
            interval_kind=cetane_interval,
        ),
        "risk_index": estimate(
            risk_index,
            Unit.RISK_INDEX.value,
            EstimateBasis.PROXY,
            "config scenario blend_components[*].risk_index",
        ),
        "throughput": estimate(
            scenario.total_mass_t,
            "t",
            EstimateBasis.FORMULA,
            "config scenario total_mass_t",
        ),
        "cost_proxy": estimate(
            cost_proxy,
            Unit.PROXY.value,
            EstimateBasis.PROXY,
            "config scenario blend_components[*].cost_proxy_per_t",
        ),
        "change_size": estimate(
            change_size,
            Unit.DIMENSIONLESS.value,
            EstimateBasis.FORMULA,
            "DESIGN.md#7",
        ),
    }


def assess_blend_candidate(
    state: ProcessState, candidate: CandidateAction, scenario: ScenarioConfig
) -> tuple[AgentAssessment, AgentAssessment, AgentAssessment]:
    """Return quality, reliability and optimizer assessments for one recipe."""
    if state.mode is not scenario.mode:
        raise ValueError("state and scenario modes must match")
    metrics = calculate_blend_metrics(candidate, scenario)
    sulfur = metrics["sulfur"]
    t95 = metrics["t95"]
    cetane = metrics["cetane_number"]
    quality_status = AssessmentStatus.OK
    quality_issues: tuple[Issue, ...] = ()
    if t95.value is None:
        quality_issues += (
            Issue(
                code="UNASSESSED_REQUIRED_PROPERTY",
                severity=Severity.BLOCKING,
                signal_id="blend:t95",
                detail="T95 is required for a blend decision but has no model-demo value.",
                source_ref=f"scenario:{scenario.id}",
            ),
        )
    if cetane.value is None:
        quality_issues += (
            Issue(
                code="UNASSESSED_REQUIRED_PROPERTY",
                severity=Severity.BLOCKING,
                signal_id="blend:cetane_number",
                detail=(
                    "Cetane number is required for a blend decision but has no model-demo value."
                ),
                source_ref=f"scenario:{scenario.id}",
            ),
        )
    if sulfur.value is None:
        quality_issues += (
            Issue(
                code="MISSING_REQUIRED_SIGNAL",
                severity=Severity.BLOCKING,
                signal_id="blend:sulfur",
                detail="Sulfur is missing for a component with a positive mass fraction.",
                source_ref=f"scenario:{scenario.id}",
            ),
        )
    elif scenario.require_upper_bound and sulfur.upper is None:
        quality_issues += (
            Issue(
                code="UNCERTAINTY_UNAVAILABLE",
                severity=Severity.BLOCKING,
                signal_id="blend:sulfur",
                detail="The scenario requires an upper sulfur estimate, but it is unavailable.",
                source_ref=f"scenario:{scenario.id}",
            ),
        )

    if quality_issues:
        quality_status = AssessmentStatus.UNAVAILABLE

    evaluated_for = state.as_of + timedelta(minutes=candidate.horizon_minutes)

    def assessment(
        agent: AssessmentAgent,
        status: AssessmentStatus,
        selected_metrics: tuple[str, ...],
        issues: tuple[Issue, ...] = (),
    ) -> AgentAssessment:
        return AgentAssessment(
            agent=agent,
            state_id=state.state_id,
            candidate_id=candidate.id,
            evaluated_for=evaluated_for,
            status=status,
            metrics={key: metrics[key] for key in selected_metrics},
            issues=issues,
        )

    return (
        assessment(
            AssessmentAgent.QUALITY,
            quality_status,
            ("sulfur", "t95", "cetane_number"),
            quality_issues,
        ),
        assessment(AssessmentAgent.RELIABILITY, AssessmentStatus.OK, ("risk_index",)),
        assessment(
            AssessmentAgent.OPTIMIZER,
            AssessmentStatus.OK,
            ("throughput", "cost_proxy", "change_size"),
        ),
    )


__all__ = ["assess_blend_candidate", "calculate_blend_metrics"]
