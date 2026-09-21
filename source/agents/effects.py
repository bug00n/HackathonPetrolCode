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
    CetaneAdditiveSpec,
    EstimateBasis,
    IntervalKind,
    Issue,
    MetricEstimate,
    OperationMode,
    ProcessState,
    ProductGrade,
    ScenarioConfig,
    Severity,
    Unit,
)

FRACTION_TOLERANCE = 1e-9


def _recipe(candidate: CandidateAction, scenario: ScenarioConfig) -> tuple[dict[str, float], float]:
    component_ids = {component.id for component in scenario.blend_components}
    if len(component_ids) != 2:
        raise ValueError("stage-1 blending requires exactly two components")
    current = scenario.current_blend_mass_fractions
    if set(current) != component_ids:
        raise ValueError("current blend must contain every component")

    if candidate.kind is CandidateKind.HOLD:
        recipe = dict(current)
        additive_fraction = scenario.current_additive_mass_fraction
    elif candidate.kind is CandidateKind.BLEND:
        recipe = dict(candidate.blend_mass_fractions)
        additive_fraction = candidate.additive_mass_fraction
    else:
        raise ValueError("stage-1 effects cannot evaluate setpoint actions")

    if set(recipe) != component_ids:
        raise ValueError("blend recipe must contain every scenario component exactly once")
    if abs(sum(recipe.values()) + additive_fraction - 1.0) > FRACTION_TOLERANCE:
        raise ValueError("blend and additive mass fractions must sum to one")
    if additive_fraction and scenario.cetane_additive is None:
        raise ValueError("additive dose needs a configured cetane additive model")
    return recipe, additive_fraction


def _weighted(
    recipe: Mapping[str, float],
    values: Mapping[str, float | None],
    *,
    normalize: bool = False,
) -> float | None:
    if any(values[component_id] is None for component_id, weight in recipe.items() if weight > 0):
        return None
    total = sum(weight * (values[component_id] or 0.0) for component_id, weight in recipe.items())
    return total / sum(recipe.values()) if normalize else total


def _additive_gain(spec: CetaneAdditiveSpec | None, dose: float) -> float:
    """Interpolate the explicit scenario response curve without extrapolation."""
    if dose == 0:
        return 0.0
    if spec is None or dose > spec.max_mass_fraction + FRACTION_TOLERANCE:
        raise ValueError("additive dose is outside the configured scenario model")
    for left, right in zip(spec.response_curve, spec.response_curve[1:], strict=False):
        if left.mass_fraction <= dose <= right.mass_fraction:
            share = (dose - left.mass_fraction) / (right.mass_fraction - left.mass_fraction)
            return left.cetane_gain + share * (right.cetane_gain - left.cetane_gain)
    raise ValueError("additive response curve does not cover the requested dose")


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
    """Calculate the complete synthetic product passport and ranking proxies."""
    if scenario.mode is not OperationMode.MODEL_DEMO:
        raise ValueError("stage-1 blending effects are only valid in model_demo mode")
    if scenario.total_mass_t is None:
        raise ValueError("blend scenario needs total_mass_t")

    recipe, additive_fraction = _recipe(candidate, scenario)
    components = {component.id: component for component in scenario.blend_components}

    sulfur_value = _weighted(
        recipe, {key: component.sulfur.value for key, component in components.items()}
    )
    sulfur_upper = _weighted(
        recipe, {key: component.sulfur.upper for key, component in components.items()}
    )
    t95_value = _weighted(
        recipe,
        {
            key: None if component.t95 is None else component.t95.value
            for key, component in components.items()
        },
        normalize=True,
    )
    t95_upper = _weighted(
        recipe,
        {
            key: None if component.t95 is None else component.t95.upper
            for key, component in components.items()
        },
        normalize=True,
    )
    cetane_value = _weighted(
        recipe,
        {
            key: None if component.cetane_number is None else component.cetane_number.value
            for key, component in components.items()
        },
        normalize=True,
    )
    cetane_lower = _weighted(
        recipe,
        {
            key: None if component.cetane_number is None else component.cetane_number.lower
            for key, component in components.items()
        },
        normalize=True,
    )

    def density_bound(field: str) -> float | None:
        if additive_fraction:
            return None
        total_volume = 0.0
        for key, weight in recipe.items():
            if weight <= 0:
                continue
            estimate = components[key].density
            if estimate is None:
                return None
            value = getattr(estimate, field)
            if value is None or value <= 0:
                return None
            total_volume += weight / value
        return 1.0 / total_volume if total_volume > 0 else None

    density_value = density_bound("value")
    density_lower = density_bound("upper")
    density_upper = density_bound("lower")
    cetane_gain = _additive_gain(scenario.cetane_additive, additive_fraction)
    if cetane_value is not None:
        cetane_value += cetane_gain
    if cetane_lower is not None:
        cetane_lower += cetane_gain
    diesel_fraction = sum(recipe.values())
    risk_index = (
        sum(recipe[key] * component.risk_index for key, component in components.items())
        / diesel_fraction
    )
    cost_proxy = sum(
        recipe[key] * component.cost_proxy_per_t for key, component in components.items()
    )
    if scenario.cetane_additive is not None:
        cost_proxy += additive_fraction * scenario.cetane_additive.cost_proxy_per_t
    change_size = sum(
        abs(recipe[key] - scenario.current_blend_mass_fractions.get(key, 0.0)) for key in recipe
    ) + abs(additive_fraction - scenario.current_additive_mass_fraction)
    assumptions = (
        tuple(scenario.assumptions)
        + (
            "Sulfur and its scenario upper bound are mixed by mass fraction.",
            "T95 and base cetane number are linear scenario approximations over diesel components.",
            (
                "The additive has zero modeled sulfur/T95 effect; "
                "cetane gain follows the configured curve."
            ),
            "Risk and cost are scenario proxies, not plant safety or currency estimates.",
        )
        + (() if scenario.cetane_additive is None else scenario.cetane_additive.assumptions)
    )

    def estimate(
        value: float | None,
        unit: str,
        basis: EstimateBasis,
        reference: str,
        *,
        lower: float | None = None,
        upper: float | None = None,
        interval_kind: IntervalKind = IntervalKind.NONE,
    ) -> MetricEstimate:
        return MetricEstimate(
            value=value,
            lower=lower,
            upper=upper,
            unit=unit,
            basis=basis,
            interval_kind=interval_kind,
            interval_level=None,
            reference=reference,
            assumptions=assumptions,
        )

    sulfur_interval = IntervalKind.SCENARIO_BOUND if sulfur_upper is not None else IntervalKind.NONE
    operating_target = scenario.operating_sulfur_target
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
            Unit.CELSIUS.value,
            EstimateBasis.FORMULA,
            "scenario linear T95 blend assumption",
            upper=t95_upper,
            interval_kind=(
                IntervalKind.SCENARIO_BOUND if t95_upper is not None else IntervalKind.NONE
            ),
        ),
        "cetane_number": estimate(
            cetane_value,
            Unit.CETANE.value,
            EstimateBasis.FORMULA,
            "scenario linear cetane blend and additive response curve",
            lower=cetane_lower,
            interval_kind=(
                IntervalKind.SCENARIO_BOUND if cetane_lower is not None else IntervalKind.NONE
            ),
        ),
        "sulfur_operating_target": estimate(
            operating_target,
            Unit.MG_KG.value,
            EstimateBasis.FORMULA,
            "ScenarioConfig.sulfur_margin_mgkg",
        ),
        "density": estimate(
            density_value,
            Unit.DENSITY.value,
            EstimateBasis.FORMULA,
            "mass/volume density blend",
            lower=density_lower,
            upper=density_upper,
            interval_kind=(
                IntervalKind.SCENARIO_BOUND
                if density_lower is not None or density_upper is not None
                else IntervalKind.NONE
            ),
        ),
        "additive_mass_fraction": estimate(
            additive_fraction,
            Unit.DIMENSIONLESS.value,
            EstimateBasis.FORMULA,
            "config scenario cetane_additive",
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
    quality_issues: tuple[Issue, ...] = ()
    quality_status = AssessmentStatus.OK
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
    if t95.value is None or t95.upper is None:
        quality_issues += (
            Issue(
                code="UNASSESSED_REQUIRED_PROPERTY",
                severity=Severity.BLOCKING,
                signal_id="blend:t95",
                detail="T95 or its required upper scenario bound is unavailable.",
                source_ref=f"scenario:{scenario.id}",
            ),
        )
    if scenario.product_grade is not ProductGrade.HDS_DIESEL and (
        cetane.value is None or cetane.lower is None
    ):
        quality_issues += (
            Issue(
                code="UNASSESSED_REQUIRED_PROPERTY",
                severity=Severity.BLOCKING,
                signal_id="blend:cetane_number",
                detail="Cetane number or its required lower scenario bound is unavailable.",
                source_ref=f"scenario:{scenario.id}",
            ),
        )
    if any(issue.severity is Severity.BLOCKING for issue in quality_issues):
        quality_status = AssessmentStatus.UNAVAILABLE
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
            ("sulfur", "sulfur_operating_target", "t95", "cetane_number", "density"),
            quality_issues,
        ),
        assessment(AssessmentAgent.RELIABILITY, AssessmentStatus.OK, ("risk_index",)),
        assessment(
            AssessmentAgent.OPTIMIZER,
            AssessmentStatus.OK,
            ("throughput", "cost_proxy", "change_size", "additive_mass_fraction"),
        ),
    )


__all__ = ["assess_blend_candidate", "calculate_blend_metrics"]
