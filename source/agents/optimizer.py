"""Candidate generation and stage-1 evaluation.

ML owns candidate semantics. Backend owns the shared evaluation envelope: every
candidate must produce assessments, pass the same constraints and expose a rank
key before it can become a recommendation.
"""

from __future__ import annotations

import pandas as pd

from source.agents.effects import assess_blend_candidate
from source.agents.quality import predict_quality
from source.agents.reliability import assess_reliability
from source.constraints import check_constraints
from source.contracts import (
    AgentAssessment,
    AssessmentStatus,
    CandidateAction,
    CandidateEvaluation,
    CandidateKind,
    ConstraintStatus,
    OperationMode,
    ProcessState,
    RuntimeConfig,
    ScenarioConfig,
    Severity,
)


def generate_candidates(
    state: ProcessState,
    scenario: ScenarioConfig,
    config: RuntimeConfig | None = None,
) -> tuple[CandidateAction, ...]:
    """Generate the deterministic stage-1 grid and an explicit hold candidate."""
    if state.mode is not scenario.mode:
        raise ValueError("state and scenario modes must match")
    horizon_minutes = config.horizon_minutes if config is not None else 60
    max_candidates = config.max_candidates if config is not None else 125
    hold = CandidateAction(
        id="hold",
        kind=CandidateKind.HOLD,
        horizon_minutes=horizon_minutes,
        is_model_scenario=scenario.mode is OperationMode.MODEL_DEMO,
    )
    if scenario.mode is not OperationMode.MODEL_DEMO or not scenario.blend_components:
        return (hold,)
    if len(scenario.blend_components) != 2:
        raise ValueError("stage-1 blending requires exactly two components")

    component_a, component_b = (component.id for component in scenario.blend_components)
    current = scenario.current_blend_mass_fractions
    if set(current) != {component_a, component_b} or abs(sum(current.values()) - 1.0) > 1e-9:
        raise ValueError("current blend must contain both components and sum to one")

    recipes: list[dict[str, float]] = []
    for step in range(21):
        fraction_b = step / 20
        recipe = {component_a: 1.0 - fraction_b, component_b: fraction_b}
        if all(abs(recipe[key] - current[key]) <= 1e-9 for key in recipe):
            continue
        recipes.append(recipe)

    candidate_count = 1 + len(recipes)
    if candidate_count > max_candidates:
        raise ValueError(
            f"candidate grid requires {candidate_count} candidates including hold, "
            f"but max_candidates={max_candidates}"
        )

    candidates = [hold]
    for recipe in recipes:
        fraction_b = recipe[component_b]
        candidates.append(
            CandidateAction(
                id=f"blend:{component_a}={recipe[component_a]:.2f},{component_b}={fraction_b:.2f}",
                kind=CandidateKind.BLEND,
                blend_mass_fractions=recipe,
                horizon_minutes=horizon_minutes,
                is_model_scenario=scenario.mode is OperationMode.MODEL_DEMO,
            )
        )
    return tuple(candidates)


def _metric_value(assessments: tuple[AgentAssessment, ...], name: str) -> float | None:
    for assessment in assessments:
        metric = assessment.metrics.get(name)
        if metric is not None and metric.value is not None:
            return metric.value
    return None


def _rank_key(
    assessments: tuple[AgentAssessment, ...],
    candidate_id: str,
    active_criteria: tuple[str, ...],
) -> tuple[float, float, float, float, str] | None:
    values = {
        name: _metric_value(assessments, name)
        for name in ("risk_index", "throughput", "cost_proxy", "change_size")
    }
    if any(values.get(name) is None for name in active_criteria):
        return None
    return (
        values["risk_index"] or 0.0,
        -(values["throughput"] or 0.0),
        values["cost_proxy"] or 0.0,
        values["change_size"] or 0.0,
        candidate_id,
    )


def _candidate_assessments(
    state: ProcessState,
    candidate: CandidateAction,
    scenario: ScenarioConfig,
    features: pd.DataFrame | None = None,
    model: object | None = None,
) -> tuple[AgentAssessment, ...]:
    if scenario.mode is OperationMode.MODEL_DEMO:
        return assess_blend_candidate(state, candidate, scenario)
    if candidate.kind is not CandidateKind.HOLD:
        raise ValueError("stage-1 history and hybrid modes support hold only")
    quality = predict_quality(
        state,
        pd.DataFrame() if features is None else features,
        model,
        scenario,
    )
    reliability = assess_reliability(state, scenario)
    return (quality, reliability)


def evaluate_candidates(
    state: ProcessState,
    candidates: tuple[CandidateAction, ...],
    scenario: ScenarioConfig,
    *,
    features: pd.DataFrame | None = None,
    model: object | None = None,
) -> tuple[CandidateEvaluation, ...]:
    """Evaluate candidates through agents and the single hard-constraint filter."""
    evaluations: list[CandidateEvaluation] = []
    for candidate in candidates:
        assessments = _candidate_assessments(state, candidate, scenario, features, model)
        checks = check_constraints(state, candidate, assessments, scenario)
        rank_key = _rank_key(assessments, candidate.id, scenario.active_criteria)
        assessments_available = all(
            assessment.status is not AssessmentStatus.UNAVAILABLE
            and all(issue.severity is not Severity.BLOCKING for issue in assessment.issues)
            for assessment in assessments
        )
        feasible = (
            bool(checks)
            and all(item.status is ConstraintStatus.PASS for item in checks)
            and assessments_available
            and rank_key is not None
        )
        evaluations.append(
            CandidateEvaluation(
                candidate=candidate,
                assessments=assessments,
                checks=checks,
                feasible=feasible,
                rank_key=rank_key if feasible else None,
            )
        )
    return tuple(evaluations)


__all__ = ["evaluate_candidates", "generate_candidates"]
