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
    CandidateAction,
    CandidateEvaluation,
    CandidateKind,
    ConstraintStatus,
    OperationMode,
    ProcessState,
    RuntimeConfig,
    ScenarioConfig,
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

    candidates = [hold]
    for step in range(21):
        fraction_b = step / 20
        recipe = {component_a: 1.0 - fraction_b, component_b: fraction_b}
        if all(abs(recipe[key] - current[key]) <= 1e-9 for key in recipe):
            continue
        candidates.append(
            CandidateAction(
                id=f"blend:{component_a}={recipe[component_a]:.2f},{component_b}={fraction_b:.2f}",
                kind=CandidateKind.BLEND,
                blend_mass_fractions=recipe,
                horizon_minutes=horizon_minutes,
                is_model_scenario=scenario.mode is OperationMode.MODEL_DEMO,
            )
        )
        if len(candidates) >= max_candidates:
            break
    return tuple(candidates)


def _metric_value(assessments: tuple[AgentAssessment, ...], name: str) -> float:
    for assessment in assessments:
        metric = assessment.metrics.get(name)
        if metric is not None and metric.value is not None:
            return metric.value
    return 0.0


def _rank_key(
    assessments: tuple[AgentAssessment, ...], candidate_id: str
) -> tuple[float, float, float, float, str]:
    return (
        _metric_value(assessments, "risk_index"),
        -_metric_value(assessments, "throughput"),
        _metric_value(assessments, "cost_proxy"),
        _metric_value(assessments, "change_size"),
        candidate_id,
    )


def _candidate_assessments(
    state: ProcessState,
    candidate: CandidateAction,
    scenario: ScenarioConfig,
) -> tuple[AgentAssessment, ...]:
    if scenario.mode is OperationMode.MODEL_DEMO:
        return assess_blend_candidate(state, candidate, scenario)
    if candidate.kind is not CandidateKind.HOLD:
        raise ValueError("stage-1 history and hybrid modes support hold only")
    quality = predict_quality(state, pd.DataFrame(), None, scenario)
    reliability = assess_reliability(state, scenario)
    return (quality, reliability)


def evaluate_candidates(
    state: ProcessState,
    candidates: tuple[CandidateAction, ...],
    scenario: ScenarioConfig,
) -> tuple[CandidateEvaluation, ...]:
    """Evaluate candidates through agents and the single hard-constraint filter."""
    evaluations: list[CandidateEvaluation] = []
    for candidate in candidates:
        assessments = _candidate_assessments(state, candidate, scenario)
        checks = check_constraints(state, candidate, assessments, scenario)
        feasible = bool(checks) and all(item.status is ConstraintStatus.PASS for item in checks)
        evaluations.append(
            CandidateEvaluation(
                candidate=candidate,
                assessments=assessments,
                checks=checks,
                feasible=feasible,
                rank_key=_rank_key(assessments, candidate.id) if feasible else None,
            )
        )
    return tuple(evaluations)


__all__ = ["evaluate_candidates", "generate_candidates"]
