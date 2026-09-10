"""Candidate generation owned by ML; hard constraint checks stay in backend."""

from __future__ import annotations

from source.contracts import (
    CandidateAction,
    CandidateKind,
    OperationMode,
    ProcessState,
    ScenarioConfig,
)


def generate_candidates(state: ProcessState, scenario: ScenarioConfig) -> list[CandidateAction]:
    """Generate the deterministic stage-1 grid and an explicit hold candidate."""
    if state.mode is not scenario.mode:
        raise ValueError("state and scenario modes must match")
    hold = CandidateAction(
        id="hold",
        kind=CandidateKind.HOLD,
        horizon_minutes=60,
        is_model_scenario=scenario.mode is OperationMode.MODEL_DEMO,
    )
    if scenario.mode is not OperationMode.MODEL_DEMO or not scenario.blend_components:
        return [hold]
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
                horizon_minutes=60,
                is_model_scenario=scenario.mode.value == "model_demo",
            )
        )
    return candidates


__all__ = ["generate_candidates"]
