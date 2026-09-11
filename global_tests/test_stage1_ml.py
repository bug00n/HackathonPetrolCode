"""Stage-1 ML acceptance tests independent of backend orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from source.agents.effects import assess_blend_candidate, calculate_blend_metrics
from source.agents.optimizer import generate_candidates
from source.agents.quality import predict_quality
from source.agents.reliability import (
    SeverityFactor,
    assess_confirmed_factors,
    assess_reliability,
    factor_contribution,
)
from source.config import load_scenario
from source.contracts import CandidateAction, CandidateKind, OperationMode, ProcessState

FIXTURES = Path(__file__).parent / "fixtures"


def _demo(name: str) -> tuple[ProcessState, object]:
    payload = json.loads((FIXTURES / f"model_demo/{name}.json").read_text(encoding="utf-8"))
    return ProcessState.model_validate(payload["state"]), load_scenario(
        f"config/scenarios/{name}.json"
    )


def test_history_quality_uses_last_valid_available_measurement() -> None:
    state = ProcessState.model_validate_json(
        (FIXTURES / "contracts/process_state.json").read_text(encoding="utf-8")
    )
    scenario = load_scenario("config/scenarios/history.json")

    assessment = predict_quality(state, pd.DataFrame(), None, scenario)

    assert assessment.status.value == "ok"
    assert assessment.candidate_id == "hold"
    assert assessment.metrics["sulfur"].value == pytest.approx(8.4)
    assert assessment.metrics["sulfur"].basis.value == "measured"
    assert assessment.metrics["sulfur"].upper is None


def test_history_quality_uses_sulfur_not_first_required_signal() -> None:
    state = ProcessState.model_validate_json(
        (FIXTURES / "contracts/process_state.json").read_text(encoding="utf-8")
    )
    sulfur_snapshot = state.signals["ht:2:Mg.Sulfur"]
    assert sulfur_snapshot.selected is not None
    unrelated = sulfur_snapshot.model_copy(
        update={
            "selected": sulfur_snapshot.selected.model_copy(
                update={"id": "unrelated", "signal_id": "unrelated", "value": 123.0}
            )
        }
    )
    state = state.model_copy(update={"signals": {"unrelated": unrelated, **state.signals}})
    scenario = load_scenario("config/scenarios/history.json").model_copy(
        update={"required_signals": ("unrelated", "ht:2:Mg.Sulfur")}
    )

    assessment = predict_quality(state, pd.DataFrame(), None, scenario)

    assert assessment.metrics["sulfur"].value == pytest.approx(8.4)
    assert assessment.metrics["sulfur"].reference != "unrelated"


def test_history_without_confirmed_severity_factors_is_unavailable_not_zero() -> None:
    state = ProcessState.model_validate_json(
        (FIXTURES / "contracts/process_state.json").read_text(encoding="utf-8")
    )
    scenario = load_scenario("config/scenarios/history.json")

    assessment = assess_reliability(state, scenario)

    assert assessment.status.value == "unavailable"
    assert "risk_index" not in assessment.metrics
    assert {issue.code for issue in assessment.issues} == {"RELIABILITY_UNAVAILABLE"}


@pytest.mark.parametrize(
    ("name", "value", "upper"),
    [("blend_normal", 8.4, 9.4), ("blend_risk", 13.2, 14.2)],
)
def test_model_demo_quality_is_calculated_from_components(
    name: str, value: float, upper: float
) -> None:
    state, scenario = _demo(name)

    assessment = predict_quality(state, pd.DataFrame(), None, scenario)

    assert assessment.status.value == "unavailable"
    assert "UNASSESSED_REQUIRED_PROPERTY" in {issue.code for issue in assessment.issues}
    assert assessment.metrics["sulfur"].value == pytest.approx(value)
    assert assessment.metrics["sulfur"].upper == pytest.approx(upper)
    assert assessment.metrics["sulfur"].basis.value == "formula"


def test_missing_component_quality_is_unavailable_not_zero() -> None:
    state, scenario = _demo("blend_missing")

    assessment = predict_quality(state, pd.DataFrame(), None, scenario)

    assert assessment.status.value == "unavailable"
    assert assessment.metrics["sulfur"].value is None
    assert assessment.metrics["sulfur"].upper is None
    assert {"MISSING_REQUIRED_SIGNAL", "UNASSESSED_REQUIRED_PROPERTY"}.issubset(
        {issue.code for issue in assessment.issues}
    )


def test_stage1_refuses_to_emulate_hybrid_mode() -> None:
    state, scenario = _demo("blend_normal")
    state = state.model_copy(update={"mode": OperationMode.HYBRID})
    scenario = scenario.model_copy(update={"mode": OperationMode.HYBRID})

    quality = predict_quality(state, pd.DataFrame(), None, scenario)
    reliability = assess_reliability(state, scenario)
    candidates = generate_candidates(state, scenario)

    assert quality.status.value == "unavailable"
    assert {issue.code for issue in quality.issues} == {"ACTION_MODEL_UNAVAILABLE"}
    assert reliability.status.value == "unavailable"
    assert [candidate.kind for candidate in candidates] == [CandidateKind.HOLD]
    with pytest.raises(ValueError, match="only valid in model_demo"):
        calculate_blend_metrics(candidates[0], scenario)


def test_severity_index_is_maximum_of_visible_linear_contributions() -> None:
    state = ProcessState.model_validate_json(
        (FIXTURES / "contracts/process_state.json").read_text(encoding="utf-8")
    )
    factor = SeverityFactor(
        signal_id="ht:2:Mg.Sulfur",
        unit="mg/kg",
        normal_edge=5.0,
        model_boundary=15.0,
        adverse_direction="higher",
        evidence_ref="synthetic train-only range",
    )

    assessment = assess_confirmed_factors(state, (factor,))

    assert assessment.status.value == "ok"
    assert assessment.metrics["severity:ht:2:Mg.Sulfur"].value == pytest.approx(0.34)
    assert assessment.metrics["risk_index"].value == pytest.approx(0.34)
    assert "not a failure probability" in assessment.metrics["risk_index"].assumptions[0]


def test_severity_contribution_handles_both_directions_and_clamps() -> None:
    higher = SeverityFactor("high", "degC", 300.0, 350.0, "higher", "fixture")
    lower = SeverityFactor("low", "MPa", 2.0, 1.0, "lower", "fixture")

    assert factor_contribution(290.0, higher) == 0.0
    assert factor_contribution(325.0, higher) == pytest.approx(0.5)
    assert factor_contribution(360.0, higher) == 1.0
    assert factor_contribution(2.5, lower) == 0.0
    assert factor_contribution(1.5, lower) == pytest.approx(0.5)
    assert factor_contribution(0.5, lower) == 1.0


def test_candidates_include_hold_and_deterministic_recipe_grid() -> None:
    state, scenario = _demo("blend_risk")

    candidates = generate_candidates(state, scenario)

    assert len(candidates) == 21
    assert candidates[0].kind is CandidateKind.HOLD
    assert sum(candidate.kind is CandidateKind.HOLD for candidate in candidates) == 1
    assert any(candidate.blend_mass_fractions == {"A": 0.9, "B": 0.1} for candidate in candidates)
    assert all(
        abs(sum(candidate.blend_mass_fractions.values()) - 1.0) <= 1e-9
        for candidate in candidates
        if candidate.kind is CandidateKind.BLEND
    )


def test_blend_effects_match_design_and_keep_state_immutable() -> None:
    state, scenario = _demo("blend_risk")
    before = state.model_dump_json()
    candidate = CandidateAction(
        id="blend:A=0.90,B=0.10",
        kind="blend",
        blend_mass_fractions={"A": 0.9, "B": 0.1},
        horizon_minutes=60,
        is_model_scenario=True,
    )

    metrics = calculate_blend_metrics(candidate, scenario)
    assessments = assess_blend_candidate(state, candidate, scenario)

    assert metrics["sulfur"].value == pytest.approx(8.4)
    assert metrics["sulfur"].upper == pytest.approx(9.4)
    assert metrics["cost_proxy"].value == pytest.approx(0.98)
    assert metrics["change_size"].value == pytest.approx(0.4)
    assert [item.agent.value for item in assessments] == ["quality", "reliability", "optimizer"]
    assert state.model_dump_json() == before
