"""The synthetic blend helper must agree with the decision engine."""

import json
from pathlib import Path

import pytest

from source.agents.blend_assist import feasible_blends
from source.agents.optimizer import evaluate_candidates, generate_candidates
from source.config import load_runtime_config, load_scenario
from source.contracts import ScenarioConfig
from source.main import run_model_demo
from source.orchestrator import _model_demo_state
from source.ui_what_if import assist_editor_values, editor_values, scenario_from_editor


def test_editor_keeps_changed_share_and_only_repairs_recipe() -> None:
    scenario = load_scenario("config/scenarios/blend_risk.json")
    values = editor_values(scenario)
    values["B.fraction"] = "10"
    options = assist_editor_values(scenario, values, frozenset({"B.fraction"}))
    assert options
    for option, updated in options:
        assert updated["B.fraction"] == "10"
        assert updated["A.sulfur.value"] == values["A.sulfur.value"]
        assert updated["total_mass_t"] == values["total_mass_t"]
        assert scenario_from_editor(scenario, updated)
        assert option.evaluation.feasible


def test_editor_explains_impossible_fixed_stock() -> None:
    scenario = load_scenario("config/scenarios/blend_risk.json")
    values = editor_values(scenario)
    values["B.fraction"] = "40"
    with pytest.raises(ValueError, match="доступно 30"):
        assist_editor_values(scenario, values, frozenset({"B.fraction"}))


def test_lowering_additive_cap_repairs_the_current_dose() -> None:
    scenario = load_scenario("config/scenarios/blend_risk.json")
    values = editor_values(scenario) | {"dose": "3", "max_dose": "1"}
    options = assist_editor_values(scenario, values, frozenset())
    assert options
    assert all(float(updated["dose"]) <= 1 for _, updated in options)


def test_continuous_search_finds_narrow_feasible_window() -> None:
    base = load_scenario("config/scenarios/blend_risk.json")
    payload = base.model_dump(mode="json")
    payload["blend_components"][0]["available_mass_t"] = 100
    payload["blend_components"][1]["available_mass_t"] = 100
    payload["blend_components"][0]["sulfur"].update(value=0, upper=0)
    payload["blend_components"][1]["sulfur"].update(value=100, upper=100)
    payload["blend_components"][0]["cetane_number"].update(value=0, lower=0)
    payload["blend_components"][1]["cetane_number"].update(value=100, lower=100)
    payload["constraints"] = [
        payload["constraints"][0] | {"upper": 48.9},
        payload["constraints"][1] | {"upper": 400},
        payload["constraints"][2] | {"lower": 48.1},
    ]
    payload["cetane_additive"]["available_mass_t"] = 0
    scenario = ScenarioConfig.model_validate(payload)
    from datetime import UTC, datetime

    state = _model_demo_state(datetime.now(UTC), scenario)
    found = feasible_blends(scenario, dict(scenario.current_blend_mass_fractions), 0)
    assert found
    assert all(0.481 <= item.fractions["B"] <= 0.489 for item in found)
    assert any(
        item.feasible
        for item in evaluate_candidates(state, generate_candidates(state, scenario), scenario)
    )
    bounded = load_runtime_config(Path("config/runtime.toml")).model_copy(
        update={"max_candidates": 85}
    )
    candidates = generate_candidates(state, scenario, bounded)
    assert len(candidates) == 85
    assert any(item.feasible for item in evaluate_candidates(state, candidates, scenario))


def test_fixed_recipe_and_editor_context_are_journaled(tmp_path: Path) -> None:
    preset = load_scenario("config/scenarios/blend_risk.json")
    values = editor_values(preset) | {"B.fraction": "10"}
    _, updated = assist_editor_values(preset, values, frozenset({"B.fraction"}))[0]
    scenario = scenario_from_editor(preset, updated)
    context = {"original_values": values, "displayed_values": updated, "assisted": True}
    result = run_model_demo(
        preset.id,
        run_dir=tmp_path,
        scenario_override=scenario,
        recipe_mode="evaluate",
        editor_context=context,
    )
    assert result.selected is not None
    assert result.selected.candidate.kind.value == "hold"
    recorded = json.loads((tmp_path / result.run_id / "input.json").read_text())
    assert recorded["editor_context"] == context
    assert (
        recorded["scenario"]["current_blend_mass_fractions"]
        == scenario.current_blend_mass_fractions
    )
    status = json.loads((tmp_path / result.run_id / "status.json").read_text())
    assert status["status"] == "completed"


def test_failed_run_has_durable_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args, **_kwargs):
        raise RuntimeError("deliberate failure")

    monkeypatch.setattr("source.orchestrator._run_cycle", fail)
    with pytest.raises(RuntimeError, match="deliberate failure"):
        run_model_demo("blend_risk", run_dir=tmp_path)
    statuses = [json.loads(path.read_text()) for path in tmp_path.glob("*/status.json")]
    assert len(statuses) == 1
    assert statuses[0]["status"] == "failed"
