"""Stage-3 backend guardrail tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from source.config import load_runtime_config, load_scenario
from source.contracts import AssessmentStatus, DecisionContext, RecommendationStatus
from source.orchestrator import run_cycle
from source.ui import recommendation_to_view

AS_OF = datetime(2026, 1, 15, 9, tzinfo=UTC)


@pytest.fixture()
def runtime_config():
    """Load the checked-in runtime config."""
    return load_runtime_config("config/runtime.toml")


def _run(scenario, runtime_config, tmp_path: Path, context: DecisionContext | None = None):
    return run_cycle(
        data=None,
        as_of=AS_OF,
        model=None,
        scenario=scenario,
        config=runtime_config,
        context=context or DecisionContext(),
        run_dir=tmp_path,
    )


def _baseline_feasible_cost_scenario(cost_threshold: float):
    scenario = load_scenario("config/scenarios/blend_normal.json")
    thresholds = dict(scenario.materiality_thresholds)
    thresholds["cost_proxy"] = cost_threshold
    components = list(scenario.blend_components)
    components[1] = components[1].model_copy(
        update={"sulfur": components[1].sulfur.model_copy(update={"value": 24.0, "upper": 25.0})}
    )
    return scenario.model_copy(
        update={
            "id": f"blend_materiality_{cost_threshold:g}",
            "blend_components": tuple(components),
            "materiality_thresholds": thresholds,
        }
    )


def test_stage3_keeps_hold_when_improvement_is_not_material(runtime_config, tmp_path: Path) -> None:
    """A sub-threshold cost improvement keeps the valid current recipe."""
    scenario = _baseline_feasible_cost_scenario(cost_threshold=0.05)

    result = _run(scenario, runtime_config, tmp_path)

    assert result.status is RecommendationStatus.HOLD
    assert result.selected is not None
    assert "NO_MATERIAL_IMPROVEMENT" in result.reason_codes


def test_stage3_cooldown_suppresses_repeat_when_baseline_is_feasible(
    runtime_config, tmp_path: Path
) -> None:
    """Cooldown suppresses a repeated material change when hold is feasible."""
    scenario = _baseline_feasible_cost_scenario(cost_threshold=0.005)
    context = DecisionContext(last_recommended_at=AS_OF - timedelta(minutes=30))

    result = _run(scenario, runtime_config, tmp_path, context)

    assert result.status is RecommendationStatus.HOLD
    assert "ACTION_COOLDOWN" in result.reason_codes


def test_material_improvement_does_not_claim_baseline_quality_failure(
    runtime_config, tmp_path: Path
) -> None:
    scenario = _baseline_feasible_cost_scenario(cost_threshold=0.005)

    result = _run(scenario, runtime_config, tmp_path)
    view = recommendation_to_view(result, scenario)

    assert result.status is RecommendationStatus.RECOMMEND
    assert result.baseline is not None and result.baseline.feasible
    assert "MATERIAL_IMPROVEMENT" in result.reason_codes
    assert "текущий режим проходит ограничения" in result.explanation
    assert "cost_proxy" in result.explanation
    assert "проходит проверки" in view.action_detail


def test_stage3_cooldown_does_not_hide_quality_violation(runtime_config, tmp_path: Path) -> None:
    """When hold violates hard constraints, cooldown must not silence a warning."""
    scenario = load_scenario("config/scenarios/blend_risk.json")
    context = DecisionContext(last_recommended_at=AS_OF - timedelta(minutes=30))

    result = _run(scenario, runtime_config, tmp_path, context)

    assert result.status is RecommendationStatus.RECOMMEND
    assert result.selected is not None
    assert "QUALITY_LIMIT" in result.reason_codes


def test_stage3_journal_records_rejection_summary(runtime_config, tmp_path: Path) -> None:
    """Run metadata and trace should summarize why candidates were rejected."""
    scenario = load_scenario("config/scenarios/blend_risk.json")

    result = _run(scenario, runtime_config, tmp_path)
    run_path = tmp_path / result.run_id
    metadata = json.loads((run_path / "metadata.json").read_text(encoding="utf-8"))
    trace = json.loads((run_path / "trace.jsonl").read_text(encoding="utf-8"))

    assert metadata["selection_reason"] == "baseline_infeasible_recommend"
    assert metadata["rejection_summary"]["QUALITY_LIMIT"] >= 1
    assert trace["selection_reason"] == metadata["selection_reason"]
    assert trace["rejection_summary"] == metadata["rejection_summary"]


def test_stage3_rechecks_selected_candidate_with_same_constraints(
    runtime_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A selected candidate must fail closed if its repeated hard checks drift."""
    scenario = load_scenario("config/scenarios/blend_normal.json")

    original = __import__(
        "source.agents.optimizer", fromlist=["assess_blend_candidate"]
    ).assess_blend_candidate

    def complete_quality(*args):
        assessments = original(*args)
        return tuple(
            item.model_copy(update={"status": AssessmentStatus.OK, "issues": ()})
            for item in assessments
        )

    monkeypatch.setattr("source.agents.optimizer.assess_blend_candidate", complete_quality)
    monkeypatch.setattr("source.orchestrator.check_constraints", lambda *args: ())

    with pytest.raises(ValueError, match="selected constraint recheck mismatch"):
        _run(scenario, runtime_config, tmp_path)
