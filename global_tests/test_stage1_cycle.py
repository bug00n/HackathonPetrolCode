"""Stage-1 recommendation-cycle tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from source.config import load_runtime_config, load_scenario
from source.contracts import DecisionContext, RecommendationStatus
from source.orchestrator import run_cycle

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def runtime_config():
    """Load the checked-in runtime config."""
    return load_runtime_config("config/runtime.toml")


def _run_demo(
    runtime_config,
    scenario_name: str,
    tmp_path: Path,
):
    scenario = load_scenario(f"config/scenarios/{scenario_name}.json")
    return _run_scenario(runtime_config, scenario, tmp_path)


def _run_scenario(runtime_config, scenario, tmp_path: Path):
    return run_cycle(
        data=None,
        as_of=datetime(2026, 1, 15, 9, tzinfo=UTC),
        model=None,
        scenario=scenario,
        config=runtime_config,
        context=DecisionContext(),
        run_dir=tmp_path,
    )


def _metric(evaluation, name: str):
    for assessment in evaluation.assessments:
        if name in assessment.metrics:
            return assessment.metrics[name]
    raise AssertionError(f"metric {name} not found")


def test_stage1_holds_complete_normal_blend(
    runtime_config,
    tmp_path: Path,
) -> None:
    """Complete model-demo product properties allow a hold decision."""
    result = _run_demo(runtime_config, "blend_normal", tmp_path)

    assert result.status is RecommendationStatus.HOLD
    assert result.selected is not None
    assert result.selected.candidate.id == "hold"
    assert "NO_MATERIAL_IMPROVEMENT" in result.reason_codes


def test_stage1_recommends_model_recipe_for_risk_blend(
    runtime_config,
    tmp_path: Path,
) -> None:
    """A risky model-demo blend can recommend a synthetic safer recipe."""
    result = _run_demo(runtime_config, "blend_risk", tmp_path)

    assert result.status is RecommendationStatus.RECOMMEND
    assert result.baseline is not None
    assert result.baseline.feasible is False
    assert result.selected is not None
    assert result.selected.candidate.blend_mass_fractions == {"A": 0.9, "B": 0.1}
    assert "QUALITY_LIMIT" in result.reason_codes


def test_stage1_abstains_when_model_demo_t95_is_missing(
    runtime_config,
    tmp_path: Path,
) -> None:
    """Missing scenario passport properties still block model-demo recommendations."""
    scenario = load_scenario("config/scenarios/blend_normal.json")
    first, second = scenario.blend_components
    broken = scenario.model_copy(
        update={"blend_components": (first.model_copy(update={"t95": None}), second)}
    )

    result = _run_scenario(runtime_config, broken, tmp_path)

    assert result.status is RecommendationStatus.ABSTAIN
    assert result.selected is None
    assert "UNASSESSED_REQUIRED_PROPERTY" in result.reason_codes


def test_stage1_abstains_when_required_component_quality_is_missing(
    runtime_config,
    tmp_path: Path,
) -> None:
    """Missing component quality should remain an explained refusal."""
    result = _run_demo(runtime_config, "blend_missing", tmp_path)

    assert result.status is RecommendationStatus.ABSTAIN
    assert result.selected is None
    assert "MISSING_REQUIRED_SIGNAL" in result.reason_codes
    assert "Надёжной рекомендации нет" in result.explanation


def test_stage1_writes_journal_files(
    runtime_config,
    tmp_path: Path,
) -> None:
    """Successful cycles should persist the inspectable run journal."""
    result = _run_demo(runtime_config, "blend_risk", tmp_path)
    run_path = tmp_path / result.run_id

    assert (run_path / "metadata.json").is_file()
    assert (run_path / "input.json").is_file()
    assert (run_path / "candidates.jsonl").is_file()
    assert (run_path / "result.json").is_file()
    saved = json.loads((run_path / "result.json").read_text(encoding="utf-8"))
    assert saved["run_id"] == result.run_id
    assert saved["status"] == "recommend"
