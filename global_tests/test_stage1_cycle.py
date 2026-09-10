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


def test_stage1_holds_when_current_blend_is_feasible(
    runtime_config,
    tmp_path: Path,
) -> None:
    """Normal model-demo scenario should preserve the current recipe."""
    result = _run_demo(runtime_config, "blend_normal", tmp_path)

    assert result.status is RecommendationStatus.HOLD
    assert result.selected is not None
    assert result.selected.candidate.id == "hold"
    assert result.selected.candidate.blend_mass_fractions == {}
    assert _metric(result.selected, "sulfur").upper == pytest.approx(9.4)


def test_stage1_recommends_feasible_blend_when_current_violates_sulfur(
    runtime_config,
    tmp_path: Path,
) -> None:
    """Risk model-demo scenario should select the cheapest feasible recipe on the grid."""
    result = _run_demo(runtime_config, "blend_risk", tmp_path)

    assert result.status is RecommendationStatus.RECOMMEND
    assert result.baseline is not None
    assert result.baseline.feasible is False
    assert result.selected is not None
    assert result.selected.candidate.blend_mass_fractions == {"A": 0.9, "B": 0.1}
    assert _metric(result.selected, "sulfur").upper == pytest.approx(9.4)
    assert "QUALITY_LIMIT" in result.reason_codes


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
