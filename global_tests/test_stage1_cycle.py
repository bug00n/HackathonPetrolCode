"""Stage-1 recommendation-cycle tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from source.config import load_runtime_config, load_scenario
from source.contracts import DecisionContext, RecommendationStatus, ScenarioConfig
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


def _run_scenario(runtime_config, scenario: ScenarioConfig, tmp_path: Path):
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


def test_complete_product_passport_allows_stable_hold(
    runtime_config,
    tmp_path: Path,
) -> None:
    """A stable recipe passes sulfur, T95 and cetane conservative bounds."""
    result = _run_demo(runtime_config, "blend_normal", tmp_path)

    assert result.status is RecommendationStatus.HOLD
    assert result.selected is not None
    assert [check.status.value for check in result.selected.checks[:3]] == ["pass"] * 3


def test_complete_product_passport_allows_risk_recommendation(
    runtime_config,
    tmp_path: Path,
) -> None:
    """The selected recipe must pass all three product-quality constraints."""
    result = _run_demo(runtime_config, "blend_risk", tmp_path)

    assert result.schema_version == "1.1"
    assert result.status is RecommendationStatus.RECOMMEND
    assert result.baseline is not None
    assert result.baseline.feasible is False
    assert result.selected is not None
    assert result.selected.candidate.blend_mass_fractions == pytest.approx({"A": 0.891, "B": 0.099})
    assert result.selected.candidate.additive_mass_fraction == pytest.approx(0.01)
    assert all(check.status.value == "pass" for check in result.selected.checks)
    legacy_payload = result.model_dump(mode="json")
    legacy_payload["schema_version"] = "1.0"
    with pytest.raises(ValueError, match="requires recommendation schema 1.1"):
        type(result).model_validate(legacy_payload)


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


@pytest.mark.parametrize("missing_field", ["t95", "cetane_number"])
def test_stage1_abstains_when_required_product_property_is_unassessed(
    runtime_config,
    tmp_path: Path,
    missing_field: str,
) -> None:
    """Unknown T95 or cetane data must never be treated as a passed constraint."""
    payload = load_scenario("config/scenarios/blend_normal.json").model_dump(mode="json")
    payload["blend_components"][0][missing_field] = None
    scenario = ScenarioConfig.model_validate(payload)

    result = _run_scenario(runtime_config, scenario, tmp_path)

    assert result.status is RecommendationStatus.ABSTAIN
    assert result.selected is None
    assert "UNASSESSED_REQUIRED_PROPERTY" in result.reason_codes


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
