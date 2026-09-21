"""Headless checks for the desktop UI projection and journal helpers."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from source.config import load_runtime_config, load_scenario
from source.contracts import DecisionContext, RecommendationStatus
from source.orchestrator import run_cycle
from source.ui import history_smoke_snapshot, journal_entries, recommendation_to_view


def _view(scenario_id: str, run_dir: Path):
    scenario = load_scenario(f"config/scenarios/{scenario_id}.json")
    result = run_cycle(
        data=None,
        as_of=datetime(2026, 1, 15, 9, tzinfo=UTC),
        model=None,
        scenario=scenario,
        config=load_runtime_config("config/runtime.toml"),
        context=DecisionContext(),
        run_dir=run_dir,
    )
    return result, recommendation_to_view(result, scenario)


@pytest.mark.parametrize(
    ("scenario_id", "expected_status"),
    (
        ("blend_normal", RecommendationStatus.HOLD),
        ("blend_risk", RecommendationStatus.RECOMMEND),
        ("blend_t95_risk", RecommendationStatus.RECOMMEND),
        ("blend_cetane_risk", RecommendationStatus.RECOMMEND),
        ("blend_missing", RecommendationStatus.ABSTAIN),
    ),
)
def test_ui_projection_preserves_backend_status(
    scenario_id: str,
    expected_status: RecommendationStatus,
    tmp_path: Path,
) -> None:
    result, view = _view(scenario_id, tmp_path)

    assert result.status is expected_status
    assert view.status == expected_status.value


def test_ui_exposes_complete_risk_recipe_and_quality_bounds(tmp_path: Path) -> None:
    _, view = _view("blend_risk", tmp_path)

    assert view.baseline_upper == pytest.approx(14.058)
    assert view.selected_upper == pytest.approx(9.306)
    assert view.selected_t95_upper == pytest.approx(354.0)
    assert view.selected_cetane_lower == pytest.approx(52.6)
    assert view.current_fractions == pytest.approx({"A": 0.693, "B": 0.297})
    assert view.proposed_fractions == pytest.approx({"A": 0.891, "B": 0.099})
    assert view.current_additive_fraction == pytest.approx(0.01)
    assert view.proposed_additive_fraction == pytest.approx(0.01)
    quality = {row.name: row for row in view.constraints if row.name in {"T95", "Цетановое число"}}
    assert all(row.status == "В пределах" for row in quality.values())
    additive = next(row for row in view.constraints if row.name == "Доля присадки")
    assert additive.actual == "1 %"
    assert additive.limit == "≤ 3 %"


def test_ui_keeps_missing_sulfur_unavailable(tmp_path: Path) -> None:
    _, view = _view("blend_missing", tmp_path)

    assert view.selected_upper is None
    assert view.action_title == "Требуется ручная проверка"


def test_journal_lists_only_complete_results_newest_first(tmp_path: Path) -> None:
    older = tmp_path / "older" / "result.json"
    newer = tmp_path / "newer" / "result.json"
    incomplete = tmp_path / "incomplete"
    older.parent.mkdir()
    newer.parent.mkdir()
    incomplete.mkdir()
    older.write_text(json.dumps({"run_id": "older"}), encoding="utf-8")
    newer.write_text(json.dumps({"run_id": "newer"}), encoding="utf-8")
    os.utime(older, (1_700_000_000, 1_700_000_000))
    os.utime(newer, (1_800_000_000, 1_800_000_000))

    assert journal_entries(tmp_path) == (newer, older)


def test_history_smoke_snapshot_runs_trusted_artifact(tmp_path: Path) -> None:
    from global_tests.test_stage7_history_serving import _prepared_data, _write_point_model
    from source.data.prepare import write_prepared_dataset

    data = _prepared_data()
    dataset_path = write_prepared_dataset(data, tmp_path / "processed")
    model_dir = _write_point_model(tmp_path / "models", data.manifest.tag_dictionary_sha256)

    payload = history_smoke_snapshot(
        dataset_path,
        model_dir,
        "2026-01-15T09:00:00Z",
        run_dir=tmp_path / "runs",
    )

    assert payload["status"] == "abstain"
    assert payload["model_id"] == model_dir.name
    assert "UNCERTAINTY_UNAVAILABLE" in payload["reason_codes"]
