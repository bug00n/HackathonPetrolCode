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
from source.ui import journal_entries, recommendation_to_view


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
        ("blend_normal", RecommendationStatus.ABSTAIN),
        ("blend_risk", RecommendationStatus.ABSTAIN),
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


def test_ui_does_not_offer_risk_recipe_or_invent_future_metrics(tmp_path: Path) -> None:
    _, view = _view("blend_risk", tmp_path)

    assert view.baseline_upper == pytest.approx(14.2)
    assert view.selected_upper is None
    assert view.current_fractions == {"A": 0.7, "B": 0.3}
    assert view.proposed_fractions == {"A": 0.7, "B": 0.3}
    future = {row.name: row for row in view.constraints if row.name in {"T95", "Цетановое число"}}
    assert set(future) == {"T95", "Цетановое число"}
    assert all(row.status == "Не оценено" for row in future.values())
    assert all(row.actual == "—" for row in future.values())


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
