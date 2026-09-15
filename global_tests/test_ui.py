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
from source.ui import (
    format_action_shadow_payload,
    format_v2_forecast_payload,
    journal_entries,
    recommendation_to_view,
)


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


def test_action_shadow_view_separates_domain_validation_and_safety() -> None:
    text = format_action_shadow_payload(
        {
            "control_id": "ht:P8",
            "proposed_delta": 0.001,
            "state": {"baseline_sulfur": 9.5},
            "predicted_sulfur": {"60": 9.4, "120": 9.3, "180": 9.2},
            "sulfur_change": {"60": -0.1, "120": -0.2, "180": -0.3},
            "sulfur_upper": {"60": 10.2, "120": 10.1, "180": 9.9},
            "within_observed_domain": True,
            "model_validated": False,
            "safety_passes": False,
            "reason_codes": ("ACTION_EFFECT_VALIDATION_FAILED",),
        }
    )

    assert "Изменение к hold" in text
    assert "Историческая область: да" in text
    assert "Validation модели: не пройдена" in text
    assert "Верхняя граница серы: не проходит" in text


def test_v2_shadow_view_shows_horizons_and_never_advises() -> None:
    text = format_v2_forecast_payload(
        {
            "as_of": "2025-06-01T12:00:00+00:00",
            "production_status": "shadow_only",
            "forecast": {
                "horizon_probabilities": {"10": 0.1, "20": 0.2, "30": 0.3, "60": 0.4},
                "event_alarm": True,
                "applicable": False,
                "point_60m": 9.4,
                "upper_60m": 10.4,
                "reason_codes": ("OOD",),
            },
        }
    )

    assert "Эпизодный прогноз серы" in text
    assert "60 мин | 40.0%" in text
    assert "не изменяет уставки" in text
    assert "shadow_only" in text


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
