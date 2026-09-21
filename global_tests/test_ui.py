"""Headless checks for the desktop UI projection and journal helpers."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

import source.ui_data as ui_data_module
from source.config import load_runtime_config, load_scenario
from source.contracts import (
    AgentAssessment,
    AssessmentAgent,
    AssessmentStatus,
    CandidateAction,
    CandidateEvaluation,
    CandidateKind,
    DatasetManifest,
    DecisionContext,
    EstimateBasis,
    IntervalKind,
    MetricEstimate,
    OperationMode,
    Recommendation,
    RecommendationStatus,
)
from source.data.prepare import PreparedData, write_prepared_dataset
from source.orchestrator import run_cycle
from source.ui import (
    format_action_shadow_payload,
    format_history_replay_view,
    format_hybrid_blend_view,
    format_v2_forecast_payload,
    history_replay_to_view,
    history_smoke_snapshot,
    journal_entries,
    recommendation_to_view,
    ui_history_snapshot,
    ui_hybrid_snapshot,
    ui_stage_snapshot,
)

FIXTURES = Path(__file__).parent / "fixtures"
AS_OF = datetime(2026, 1, 15, 9, tzinfo=UTC)


def _ui_dataset(tmp_path: Path) -> Path:
    manifest = DatasetManifest.model_validate_json(
        (FIXTURES / "data/manifest.json").read_text(encoding="utf-8")
    ).model_copy(update={"dataset_id": "ui0000000001"})
    telemetry = pd.DataFrame(
        {
            "timestamp": [AS_OF.isoformat()],
            "avt:F65": [101.0],
            "avt:T20": [260.0],
            "ht:P8": [0.15],
            "ht:F19": [205.0],
            "ht:T11": [315.0],
            "ht:F26": [78.0],
            "ht:F2": [1200.0],
            "ht:F22": [40.0],
            "ht:F25": [80.0],
        }
    )
    quality = pd.DataFrame(
        [
            {
                "observation_id": "pak-sulfur-ui",
                "signal_id": "ht:2:Mg.Sulfur",
                "stage": "ht",
                "source": "pak",
                "measured_at": AS_OF.isoformat(),
                "available_at": AS_OF.isoformat(),
                "value": 8.7,
                "unit": "mg/kg",
                "validity": "valid",
                "source_ref": "fixture-pak",
            },
            {
                "observation_id": "pak-density-ui",
                "signal_id": "ht:density_15c",
                "stage": "ht",
                "source": "pak",
                "measured_at": AS_OF.isoformat(),
                "available_at": AS_OF.isoformat(),
                "value": 833.0,
                "unit": "kg/m3",
                "validity": "valid",
                "source_ref": "fixture-pak",
            },
        ]
    )
    data = PreparedData(
        telemetry=telemetry,
        quality=quality,
        issues=pd.DataFrame(columns=["code", "severity", "signal_id", "detail", "source_ref"]),
        manifest=manifest,
        feature_order=tuple(column for column in telemetry.columns if column != "timestamp"),
    )
    return write_prepared_dataset(data, tmp_path)


def _history_recommendation(*, actionable: bool = False) -> Recommendation:
    def evaluation(candidate: CandidateAction, point: float, upper: float) -> CandidateEvaluation:
        return CandidateEvaluation(
            candidate=candidate,
            assessments=(
                AgentAssessment(
                    agent=AssessmentAgent.QUALITY,
                    state_id="history-ui-state",
                    candidate_id=candidate.id,
                    evaluated_for=AS_OF,
                    status=AssessmentStatus.OK,
                    metrics={
                        "sulfur": MetricEstimate(
                            value=point,
                            lower=None,
                            upper=upper,
                            unit="mg/kg",
                            basis=EstimateBasis.FORECAST,
                            interval_kind=IntervalKind.EMPIRICAL,
                            interval_level=0.95,
                            reference="fixture",
                        )
                    },
                ),
            ),
            checks=(),
            feasible=True,
            rank_key=(0.0, 0.0, 0.0, 0.0, candidate.id),
        )

    baseline = evaluation(
        CandidateAction(id="hold", kind=CandidateKind.HOLD, horizon_minutes=60),
        9.8,
        11.0,
    )
    selected = None
    status = RecommendationStatus.ABSTAIN
    reasons = ("ACTION_MODEL_UNAVAILABLE",)
    if actionable:
        selected = evaluation(
            CandidateAction(
                id="setpoints-p8",
                kind=CandidateKind.SETPOINTS,
                setpoints={"ht:P8": 0.145},
                horizon_minutes=60,
            ),
            8.9,
            9.6,
        )
        status = RecommendationStatus.RECOMMEND
        reasons = ()
    return Recommendation(
        run_id="history-ui-run",
        state_id="history-ui-state",
        as_of=AS_OF,
        scenario_id="history",
        mode=OperationMode.HISTORY,
        status=status,
        baseline=baseline,
        selected=selected,
        alternatives=(),
        reason_codes=reasons,
        explanation="fixture history replay",
        assumptions=("fixture",),
        model_id="forecast-fixture",
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


def test_stage_snapshots_show_avt_and_hydrotreating_without_controls(tmp_path: Path) -> None:
    dataset = _ui_dataset(tmp_path)

    avt = ui_stage_snapshot("avt", dataset, AS_OF)
    hydro = ui_stage_snapshot("hydrotreating", dataset, AS_OF)

    assert avt.status == "ready"
    avt_rows = {row.signal_id: row for row in avt.rows}
    assert avt_rows["avt:F65"].freshness == "fresh"
    assert "unit is unknown" in avt_rows["avt:F65"].read_only_reason
    assert avt_rows["avt:F12"].freshness == "missing"
    hydro_rows = {row.signal_id: row for row in hydro.rows}
    assert hydro_rows["ht:2:Mg.Sulfur"].value == pytest.approx(8.7)
    assert hydro_rows["ht:P8"].freshness == "fresh"
    assert "verified action artifact" in hydro_rows["ht:P8"].read_only_reason
    assert "context-only" in hydro_rows["ht:T11"].read_only_reason


def test_stage_snapshot_reports_missing_dataset_gracefully(tmp_path: Path) -> None:
    snapshot = ui_stage_snapshot("avt", tmp_path / "missing", AS_OF)

    assert snapshot.status == "error"
    assert "missing prepared dataset files" in snapshot.message


def test_history_snapshot_with_forecast_only_is_not_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ui_data_module,
        "replay_command",
        lambda *args, **kwargs: _history_recommendation(),
    )

    view = ui_history_snapshot("dataset", "model", AS_OF)

    assert view.status == "ready"
    assert view.action_state == "not_actionable"
    assert view.sulfur_upper == pytest.approx(11.0)
    assert "ACTION_MODEL_UNAVAILABLE" in view.reason_codes
    assert "Raw recommendation JSON" in format_history_replay_view(view)


def test_history_projection_marks_verified_setpoint_result_actionable() -> None:
    view = history_replay_to_view(_history_recommendation(actionable=True))

    assert view.action_state == "actionable"
    assert view.selected_kind == "setpoints"
    assert view.upper_status == "pass"


def test_hybrid_snapshot_uses_history_forecast_without_imputing_component_a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = ui_hybrid_snapshot(None, None, AS_OF)
    assert empty.status == "empty"
    assert "FORECAST_UNAVAILABLE" in empty.reason_codes

    monkeypatch.setattr(
        ui_data_module,
        "replay_command",
        lambda *args, **kwargs: _history_recommendation(),
    )

    view = ui_hybrid_snapshot("dataset", "model", AS_OF)

    assert view.status == "ready"
    assert view.component_sulfur_upper == pytest.approx(11.0)
    assert view.blend_sulfur_upper == pytest.approx(17.0)
    assert view.constraint_status == "fail"
    assert "Hybrid sulfur-only blend" in format_hybrid_blend_view(view)


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
