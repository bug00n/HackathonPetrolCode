"""Synthetic what-if contracts and history chart regressions."""

import json
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk

import pytest

from source.config import load_scenario
from source.main import run_model_demo
from source.ui_charts import HistoryChart, history_points
from source.ui_what_if import editor_values, open_editor, scenario_from_editor


def test_what_if_changes_forecast_and_journals_inputs(tmp_path: Path) -> None:
    preset_path = Path("config/scenarios/blend_risk.json")
    before = preset_path.read_bytes()
    preset = load_scenario(preset_path)
    values = editor_values(preset)
    values.update({"A.sulfur.value": "1,0", "A.sulfur.upper": "2"})
    scenario = scenario_from_editor(preset, values)
    original = run_model_demo(preset.id, run_dir=tmp_path)
    changed = run_model_demo(preset.id, run_dir=tmp_path, scenario_override=scenario)

    def baseline_sulfur(result):
        return history_points(
            {"as_of": result.as_of.isoformat(), "baseline": result.baseline.model_dump(mode="json")}
        )[0][1:]

    assert baseline_sulfur(original) != baseline_sulfur(changed)
    assert changed.scenario_id == "blend_risk_what_if"
    assert changed.selected is not None and changed.selected.candidate.is_model_scenario
    assert changed.selected.candidate.kind.value == "blend"
    recorded = json.loads((tmp_path / changed.run_id / "input.json").read_text())
    assert recorded["scenario"]["blend_components"][0]["sulfur"]["value"] == 1
    assert recorded["scenario"]["controls"] == []
    assert preset_path.read_bytes() == before
    reset = scenario_from_editor(preset, editor_values(preset))
    restored = run_model_demo(preset.id, run_dir=tmp_path, scenario_override=reset)
    assert restored.status == original.status
    assert restored.selected.candidate == original.selected.candidate
    assert baseline_sulfur(restored) == baseline_sulfur(original)


@pytest.mark.parametrize(
    "updates",
    [
        {"A.fraction": "99"},
        {"A.available_mass_t": "-1"},
        {"A.sulfur.value": "nan"},
        {"A.sulfur.upper": "inf"},
        {"A.sulfur.upper": "2"},
        {"A.cetane_number.lower": "99"},
        {"A.risk_index": "1.1"},
        {"dose": "4"},
        {"total_mass_t": "0"},
        {"max_dose": "0.5"},
        {"gain_1": "20"},
        {"additive_stock": ""},
    ],
)
def test_what_if_rejects_invalid_inputs(updates: dict[str, str]) -> None:
    preset = load_scenario("config/scenarios/blend_risk.json")
    with pytest.raises(ValueError):
        scenario_from_editor(preset, editor_values(preset) | updates)


@pytest.mark.parametrize(
    "updates",
    [
        {"A.sulfur.value": "", "A.sulfur.upper": ""},
        {"A.available_mass_t": "0", "B.available_mass_t": "0"},
        {"additive_stock": "0"},
        {"gain_1": "0", "gain_2": "0", "gain_3": "0"},
    ],
)
def test_what_if_missing_quality_or_insufficient_resources_abstains(
    updates: dict[str, str],
    tmp_path: Path,
) -> None:
    preset = load_scenario("config/scenarios/blend_risk.json")
    scenario = scenario_from_editor(preset, editor_values(preset) | updates)
    result = run_model_demo(preset.id, run_dir=tmp_path, scenario_override=scenario)
    assert result.status.value == "abstain"
    assert result.selected is None


def test_what_if_cannot_override_with_history(tmp_path: Path) -> None:
    scenario = load_scenario("config/scenarios/history.json")
    with pytest.raises(ValueError, match="synthetic blending"):
        run_model_demo("blend_risk", run_dir=tmp_path, scenario_override=scenario)


def interval_payload() -> dict:
    return {
        "start": "2025-06-01T12:00:00+03:00",
        "cancelled": True,
        "results": [
            {"as_of": "2025-06-01T09:00:00+00:00", "sulfur": {"point": 8, "upper": 11}},
            {"as_of": "2025-06-01T10:00:00+00:00", "sulfur": None},
            {"as_of": "2025-06-01T12:00:00+00:00", "sulfur": {"point": 20, "upper": 22}},
        ],
    }


def test_chart_uses_target_times_and_keeps_missing_values() -> None:
    points = history_points(interval_payload())
    assert len(points) == 3
    assert points[0] == (datetime.fromisoformat("2025-06-01T13:00:00+03:00"), 8, 11)
    assert points[1][1:] == (None, None)
    assert points[-1][0].hour == 16
    assert history_points({"results": []}) == []


def test_chart_accepts_single_recommendation(tmp_path: Path) -> None:
    result = run_model_demo("blend_risk", run_dir=tmp_path)
    points = history_points(result.model_dump(mode="json"))
    assert len(points) == 1
    assert points[0][2] == pytest.approx(9.306)


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


def test_editor_reset_validation_and_chart_on_real_tk() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    try:
        preset = load_scenario("config/scenarios/blend_risk.json")
        received = []
        dialog = open_editor(root, preset, preset, received.append)
        dialog.withdraw()
        entries = [w for w in descendants(dialog) if isinstance(w, ttk.Entry)]
        first = entries[0]
        first.delete(0, "end")
        first.insert(0, "bad input")
        buttons = {w.cget("text"): w for w in descendants(dialog) if isinstance(w, ttk.Button)}
        buttons["Пересчитать what-if"].invoke()
        assert not received and dialog.winfo_exists()
        buttons["Сбросить к сценарию"].invoke()
        assert first.get() == "6"
        buttons["Пересчитать what-if"].invoke()
        assert len(received) == 1 and received[0].id.endswith("_what_if")
        chart = HistoryChart(root)
        chart.set_payload(interval_payload())
        assert chart.find_withtag("limit")
        assert chart.find_withtag("point")
        # Neither series may draw a line across the missing middle point.
        assert all(chart.type(item) != "line" for item in chart.find_withtag("point"))
        chart.set_payload(None)
        assert not chart.find_withtag("point")
    finally:
        root.destroy()
