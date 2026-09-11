"""Stage-4 ML tests for hybrid component provenance and mass blending."""

from __future__ import annotations

import pytest

from source.config import load_scenario, load_tag_dictionary
from source.contracts import ConstraintStatus, EstimateBasis, IntervalKind, MetricEstimate
from source.ml.blending import (
    GAS_CONTEXT_REASON,
    GAS_CONTEXT_SIGNAL_IDS,
    HybridComponentForecast,
    apply_hydrotreater_forecast,
    calculate_mass_blend,
    collect_gas_context,
    enumerate_two_component_recipes,
    rank_feasible_blends,
    sulfur_constraint_status,
)


def _forecast(*, value: float | None = 6.0, upper: float | None = 7.0):
    estimate = MetricEstimate(
        value=value,
        lower=None,
        upper=upper,
        unit="mg/kg",
        basis=EstimateBasis.FORECAST,
        interval_kind=IntervalKind.EMPIRICAL if upper is not None else IntervalKind.NONE,
        interval_level=0.95 if upper is not None else None,
        reference="model:test",
        assumptions=(),
    )
    return HybridComponentForecast(
        component_id="A",
        sulfur=estimate,
        source_state_id="ht-state-1",
        upstream_quality_reference="avt-state-1:straight-run-diesel",
        lag_min_minutes=0,
        lag_max_minutes=180,
        link_confirmed=False,
        evidence_ref="expert clarification: delay 0-3 h",
    )


def _components(forecast: HybridComponentForecast | None = None):
    base = load_scenario("config/scenarios/blend_risk.json").blend_components
    return apply_hydrotreater_forecast(base, forecast or _forecast())


def test_hydrotreater_forecast_replaces_only_corresponding_component() -> None:
    base = load_scenario("config/scenarios/blend_risk.json").blend_components

    result = apply_hydrotreater_forecast(base, _forecast(value=8.0, upper=9.0))

    assert base[0].sulfur.value == 6.0
    assert result[0].sulfur.value == 8.0
    assert result[0].source_state_id == "ht-state-1"
    assert result[1] == base[1]
    assert any("not confirmed" in item for item in result[0].sulfur.assumptions)


def test_mass_balance_uses_point_and_upper_sulfur_by_mass() -> None:
    result = calculate_mass_blend({"A": 0.9, "B": 0.1}, _components(), 100.0)

    assert result.sulfur.value == pytest.approx(8.4)
    assert result.sulfur.upper == pytest.approx(9.4)
    assert sulfur_constraint_status(result, 10.0) is ConstraintStatus.PASS
    assert result.checked_properties == ("sulfur", "component_stock")
    assert result.unassessed_properties == ("t95", "cetane_number")
    assert result.full_specification_status == "not_assessed"


def test_component_quality_changes_recipe_feasibility() -> None:
    recipe = {"A": 0.9, "B": 0.1}
    good = calculate_mass_blend(recipe, _components(_forecast(upper=7.0)), 100.0)
    degraded = calculate_mass_blend(recipe, _components(_forecast(upper=20.0)), 100.0)

    assert sulfur_constraint_status(good, 10.0) is ConstraintStatus.PASS
    assert sulfur_constraint_status(degraded, 10.0) is ConstraintStatus.FAIL


def test_hybrid_quality_changes_the_set_of_feasible_recipes() -> None:
    recipes = enumerate_two_component_recipes(("A", "B"), fraction_step=0.05)
    scenario = load_scenario("config/scenarios/blend_risk.json")
    good = rank_feasible_blends(
        recipes,
        _components(_forecast(upper=7.0)),
        100.0,
        scenario.current_blend_mass_fractions,
    )
    degraded = rank_feasible_blends(
        recipes,
        _components(_forecast(upper=9.0)),
        100.0,
        scenario.current_blend_mass_fractions,
    )

    assert good and degraded
    assert {option.recipe_id for option in good} != {option.recipe_id for option in degraded}
    assert all(option.result.full_specification_status == "not_assessed" for option in good)


def test_missing_upper_is_unknown_not_zero_or_pass() -> None:
    result = calculate_mass_blend({"A": 0.9, "B": 0.1}, _components(_forecast(upper=None)), 100.0)

    assert result.sulfur.value == pytest.approx(8.4)
    assert result.sulfur.upper is None
    assert sulfur_constraint_status(result, 10.0) is ConstraintStatus.UNKNOWN
    assert sulfur_constraint_status(result, 10.0, require_upper=False) is ConstraintStatus.PASS


def test_stock_and_recipe_validation_are_enforced() -> None:
    components = _components()
    result = calculate_mass_blend({"A": 0.5, "B": 0.5}, components, 100.0)

    assert result.stock_shortfalls_t == (("B", 20.0),)
    assert sulfur_constraint_status(result, 100.0) is ConstraintStatus.FAIL
    with pytest.raises(ValueError, match="sum to one"):
        calculate_mass_blend({"A": 0.8, "B": 0.1}, components, 100.0)
    with pytest.raises(ValueError, match="nonnegative"):
        calculate_mass_blend({"A": 1.1, "B": -0.1}, components, 100.0)


def test_gas_tags_are_context_only_not_action_controls() -> None:
    tags = load_tag_dictionary("config/tags.csv")

    context = collect_gas_context(tags)

    assert tuple(item.signal_id for item in context) == GAS_CONTEXT_SIGNAL_IDS
    assert all(not item.action_enabled for item in context)
    assert all(item.reason == GAS_CONTEXT_REASON for item in context)
    assert all(tags[item.signal_id].controllable is False for item in context)
    assert {item.unit for item in context} == {"unknown"}
