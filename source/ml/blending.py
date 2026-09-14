"""Mass-balance and provenance helpers for the stage-4 hybrid blend scenario."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from source.contracts import (
    BlendComponent,
    ConstraintStatus,
    EstimateBasis,
    IntervalKind,
    MetricEstimate,
    Stage,
    TagMeta,
    Unit,
)

FRACTION_TOLERANCE = 1e-9
GAS_CONTEXT_SIGNAL_IDS = ("ht:F9", "ht:F22", "ht:Q21")
GAS_CONTEXT_REASON = (
    "context_only: gas signals are observed process context, not enabled action controls"
)


@dataclass(frozen=True)
class HybridComponentForecast:
    """Hydrotreater forecast and the explicit upstream-link assumptions behind it."""

    component_id: str
    sulfur: MetricEstimate
    source_state_id: str
    upstream_quality_reference: str
    lag_min_minutes: int
    lag_max_minutes: int
    link_confirmed: bool
    evidence_ref: str

    def __post_init__(self) -> None:
        if self.sulfur.unit != Unit.MG_KG.value:
            raise ValueError("hydrotreater sulfur forecast must use mg/kg")
        if self.sulfur.basis is not EstimateBasis.FORECAST:
            raise ValueError("hybrid component quality must be a forecast")
        if not self.component_id or not self.source_state_id:
            raise ValueError("component_id and source_state_id are required")
        if not self.upstream_quality_reference or not self.evidence_ref:
            raise ValueError("upstream and evidence references are required")
        if self.lag_min_minutes < 0 or self.lag_max_minutes < self.lag_min_minutes:
            raise ValueError("transport lag range is invalid")


@dataclass(frozen=True)
class BlendResult:
    """Partial blend result: core demo properties are assessed, not the full passport."""

    sulfur: MetricEstimate
    t95: MetricEstimate
    cetane_number: MetricEstimate
    stock_shortfalls_t: tuple[tuple[str, float], ...]
    checked_properties: tuple[str, ...]
    unassessed_properties: tuple[str, ...]
    full_specification_status: Literal["not_assessed"]
    assumptions: tuple[str, ...]

    @property
    def stock_feasible(self) -> bool:
        return not self.stock_shortfalls_t


@dataclass(frozen=True)
class BlendOption:
    """One feasible sulfur-only recipe with transparent ranking proxies."""

    recipe_id: str
    mass_fractions: dict[str, float]
    result: BlendResult
    risk_index: float
    throughput: float
    cost_proxy: float
    change_size: float

    @property
    def rank_key(self) -> tuple[float, float, float, float, str]:
        return (
            self.risk_index,
            -self.throughput,
            self.cost_proxy,
            self.change_size,
            self.recipe_id,
        )


@dataclass(frozen=True)
class GasContextSignal:
    """Observed gas signal metadata that is deliberately not an action control."""

    signal_id: str
    meaning: str
    unit: str
    evidence_ref: str
    action_enabled: Literal[False]
    reason: str


def collect_gas_context(
    tag_dictionary: Mapping[str, TagMeta],
    signal_ids: tuple[str, ...] = GAS_CONTEXT_SIGNAL_IDS,
) -> tuple[GasContextSignal, ...]:
    """Return known hydrotreatment gas tags as context-only process signals."""
    by_signal_id = {tag.signal_id: tag for tag in tag_dictionary.values()}
    result: list[GasContextSignal] = []
    for signal_id in signal_ids:
        tag = by_signal_id.get(signal_id)
        if tag is None:
            continue
        if tag.stage is not Stage.HYDROTREATMENT:
            raise ValueError(f"gas context signal must belong to hydrotreatment: {signal_id}")
        result.append(
            GasContextSignal(
                signal_id=tag.signal_id,
                meaning=tag.meaning,
                unit=tag.canonical_unit,
                evidence_ref=tag.evidence_ref,
                action_enabled=False,
                reason=GAS_CONTEXT_REASON,
            )
        )
    return tuple(result)


def apply_hydrotreater_forecast(
    components: tuple[BlendComponent, ...], forecast: HybridComponentForecast
) -> tuple[BlendComponent, ...]:
    """Replace one model component with the corresponding hydrotreater forecast."""
    matching = [component for component in components if component.id == forecast.component_id]
    if len(matching) != 1:
        raise ValueError("forecast component must match exactly one blend component")

    confirmation = "confirmed" if forecast.link_confirmed else "model assumption, not confirmed"
    provenance = (
        f"Incoming quality context: {forecast.upstream_quality_reference}.",
        (
            f"Transport lag {forecast.lag_min_minutes}-{forecast.lag_max_minutes} min "
            f"is {confirmation}."
        ),
        f"Flow-link evidence: {forecast.evidence_ref}.",
    )
    sulfur = forecast.sulfur.model_copy(
        update={"assumptions": tuple(forecast.sulfur.assumptions) + provenance}
    )
    return tuple(
        component.model_copy(update={"sulfur": sulfur, "source_state_id": forecast.source_state_id})
        if component.id == forecast.component_id
        else component
        for component in components
    )


def calculate_mass_blend(
    mass_fractions: dict[str, float],
    components: tuple[BlendComponent, ...],
    total_mass_t: float,
) -> BlendResult:
    """Calculate sulfur by mass and preserve missing values instead of imputing zero."""
    if total_mass_t <= 0:
        raise ValueError("total_mass_t must be positive")
    if not components:
        raise ValueError("at least one blend component is required")

    component_map = {component.id: component for component in components}
    if len(component_map) != len(components):
        raise ValueError("blend component ids must be unique")
    if set(mass_fractions) != set(component_map):
        raise ValueError("mass fractions must contain every component exactly once")
    if any(weight < 0 for weight in mass_fractions.values()):
        raise ValueError("mass fractions must be nonnegative")
    if abs(sum(mass_fractions.values()) - 1.0) > FRACTION_TOLERANCE:
        raise ValueError("mass fractions must sum to one")
    if {component.sulfur.unit for component in components} != {Unit.MG_KG.value}:
        raise ValueError("all component sulfur estimates must use mg/kg")

    positive_ids = [key for key, weight in mass_fractions.items() if weight > 0]

    def weighted(
        metric: Literal["sulfur", "t95", "cetane_number"], field: Literal["value", "upper"]
    ) -> float | None:
        total = 0.0
        for key in positive_ids:
            estimate = getattr(component_map[key], metric)
            value = None if estimate is None else getattr(estimate, field)
            if value is None:
                return None
            total += mass_fractions[key] * value
        return total

    def metric_estimate(metric: Literal["sulfur", "t95", "cetane_number"]) -> MetricEstimate:
        estimates = [getattr(component_map[key], metric) for key in positive_ids]
        present = [item for item in estimates if item is not None]
        if len(present) != len(estimates):
            return MetricEstimate(
                value=None,
                lower=None,
                upper=None,
                unit=Unit.UNKNOWN.value,
                basis=EstimateBasis.FORMULA,
                interval_kind=IntervalKind.NONE,
                interval_level=None,
                reference="DESIGN.md#9",
                assumptions=assumptions,
            )
        units = {item.unit for item in present}
        if len(units) != 1:
            raise ValueError(f"all component {metric} estimates must use one unit")
        upper = weighted(metric, "upper")
        return MetricEstimate(
            value=weighted(metric, "value"),
            lower=None,
            upper=upper,
            unit=present[0].unit,
            basis=EstimateBasis.FORMULA,
            interval_kind=IntervalKind.SCENARIO_BOUND if upper is not None else IntervalKind.NONE,
            interval_level=None,
            reference="DESIGN.md#9",
            assumptions=assumptions
            + tuple(
                assumption
                for component in components
                for estimate in (getattr(component, metric),)
                if estimate is not None
                for assumption in estimate.assumptions
            ),
        )

    shortfalls = tuple(
        (component.id, mass_fractions[component.id] * total_mass_t - component.available_mass_t)
        for component in components
        if mass_fractions[component.id] * total_mass_t
        > component.available_mass_t + FRACTION_TOLERANCE
    )
    assumptions = (
        "Sulfur, T95, cetane number and their upper bounds are mixed by mass fraction.",
        "The weighted upper bounds are conservative scenario bounds without joint coverage claims.",
        "Only the model-demo product properties and component stock are assessed.",
    )
    sulfur = metric_estimate("sulfur")
    t95 = metric_estimate("t95")
    cetane_number = metric_estimate("cetane_number")
    return BlendResult(
        sulfur=sulfur,
        t95=t95,
        cetane_number=cetane_number,
        stock_shortfalls_t=shortfalls,
        checked_properties=("sulfur", "t95", "cetane_number", "component_stock"),
        unassessed_properties=("full_product_passport",),
        full_specification_status="not_assessed",
        assumptions=assumptions,
    )


def sulfur_constraint_status(
    result: BlendResult, limit_mg_kg: float, *, require_upper: bool = True
) -> ConstraintStatus:
    """Classify a sulfur-only recipe without turning absent uncertainty into a pass."""
    if limit_mg_kg < 0:
        raise ValueError("sulfur limit must be nonnegative")
    if not result.stock_feasible:
        return ConstraintStatus.FAIL
    selected = result.sulfur.upper if require_upper else result.sulfur.value
    if selected is None:
        return ConstraintStatus.UNKNOWN
    return ConstraintStatus.PASS if selected <= limit_mg_kg else ConstraintStatus.FAIL


def enumerate_two_component_recipes(
    component_ids: tuple[str, str],
    *,
    fraction_step: float = 0.05,
    max_recipes: int = 125,
) -> tuple[dict[str, float], ...]:
    """Build a deterministic mutable recipe grid without silent truncation."""
    if len(set(component_ids)) != 2:
        raise ValueError("exactly two unique component ids are required")
    if not 0 < fraction_step <= 1:
        raise ValueError("fraction_step must be in (0, 1]")
    step_count = round(1.0 / fraction_step)
    if abs(step_count * fraction_step - 1.0) > FRACTION_TOLERANCE:
        raise ValueError("fraction_step must divide one exactly")
    recipe_count = step_count + 1
    if recipe_count > max_recipes:
        raise ValueError(
            f"recipe grid requires {recipe_count} recipes, but max_recipes={max_recipes}"
        )
    component_a, component_b = component_ids
    return tuple(
        {
            component_a: float(1.0 - index * fraction_step),
            component_b: float(index * fraction_step),
        }
        for index in range(recipe_count)
    )


def rank_feasible_blends(
    recipes: tuple[dict[str, float], ...],
    components: tuple[BlendComponent, ...],
    total_mass_t: float,
    current_fractions: dict[str, float],
    *,
    sulfur_upper_limit: float = 10.0,
) -> tuple[BlendOption, ...]:
    """Filter quality/stock failures and rank the remaining hybrid recipes."""
    component_map = {component.id: component for component in components}
    if set(current_fractions) != set(component_map):
        raise ValueError("current fractions must contain every component")
    options: list[BlendOption] = []
    for recipe in recipes:
        result = calculate_mass_blend(recipe, components, total_mass_t)
        if sulfur_constraint_status(result, sulfur_upper_limit) is not ConstraintStatus.PASS:
            continue
        recipe_id = "blend:" + ",".join(f"{key}={recipe[key]:.2f}" for key in sorted(recipe))
        options.append(
            BlendOption(
                recipe_id=recipe_id,
                mass_fractions=recipe,
                result=result,
                risk_index=sum(
                    recipe[key] * component_map[key].risk_index for key in component_map
                ),
                throughput=total_mass_t,
                cost_proxy=sum(
                    recipe[key] * component_map[key].cost_proxy_per_t for key in component_map
                ),
                change_size=sum(abs(recipe[key] - current_fractions[key]) for key in component_map),
            )
        )
    return tuple(sorted(options, key=lambda option: option.rank_key))


__all__ = [
    "BlendOption",
    "BlendResult",
    "GAS_CONTEXT_REASON",
    "GAS_CONTEXT_SIGNAL_IDS",
    "GasContextSignal",
    "HybridComponentForecast",
    "apply_hydrotreater_forecast",
    "calculate_mass_blend",
    "collect_gas_context",
    "enumerate_two_component_recipes",
    "rank_feasible_blends",
    "sulfur_constraint_status",
]
