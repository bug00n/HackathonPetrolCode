"""Find feasible synthetic two-component recipes without a coarse share grid."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from source.agents.optimizer import evaluate_candidates
from source.contracts import CandidateAction, CandidateEvaluation, CandidateKind, ScenarioConfig
from source.orchestrator import _model_demo_state


@dataclass(frozen=True)
class BlendOption:
    label: str
    fractions: dict[str, float]
    dose: float
    evaluation: CandidateEvaluation


def _bound(
    interval: tuple[float, float], slope: float, intercept: float, limit: float, *, upper: bool
) -> tuple[float, float]:
    """Intersect lo <= share_B <= hi with one linear quality inequality."""
    lo, hi = interval
    if abs(slope) < 1e-12:
        valid = intercept <= limit + 1e-10 if upper else intercept >= limit - 1e-10
        return (lo, hi) if valid else (1.0, 0.0)
    edge = (limit - intercept) / slope
    if (slope > 0) == upper:
        return lo, min(hi, edge)
    return max(lo, edge), hi


def _gain(scenario: ScenarioConfig, dose: float) -> float:
    additive = scenario.cetane_additive
    if additive is None or dose == 0:
        return 0.0
    points = additive.response_curve
    for left, right in zip(points, points[1:], strict=False):
        if left.mass_fraction <= dose <= right.mass_fraction:
            share = (dose - left.mass_fraction) / (right.mass_fraction - left.mass_fraction)
            return left.cetane_gain + share * (right.cetane_gain - left.cetane_gain)
    raise ValueError("Доза вне заданной кривой присадки")


def _shares(scenario: ScenarioConfig, dose: float, fixed: dict[str, float]) -> tuple[float, float]:
    a, b = scenario.blend_components
    mass = scenario.total_mass_t
    assert mass is not None and mass > 0
    diesel = 1.0 - dose
    lo = max(0.0, diesel - a.available_mass_t / mass)
    hi = min(diesel, b.available_mass_t / mass)
    if a.id in fixed:
        lo = max(lo, diesel - fixed[a.id])
        hi = min(hi, diesel - fixed[a.id])
    if b.id in fixed:
        lo = max(lo, fixed[b.id])
        hi = min(hi, fixed[b.id])
    for check in scenario.constraints:
        if check.metric not in {"sulfur", "t95", "cetane_number"}:
            raise ValueError(f"Автоподбор не поддерживает ограничение {check.metric}")
        left_metric = getattr(a, check.metric)
        right_metric = getattr(b, check.metric)
        if left_metric is None or right_metric is None:
            return 1.0, 0.0
        field = (
            "upper"
            if check.use_upper_estimate
            else "lower"
            if check.use_lower_estimate
            else "value"
        )
        left = getattr(left_metric, field)
        right = getattr(right_metric, field)
        if left is None or right is None:
            return 1.0, 0.0
        if check.metric == "sulfur":
            intercept, slope, correction = diesel * left, right - left, 0.0
        else:
            intercept, slope = left, (right - left) / diesel
            correction = _gain(scenario, dose) if check.metric == "cetane_number" else 0.0
        if check.upper is not None:
            lo, hi = _bound((lo, hi), slope, intercept + correction, check.upper, upper=True)
        if check.lower is not None:
            lo, hi = _bound((lo, hi), slope, intercept + correction, check.lower, upper=False)
    return lo, hi


def assist_blend(
    scenario: ScenarioConfig,
    desired: dict[str, float],
    desired_dose: float,
    locked: frozenset[str] = frozenset(),
) -> tuple[BlendOption, ...]:
    """Return three distinct useful recipes; desired and locked fractions use 0..1 units."""
    found = feasible_blends(scenario, desired, desired_dose, locked)
    if not found:
        return ()
    a, b = scenario.blend_components

    def distance(item: BlendOption) -> float:
        return sum(abs(item.fractions[key] - desired[key]) for key in desired) + abs(
            item.dose - desired_dose
        )

    def cost(item: BlendOption) -> float:
        for assessment in item.evaluation.assessments:
            metric = assessment.metrics.get("cost_proxy")
            if metric and metric.value is not None:
                return metric.value
        return float("inf")

    rankings = (
        ("Ближе к введённому", lambda item: (distance(item), cost(item))),
        ("Ниже стоимость", lambda item: (cost(item), distance(item))),
        ("Меньше присадки", lambda item: (item.dose, distance(item), cost(item))),
    )
    chosen: list[BlendOption] = []
    for label, key in rankings:
        best = min(found, key=lambda item: (*key(item), item.dose, item.fractions[b.id]))
        if not any(
            abs(best.dose - other.dose) < 1e-10
            and abs(best.fractions[b.id] - other.fractions[b.id]) < 1e-10
            for other in chosen
        ):
            chosen.append(BlendOption(label, best.fractions, best.dose, best.evaluation))
    return tuple(chosen)


def feasible_blends(
    scenario: ScenarioConfig,
    desired: dict[str, float],
    desired_dose: float,
    locked: frozenset[str] = frozenset(),
) -> tuple[BlendOption, ...]:
    """Evaluate interval boundaries and nearest points through the shared hard checks."""
    if scenario.mode.value != "model_demo" or len(scenario.blend_components) != 2:
        raise ValueError("Автоподбор доступен только для модельной смеси из двух компонентов")
    a, b = scenario.blend_components
    if set(desired) != {a.id, b.id}:
        raise ValueError("Укажите доли обоих компонентов")
    if any(value < 0 or value > 1 for value in (*desired.values(), desired_dose)):
        raise ValueError("Доли должны быть в пределах 0–100%")
    additive = scenario.cetane_additive
    doses: tuple[float, ...]
    if additive is None:
        doses = (0.0,)
    elif "dose" in locked:
        doses = (desired_dose,)
    else:
        steps = round(additive.max_mass_fraction / additive.fraction_step)
        doses = tuple(
            sorted({desired_dose, *(i * additive.fraction_step for i in range(steps + 1))})
        )
    state = _model_demo_state(datetime.now(UTC), scenario)
    assert scenario.total_mass_t is not None
    found: list[BlendOption] = []
    fixed = {name: desired[name] for name in (a.id, b.id) if name in locked}
    for dose in doses:
        if dose < 0 or dose > (additive.max_mass_fraction if additive else 0.0) + 1e-9:
            continue
        if additive and dose * scenario.total_mass_t > additive.available_mass_t + 1e-9:
            continue
        lo, hi = _shares(scenario, dose, fixed)
        if lo > hi + 1e-9:
            continue
        # Endpoints, closest shares, and the former grid's useful interior points.
        points = {
            max(lo, min(hi, value))
            for value in (
                lo,
                hi,
                desired[b.id],
                1 - dose - desired[a.id],
                (lo + hi) / 2,
            )
        }
        for share_b in sorted(points):
            fractions = {a.id: 1 - dose - share_b, b.id: share_b}
            candidate = CandidateAction(
                id=f"assisted:{a.id}={fractions[a.id]:.12g},{b.id}={share_b:.12g},dose={dose:.12g}",
                kind=CandidateKind.BLEND,
                blend_mass_fractions=fractions,
                additive_mass_fraction=dose,
                horizon_minutes=60,
                is_model_scenario=True,
            )
            evaluation = evaluate_candidates(state, (candidate,), scenario)[0]
            if evaluation.feasible:
                found.append(BlendOption("", fractions, dose, evaluation))
    return tuple(found)
