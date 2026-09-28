"""Operator-facing explanations; technical codes remain in the saved result."""

from __future__ import annotations

from source.contracts import (
    CandidateEvaluation,
    ConstraintStatus,
    RecommendationStatus,
    ScenarioConfig,
)

REASON_TEXT = {
    "QUALITY_LIMIT": "ограничение качества не выполнено",
    "MISSING_REQUIRED_SIGNAL": "нет обязательного показателя качества",
    "UNASSESSED_REQUIRED_PROPERTY": "обязательное свойство смеси не оценено",
    "UNCERTAINTY_UNAVAILABLE": "нет консервативной оценки качества",
    "COMPONENT_STOCK": "недостаточный запас компонента",
    "ADDITIVE_LIMIT": "превышена допустимая доля присадки",
    "ADDITIVE_STOCK": "недостаточный запас присадки",
    "NO_FEASIBLE_CANDIDATE": "нет смеси, проходящей все проверки",
    "BASELINE_INFEASIBLE": "текущая рецептура не проходит ограничения",
    "NO_MATERIAL_IMPROVEMENT": "существенного улучшения не найдено",
    "MATERIAL_IMPROVEMENT": "найдено существенное модельное улучшение",
    "ACTION_COOLDOWN": "повторное изменение временно ограничено",
    "ACTION_MODEL_UNAVAILABLE": "невозможно оценить эффект реального действия",
    "ACTION_EFFECT_VALIDATION_FAILED": "модель эффекта действия не прошла проверку",
    "RELIABILITY_UNAVAILABLE": "надёжность оценки не подтверждена",
    "FORECAST_UNAVAILABLE": "нет применимого прогноза серы",
    "OUT_OF_DOMAIN": "условия вне проверенной области модели",
    "OOD": "условия вне проверенной области модели",
    "UNIT_MISMATCH": "единицы измерения не совпадают",
    "STALE_REQUIRED_SIGNAL": "обязательные данные устарели",
    "APPLICABILITY_UNAVAILABLE": "применимость модели не подтверждена",
    "RISK_MODEL_UNAVAILABLE": "оценка риска недоступна",
    "CRITERION_UNAVAILABLE": "критерий выбора не рассчитан",
    "QUALITY_LIMIT_VIOLATION": "превышен предел качества",
    "CURRENT_SULFUR_LIMIT": "текущая сера превышает предел",
    "EXCEEDANCE_PROBABILITY_THRESHOLD": "вероятность превышения выше порога",
    "ACTION_DELTA_OUT_OF_OBSERVED_RANGE": "изменение вне наблюдавшегося диапазона",
    "SULFUR_UPPER_LIMIT_60M": "верхняя оценка серы через час превышает предел",
}

METRIC_TEXT = {
    "sulfur": "сера",
    "t95": "T95",
    "cetane_number": "цетановое число",
    "risk_index": "индекс риска",
    "cost_proxy": "стоимостной индекс",
    "throughput": "производительность",
}


def reason_text(code: str) -> str:
    """Translate a backend reason without exposing an untranslated code in the UI."""
    return REASON_TEXT.get(code, "требуется дополнительная техническая проверка")


def reasons_text(codes: tuple[str, ...] | list[str]) -> str:
    """Collapse repeated reasons while keeping their order."""
    return "; ".join(dict.fromkeys(reason_text(code) for code in codes)) or "не указана"


def _number(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", ",")


def _metric(evaluation: CandidateEvaluation | None, name: str) -> float | None:
    if evaluation is not None:
        for assessment in evaluation.assessments:
            estimate = assessment.metrics.get(name)
            if estimate is not None:
                return estimate.value
    return None


def _quality_text(evaluation: CandidateEvaluation | None) -> str:
    if evaluation is None:
        return "оценки качества недоступны"
    metrics = {
        name: estimate for item in evaluation.assessments for name, estimate in item.metrics.items()
    }
    values = []
    for name, bound, unit in (
        ("sulfur", "upper", "мг/кг"),
        ("t95", "upper", "°C"),
        ("cetane_number", "lower", "единиц"),
    ):
        estimate = metrics.get(name)
        value = getattr(estimate, bound, None)
        if value is not None:
            edge = "верхняя" if bound == "upper" else "нижняя"
            values.append(f"{METRIC_TEXT[name]} ({edge} оценка) — {_number(value)} {unit}")
    return "; ".join(values) if values else "оценки качества недоступны"


def _constraint_name(constraint_id: str) -> str:
    if constraint_id.startswith("component_stock:"):
        return f"запас компонента {constraint_id.rsplit(':', 1)[-1]}"
    return {
        "blend_sulfur": "сера",
        "blend_t95": "T95",
        "blend_cetane": "цетановое число",
        "additive_fraction": "доля присадки",
        "additive_stock": "запас присадки",
    }.get(constraint_id, "обязательное ограничение")


def _failed_checks_text(evaluation: CandidateEvaluation | None) -> str:
    if evaluation is None:
        return "обязательные проверки недоступны"
    failed = []
    for check in evaluation.checks:
        if check.status is ConstraintStatus.PASS:
            continue
        name = _constraint_name(check.constraint_id)
        if check.status is ConstraintStatus.UNKNOWN or check.actual is None:
            failed.append(f"{name}: нет надёжной оценки")
            continue
        unit = {
            "mg/kg": "мг/кг",
            "degC": "°C",
            "cetane_number": "ед.",
            "t": "т",
            "1": "доля",
        }.get(check.unit or "", check.unit or "")
        limit = (
            f"предел ≤ {_number(check.upper)} {unit}"
            if check.upper is not None
            else f"предел ≥ {_number(check.lower)} {unit}"
            if check.lower is not None
            else "предел недоступен"
        )
        failed.append(f"{name}: {_number(check.actual)} {unit}, {limit}".replace("  ", " "))
    return "; ".join(failed) if failed else "нарушений среди рассчитанных проверок нет"


def _comparison_text(
    baseline: CandidateEvaluation, selected: CandidateEvaluation, scenario: ScenarioConfig
) -> str:
    changes = []
    for name in scenario.active_criteria:
        if name not in METRIC_TEXT:
            continue
        before, after = _metric(baseline, name), _metric(selected, name)
        if before is not None and after is not None and abs(before - after) > 1e-9:
            changes.append(f"{METRIC_TEXT[name]} {_number(before)} → {_number(after)}")
    return "; ".join(changes)


def build_explanation(
    status: RecommendationStatus,
    baseline: CandidateEvaluation | None,
    selected: CandidateEvaluation | None,
    reason_codes: tuple[str, ...],
    scenario: ScenarioConfig,
) -> str:
    """Explain the decision in Russian; keep machine IDs in the journal fields."""
    if status is RecommendationStatus.ABSTAIN:
        reasons = "\n".join(
            f"• {text}" for text in dict.fromkeys(reason_text(code) for code in reason_codes)
        )
        return (
            "Надёжной рекомендации нет.\n\n"
            f"При проверке вариантов обнаружено:\n{reasons}\n"
            f"Проверка текущего варианта: {_failed_checks_text(baseline)}.\n"
            "Рецептура не выбирается автоматически."
        )
    if status is RecommendationStatus.HOLD:
        why = (
            reasons_text(reason_codes)
            if reason_codes
            else "текущая рецептура проходит проверки, а менять её не требуется"
        )
        return (
            "Текущую рецептуру менять не нужно.\n\n"
            f"Качество: {_quality_text(selected)}.\n"
            f"Почему: {why}."
        )
    comparison = (
        _comparison_text(baseline, selected, scenario)
        if baseline is not None and selected is not None
        else ""
    )
    if baseline is not None and baseline.feasible:
        start = "Выбран допустимый модельный вариант: текущий режим проходит ограничения."
        why = "Сравнение с текущим режимом: " + (comparison or "найдено существенное улучшение")
    else:
        start = "Текущая рецептура нарушает ограничения; найден допустимый модельный вариант."
        why = f"Что не прошло: {_failed_checks_text(baseline)}"
    return (
        f"{start}\n\n{why}.\n"
        f"Качество выбранной смеси: {_quality_text(selected)}.\n"
        "Выбранный вариант прошёл обязательные модельные проверки."
    )


__all__ = ["build_explanation", "reason_text", "reasons_text"]
