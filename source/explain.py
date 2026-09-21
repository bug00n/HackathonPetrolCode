"""Human-readable Russian explanations for recommendation decisions."""

from __future__ import annotations

from source.contracts import CandidateEvaluation, RecommendationStatus, ScenarioConfig


def _quality_text(evaluation: CandidateEvaluation | None) -> str:
    if evaluation is None:
        return "паспорт качества не рассчитан"
    metrics = {}
    for assessment in evaluation.assessments:
        metrics.update(assessment.metrics)
    sulfur = metrics.get("sulfur")
    t95 = metrics.get("t95")
    cetane = metrics.get("cetane_number")
    return ", ".join(
        (
            "сера недоступна"
            if sulfur is None or sulfur.upper is None
            else f"сера upper {sulfur.upper:.3g} {sulfur.unit}",
            "T95 недоступен"
            if t95 is None or t95.upper is None
            else f"T95 upper {t95.upper:.3g} {t95.unit}",
            "цетановое число недоступно"
            if cetane is None or cetane.lower is None
            else f"цетановое lower {cetane.lower:.3g}",
        )
    )


def _checks_text(evaluation: CandidateEvaluation | None) -> str:
    if evaluation is None or not evaluation.checks:
        return "проверки недоступны"
    parts = []
    for check in evaluation.checks:
        bound = ""
        if check.lower is not None and check.upper is not None:
            bound = f" [{check.lower:g}; {check.upper:g}]"
        elif check.upper is not None:
            bound = f" <= {check.upper:g}"
        elif check.lower is not None:
            bound = f" >= {check.lower:g}"
        unit = f" {check.unit}" if check.unit else ""
        parts.append(
            f"{check.constraint_id}: {check.status.value}{bound}{unit}, "
            f"reason={check.reason_code}, source={check.evidence_ref}"
        )
    return "; ".join(parts)


def _improvement_text(
    baseline: CandidateEvaluation, selected: CandidateEvaluation, scenario: ScenarioConfig
) -> str:
    def value(evaluation: CandidateEvaluation, name: str) -> float | None:
        for assessment in evaluation.assessments:
            metric = assessment.metrics.get(name)
            if metric is not None:
                return metric.value
        return None

    improved = []
    for name in scenario.active_criteria:
        before, after = value(baseline, name), value(selected, name)
        if before is None or after is None:
            continue
        better = after > before if name == "throughput" else after < before
        if better:
            improved.append(f"{name}: {before:.3g} → {after:.3g}")
    return ", ".join(improved) or "модельные критерии выбора"


def build_explanation(
    status: RecommendationStatus,
    baseline: CandidateEvaluation | None,
    selected: CandidateEvaluation | None,
    reason_codes: tuple[str, ...],
    scenario: ScenarioConfig,
) -> str:
    """Build a short operator-facing explanation from saved calculations."""
    if status is RecommendationStatus.ABSTAIN:
        reasons = ", ".join(reason_codes)
        return (
            f"Надёжной рекомендации нет: {reasons}. Проверки baseline: "
            f"{_checks_text(baseline)}. Сценарий {scenario.id} требует ручной проверки."
        )
    if status is RecommendationStatus.HOLD:
        suffix = f" Причины выбора: {', '.join(reason_codes)}." if reason_codes else ""
        return (
            f"Рекомендуется сохранить режим: {_quality_text(selected)}. "
            f"Проверки: {_checks_text(selected)}.{suffix}"
        )
    action = selected.candidate.id if selected else "unknown"
    if baseline is not None and baseline.feasible and selected is not None:
        return (
            f"Рекомендуется вариант {action}: текущий режим проходит ограничения, "
            f"но выбранный существенно улучшает {_improvement_text(baseline, selected, scenario)}. "
            f"Качество выбранного варианта: {_quality_text(selected)}; "
            f"проверки: {_checks_text(selected)}."
        )
    return (
        f"Рекомендуется вариант {action}: текущий режим не проходит ограничения "
        f"({_quality_text(baseline)}; проверки: {_checks_text(baseline)}), "
        f"выбранный вариант проходит "
        f"({_quality_text(selected)}; проверки: {_checks_text(selected)})."
    )


__all__ = ["build_explanation"]
