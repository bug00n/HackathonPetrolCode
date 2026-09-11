"""Human-readable Russian explanations for recommendation decisions."""

from __future__ import annotations

from source.contracts import CandidateEvaluation, RecommendationStatus, ScenarioConfig


def _sulfur_text(evaluation: CandidateEvaluation | None) -> str:
    if evaluation is None:
        return "сера не рассчитана"
    for assessment in evaluation.assessments:
        sulfur = assessment.metrics.get("sulfur")
        if sulfur is not None:
            if sulfur.value is None:
                return "сера не рассчитана"
            if sulfur.upper is None:
                return f"сера {sulfur.value:.3g} {sulfur.unit}, верхняя оценка недоступна"
            return f"сера {sulfur.value:.3g} {sulfur.unit}, верхняя оценка {sulfur.upper:.3g}"
    return "сера не рассчитана"


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
            f"Рекомендуется сохранить режим: {_sulfur_text(selected)}. "
            f"Проверки: {_checks_text(selected)}.{suffix}"
        )
    action = selected.candidate.id if selected else "unknown"
    return (
        f"Рекомендуется вариант {action}: текущий режим не проходит ограничения "
        f"({_sulfur_text(baseline)}; проверки: {_checks_text(baseline)}), "
        f"выбранный вариант проходит "
        f"({_sulfur_text(selected)}; проверки: {_checks_text(selected)})."
    )


__all__ = ["build_explanation"]
