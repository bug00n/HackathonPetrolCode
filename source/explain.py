"""Human-readable Russian explanations for stage-1 decisions."""

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
            f"Надёжной рекомендации нет: {reasons}. Сценарий {scenario.id} требует ручной проверки."
        )
    if status is RecommendationStatus.HOLD:
        return (
            f"Рекомендуется сохранить режим: {_sulfur_text(selected)} проходит жёсткие ограничения."
        )
    action = selected.candidate.id if selected else "unknown"
    return (
        f"Рекомендуется вариант {action}: текущий режим не проходит ограничения "
        f"({_sulfur_text(baseline)}), выбранный вариант проходит ({_sulfur_text(selected)})."
    )


__all__ = ["build_explanation"]
