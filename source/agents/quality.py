"""Stage-1 current-quality baseline and model-demo quality assessment."""

from __future__ import annotations

import pandas as pd

from source.agents.effects import assess_blend_candidate
from source.contracts import (
    AgentAssessment,
    AssessmentAgent,
    AssessmentStatus,
    CandidateAction,
    CandidateKind,
    EstimateBasis,
    IntervalKind,
    Issue,
    MetricEstimate,
    OperationMode,
    ProcessState,
    ScenarioConfig,
    Severity,
    Validity,
)

HISTORY_SULFUR_SIGNAL_ID = "ht:2:Mg.Sulfur"


def _unavailable(
    state: ProcessState, code: str, signal_id: str | None, detail: str
) -> AgentAssessment:
    return AgentAssessment(
        agent=AssessmentAgent.QUALITY,
        state_id=state.state_id,
        candidate_id="hold",
        evaluated_for=state.as_of,
        status=AssessmentStatus.UNAVAILABLE,
        metrics={},
        issues=(
            Issue(
                code=code,
                severity=Severity.BLOCKING,
                signal_id=signal_id,
                detail=detail,
                source_ref=None,
            ),
        ),
    )


def _current_history_quality(state: ProcessState, scenario: ScenarioConfig) -> AgentAssessment:
    sulfur_constraints = [item for item in scenario.constraints if item.metric == "sulfur"]
    expected_unit = sulfur_constraints[0].unit if sulfur_constraints else None
    snapshot = state.signals.get(HISTORY_SULFUR_SIGNAL_ID)
    if (
        HISTORY_SULFUR_SIGNAL_ID not in scenario.required_signals
        or snapshot is None
        or snapshot.selected is None
        or snapshot.selected.validity is not Validity.VALID
        or snapshot.selected.value is None
    ):
        return _unavailable(
            state,
            "MISSING_REQUIRED_SIGNAL",
            HISTORY_SULFUR_SIGNAL_ID,
            "No valid available sulfur observation was found.",
        )

    observation = snapshot.selected
    if expected_unit is not None and observation.unit != expected_unit:
        return _unavailable(
            state,
            "UNIT_UNCONFIRMED",
            HISTORY_SULFUR_SIGNAL_ID,
            f"Expected {expected_unit}, received {observation.unit}.",
        )

    issues: tuple[Issue, ...] = ()
    status = AssessmentStatus.OK
    if not snapshot.fresh:
        status = AssessmentStatus.DEGRADED
        issues = (
            Issue(
                code="STALE_REQUIRED_SIGNAL",
                severity=Severity.BLOCKING,
                signal_id=HISTORY_SULFUR_SIGNAL_ID,
                detail="The latest valid quality observation is stale.",
                source_ref=observation.source_ref,
            ),
        )
    return AgentAssessment(
        agent=AssessmentAgent.QUALITY,
        state_id=state.state_id,
        candidate_id="hold",
        evaluated_for=state.as_of,
        status=status,
        metrics={
            "sulfur": MetricEstimate(
                value=observation.value,
                lower=None,
                upper=None,
                unit=observation.unit,
                basis=EstimateBasis.MEASURED,
                interval_kind=IntervalKind.NONE,
                interval_level=None,
                reference=observation.source_ref,
                assumptions=(),
            )
        },
        issues=issues,
    )


def predict_quality(
    state: ProcessState,
    features: pd.DataFrame,
    model: object | None,
    scenario: ScenarioConfig,
) -> AgentAssessment:
    """Assess current quality without pretending that a forecast model exists."""
    if state.mode is not scenario.mode:
        raise ValueError("state and scenario modes must match")
    if model is not None:
        raise ValueError("stage-1 quality supports model=None only")
    if state.mode is OperationMode.HISTORY:
        return _current_history_quality(state, scenario)
    if state.mode is OperationMode.HYBRID:
        return _unavailable(
            state,
            "ACTION_MODEL_UNAVAILABLE",
            None,
            "Hybrid quality is not implemented in stage 1.",
        )
    if state.mode is OperationMode.MODEL_DEMO:
        if not features.empty:
            raise ValueError("model-demo quality expects an empty feature frame")
        hold = CandidateAction(
            id="hold",
            kind=CandidateKind.HOLD,
            horizon_minutes=60,
            is_model_scenario=state.mode is OperationMode.MODEL_DEMO,
        )
        return assess_blend_candidate(state, hold, scenario)[0]
    return _unavailable(state, "MISSING_REQUIRED_SIGNAL", None, "No quality path is configured.")


__all__ = ["predict_quality"]
