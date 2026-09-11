"""Stage-1 current-quality baseline and model-demo quality assessment."""

from __future__ import annotations

from datetime import timedelta

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


def _forecast_history_quality(
    state: ProcessState,
    features: pd.DataFrame,
    model: object,
    scenario: ScenarioConfig,
) -> AgentAssessment:
    """Return a compatible 60-minute point forecast without inventing an interval."""
    metadata = getattr(model, "metadata", None)
    capabilities = getattr(metadata, "capabilities", None)
    if metadata is None or not getattr(capabilities, "supports_forecast", False):
        return _unavailable(
            state,
            "MODEL_INCOMPATIBLE",
            HISTORY_SULFUR_SIGNAL_ID,
            "The selected artifact does not declare forecast capability.",
        )
    target_signal = str(getattr(metadata, "target_signal", ""))
    target_unit = str(getattr(metadata, "target_unit", ""))
    horizon_minutes = int(getattr(metadata, "horizon_minutes", 0))
    if target_signal != HISTORY_SULFUR_SIGNAL_ID or target_signal not in scenario.required_signals:
        return _unavailable(
            state,
            "MODEL_TARGET_MISMATCH",
            target_signal or None,
            "The model target does not match the history sulfur scenario.",
        )
    sulfur_constraints = [item for item in scenario.constraints if item.metric == "sulfur"]
    expected_unit = sulfur_constraints[0].unit if sulfur_constraints else target_unit
    if target_unit != expected_unit:
        return _unavailable(
            state,
            "UNIT_UNCONFIRMED",
            target_signal,
            f"Expected {expected_unit}, model provides {target_unit}.",
        )
    if horizon_minutes != 60:
        return _unavailable(
            state,
            "MODEL_HORIZON_MISMATCH",
            target_signal,
            "Stage 2 requires a 60-minute forecast horizon.",
        )
    if len(features) != 1:
        raise ValueError("history forecast requires exactly one feature row")
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise ValueError("model bundle must expose predict(features)")
    prediction = float(predict(features)[0])
    issues: tuple[Issue, ...] = ()
    status = AssessmentStatus.OK
    if scenario.require_upper_bound:
        status = AssessmentStatus.DEGRADED
        issues = (
            Issue(
                code="UNCERTAINTY_UNAVAILABLE",
                severity=Severity.BLOCKING,
                signal_id=target_signal,
                detail="Stage-2 point forecasts have no validated upper interval.",
                source_ref=f"model:{metadata.model_id}",
            ),
        )
    return AgentAssessment(
        agent=AssessmentAgent.QUALITY,
        state_id=state.state_id,
        candidate_id="hold",
        evaluated_for=state.as_of + timedelta(minutes=horizon_minutes),
        status=status,
        metrics={
            "sulfur": MetricEstimate(
                value=prediction,
                lower=None,
                upper=None,
                unit=target_unit,
                basis=EstimateBasis.FORECAST,
                interval_kind=IntervalKind.NONE,
                interval_level=None,
                reference=f"model:{metadata.model_id}",
                assumptions=(
                    f"{horizon_minutes}-minute point forecast",
                    f"training target source: {metadata.target_source}",
                    "supports_forecast does not imply supports_actions",
                ),
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
    """Assess measured quality, a compatible forecast, or an explicit demo formula."""
    if state.mode is not scenario.mode:
        raise ValueError("state and scenario modes must match")
    if state.mode is OperationMode.HISTORY:
        if model is not None:
            return _forecast_history_quality(state, features, model, scenario)
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
