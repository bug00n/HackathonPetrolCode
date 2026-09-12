"""Recommendation-cycle orchestration with backend guardrails."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from source.agents.optimizer import evaluate_candidates, generate_candidates
from source.constraints import check_constraints
from source.contracts import (
    AgentAssessment,
    CandidateEvaluation,
    CandidateKind,
    ConstraintResult,
    DecisionContext,
    Issue,
    OperationMode,
    ProcessState,
    Recommendation,
    RecommendationStatus,
    RuntimeConfig,
    ScenarioConfig,
    Severity,
    SignalSnapshot,
)
from source.explain import build_explanation
from source.journal import write_run_journal
from source.ml.features import build_features
from source.ml.policy import PolicyParameters, assess_change_policy


def _model_demo_state(as_of: datetime, scenario: ScenarioConfig) -> ProcessState:
    issues: list[Issue] = []
    signals: dict[str, SignalSnapshot] = {}
    for component in scenario.blend_components:
        if component.sulfur.value is not None and (
            not scenario.require_upper_bound or component.sulfur.upper is not None
        ):
            continue
        signal_id = f"blend:{component.id}:sulfur"
        issue = Issue(
            code="MISSING_REQUIRED_SIGNAL",
            severity=Severity.BLOCKING,
            signal_id=signal_id,
            detail=f"component {component.id} has no complete sulfur estimate",
            source_ref=component.sulfur.reference,
        )
        issues.append(issue)
        signals[signal_id] = SignalSnapshot(
            selected=None,
            alternatives=(),
            age_seconds=None,
            fresh=False,
            issues=(issue,),
        )
    payload = {
        "schema_version": "1.0",
        "as_of": as_of.isoformat(),
        "dataset_id": "modeldemo001",
        "mode": OperationMode.MODEL_DEMO.value,
        "signals": {key: value.model_dump(mode="json") for key, value in sorted(signals.items())},
        "issues": [item.model_dump(mode="json") for item in issues],
    }
    state_id = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return ProcessState(state_id=state_id, **payload)


def _build_cycle_state(
    data: Any, as_of: datetime, scenario: ScenarioConfig, config: RuntimeConfig
) -> ProcessState:
    if scenario.mode is OperationMode.MODEL_DEMO:
        return _model_demo_state(as_of, scenario)
    from source.data.state import build_state

    return build_state(data, as_of, scenario, config)


def _blocking_reason_codes(evaluations: tuple[CandidateEvaluation, ...]) -> tuple[str, ...]:
    codes: list[str] = []
    for evaluation in evaluations:
        for assessment in evaluation.assessments:
            codes.extend(
                issue.code for issue in assessment.issues if issue.severity is Severity.BLOCKING
            )
        codes.extend(check.reason_code for check in evaluation.checks if check.reason_code != "OK")
    return tuple(dict.fromkeys(codes))


def _rejection_summary(evaluations: tuple[CandidateEvaluation, ...]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for evaluation in evaluations:
        if evaluation.feasible:
            continue
        for assessment in evaluation.assessments:
            for issue in assessment.issues:
                summary[issue.code] = summary.get(issue.code, 0) + 1
        for check in evaluation.checks:
            if check.reason_code != "OK":
                summary[check.reason_code] = summary.get(check.reason_code, 0) + 1
    return dict(sorted(summary.items()))


def _metric_value(
    assessments: tuple[AgentAssessment, ...],
    name: str,
) -> float | None:
    for assessment in assessments:
        metric = assessment.metrics.get(name)
        if metric is not None and metric.value is not None:
            return metric.value
    return None


def _select_result(
    evaluations: tuple[CandidateEvaluation, ...],
    scenario: ScenarioConfig,
    context: DecisionContext,
    as_of: datetime,
) -> tuple[RecommendationStatus, CandidateEvaluation | None, tuple[str, ...], str]:
    baseline = next(item for item in evaluations if item.candidate.kind is CandidateKind.HOLD)
    feasible = [
        item
        for item in evaluations
        if item.feasible and item.candidate.kind is not CandidateKind.HOLD
    ]
    if baseline.feasible:
        if not feasible:
            return RecommendationStatus.HOLD, baseline, (), "hold"
        best = min(feasible, key=_required_rank_key)
        hold_values = {
            criterion: _metric_value(baseline.assessments, criterion)
            for criterion in scenario.active_criteria
        }
        candidate_values = {
            criterion: _metric_value(best.assessments, criterion)
            for criterion in scenario.active_criteria
        }
        policy = assess_change_policy(
            hold_values,
            candidate_values,
            scenario.active_criteria,
            hold_feasible=True,
            now=as_of,
            last_recommended_at=context.last_recommended_at,
            parameters=PolicyParameters(
                risk=scenario.materiality_thresholds.get("risk_index", 0.0),
                throughput=scenario.materiality_thresholds.get("throughput", 0.0),
                cost_proxy=scenario.materiality_thresholds.get("cost_proxy", 0.0),
                cooldown_minutes=scenario.action_cooldown_minutes,
            ),
        )
        if not policy.allow_change:
            return (
                RecommendationStatus.HOLD,
                baseline,
                (policy.reason_code,),
                policy.reason_code.lower(),
            )
        return RecommendationStatus.RECOMMEND, best, ("MATERIAL_IMPROVEMENT",), "recommend"
    if not feasible:
        reasons = _blocking_reason_codes(evaluations) or ("NO_FEASIBLE_CANDIDATE",)
        return RecommendationStatus.ABSTAIN, None, reasons, "no_feasible_candidate"
    reasons = _blocking_reason_codes((baseline,)) or ("BASELINE_INFEASIBLE",)
    return (
        RecommendationStatus.RECOMMEND,
        min(feasible, key=_required_rank_key),
        reasons,
        "baseline_infeasible_recommend",
    )


def _required_rank_key(evaluation: CandidateEvaluation) -> tuple[float, float, float, float, str]:
    if evaluation.rank_key is None:
        raise ValueError("feasible candidate has no rank_key")
    return evaluation.rank_key


def _supports_actions(model: object | None) -> bool:
    if model is None:
        return False
    metadata = getattr(model, "metadata", None)
    capabilities = (
        metadata.get("capabilities")
        if isinstance(metadata, dict)
        else getattr(metadata, "capabilities", None)
    )
    if isinstance(capabilities, dict):
        return capabilities.get("supports_actions") is True
    return getattr(capabilities, "supports_actions", False) is True


def _recheck_selected(
    state: ProcessState,
    selected: CandidateEvaluation | None,
    scenario: ScenarioConfig,
) -> tuple[ConstraintResult, ...]:
    if selected is None:
        return ()
    rechecked = check_constraints(state, selected.candidate, selected.assessments, scenario)
    if rechecked != selected.checks:
        raise ValueError("selected constraint recheck mismatch")
    return rechecked


def run_cycle(
    data: Any,
    as_of: datetime,
    model: Any,
    scenario: ScenarioConfig,
    config: RuntimeConfig,
    context: DecisionContext,
    run_dir: Path,
) -> Recommendation:
    """Run one backend recommendation cycle and persist its journal."""
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    as_of = as_of.astimezone(UTC)
    if model is not None and scenario.mode is OperationMode.HISTORY and data is None:
        raise ValueError("prepared data is required when a forecast model is supplied")
    state = _build_cycle_state(data, as_of, scenario, config)
    features: pd.DataFrame | None = None
    if model is not None and scenario.mode is OperationMode.HISTORY:
        features = build_features(data, as_of, state, model)
    candidates = generate_candidates(state, scenario, config)
    evaluations = evaluate_candidates(state, candidates, scenario, features=features, model=model)
    baseline = next(item for item in evaluations if item.candidate.kind is CandidateKind.HOLD)
    status, selected, reason_codes, selection_reason = _select_result(
        evaluations, scenario, context, as_of
    )
    if scenario.mode is OperationMode.HISTORY and not _supports_actions(model):
        status = RecommendationStatus.ABSTAIN
        selected = None
        reason_codes = tuple(dict.fromkeys((*reason_codes, "ACTION_MODEL_UNAVAILABLE")))
        selection_reason = "action_model_unavailable"
    _recheck_selected(state, selected, scenario)
    alternatives = (
        ()
        if status is RecommendationStatus.ABSTAIN
        else tuple(
            item
            for item in sorted(
                (
                    candidate
                    for candidate in evaluations
                    if candidate.feasible and candidate != selected
                ),
                key=_required_rank_key,
            )[:3]
        )
    )
    result = Recommendation(
        run_id=str(uuid4()),
        state_id=state.state_id,
        as_of=state.as_of,
        scenario_id=scenario.id,
        mode=OperationMode.MODEL_DEMO if scenario.mode is OperationMode.MODEL_DEMO else state.mode,
        status=status,
        baseline=baseline,
        selected=selected,
        alternatives=alternatives,
        reason_codes=reason_codes,
        explanation=build_explanation(status, baseline, selected, reason_codes, scenario),
        assumptions=tuple(
            dict.fromkeys((*scenario.assumptions, "stage-3 deterministic backend cycle"))
        ),
        model_id=(
            (
                getattr(model.metadata, "model_id", None)
                if not isinstance(getattr(model, "metadata", None), dict)
                else model.metadata.get("model_id")
            )
            if model is not None
            else None
        ),
    )
    write_run_journal(
        run_dir,
        state,
        scenario,
        context,
        evaluations,
        result,
        selection_reason=selection_reason,
        rejection_summary=_rejection_summary(evaluations),
        features=features,
    )
    return result


__all__ = ["run_cycle"]
