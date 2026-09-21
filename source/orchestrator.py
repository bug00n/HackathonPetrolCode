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
    AssessmentAgent,
    AssessmentStatus,
    CandidateAction,
    CandidateEvaluation,
    CandidateKind,
    ConstraintResult,
    ConstraintStatus,
    DecisionContext,
    EstimateBasis,
    IntervalKind,
    Issue,
    MetricEstimate,
    OperationMode,
    ProcessState,
    Recommendation,
    RecommendationStatus,
    RuntimeConfig,
    ScenarioConfig,
    Severity,
    SignalSnapshot,
    Unit,
)
from source.explain import build_explanation
from source.journal import write_run_journal
from source.ml.action_effects import ActionOutcome, VerifiedActionEffectModel
from source.ml.controls import generate_setpoint_candidates
from source.ml.features import build_features
from source.ml.policy import PolicyParameters, assess_change_policy


def _model_demo_state(as_of: datetime, scenario: ScenarioConfig) -> ProcessState:
    issues: list[Issue] = []
    signals: dict[str, SignalSnapshot] = {}
    for component in scenario.blend_components:
        properties = (
            ("sulfur", component.sulfur, "upper", "MISSING_REQUIRED_SIGNAL"),
            ("t95", component.t95, "upper", "UNASSESSED_REQUIRED_PROPERTY"),
            (
                "cetane_number",
                component.cetane_number,
                "lower",
                "UNASSESSED_REQUIRED_PROPERTY",
            ),
        )
        for name, estimate, bound, reason_code in properties:
            if (
                estimate is not None
                and estimate.value is not None
                and getattr(estimate, bound) is not None
            ):
                continue
            signal_id = f"blend:{component.id}:{name}"
            issue = Issue(
                code=reason_code,
                severity=Severity.BLOCKING,
                signal_id=signal_id,
                detail=f"component {component.id} has no complete {name} estimate",
                source_ref=f"scenario:{scenario.id}",
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


def _rank_key(
    assessments: tuple[AgentAssessment, ...],
    candidate_id: str,
    active_criteria: tuple[str, ...],
) -> tuple[float, float, float, float, str] | None:
    values = {
        name: _metric_value(assessments, name)
        for name in ("risk_index", "throughput", "cost_proxy", "change_size")
    }
    if any(values.get(name) is None for name in active_criteria):
        return None
    return (
        values["risk_index"] or 0.0,
        -(values["throughput"] or 0.0),
        values["cost_proxy"] or 0.0,
        values["change_size"] or 0.0,
        candidate_id,
    )


def _metric_estimate(
    value: float,
    unit: str,
    basis: EstimateBasis,
    reference: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
    interval_kind: IntervalKind = IntervalKind.NONE,
    interval_level: float | None = None,
    assumptions: tuple[str, ...] = (),
) -> MetricEstimate:
    return MetricEstimate(
        value=value,
        lower=lower,
        upper=upper,
        unit=unit,
        basis=basis,
        interval_kind=interval_kind,
        interval_level=interval_level,
        reference=reference,
        assumptions=assumptions,
    )


def _action_model(model: object | None) -> VerifiedActionEffectModel | None:
    if model is None:
        return None
    action_model = getattr(model, "action_model", None)
    return action_model if isinstance(action_model, VerifiedActionEffectModel) else None


def _scenario_with_action_controls(
    scenario: ScenarioConfig,
    model: object | None,
) -> ScenarioConfig:
    action_model = _action_model(model)
    if scenario.mode is not OperationMode.HISTORY or action_model is None:
        return scenario
    controls = tuple(action_model.controls)
    required = tuple(
        dict.fromkeys((*scenario.required_signals, *(item.signal_id for item in controls)))
    )
    assumptions = tuple(
        dict.fromkeys(
            (
                *scenario.assumptions,
                "Real setpoint controls are enabled only by a verified action artifact.",
            )
        )
    )
    return scenario.model_copy(
        update={"controls": controls, "required_signals": required, "assumptions": assumptions}
    )


def _sulfur_upper_limit(scenario: ScenarioConfig) -> float:
    for constraint in scenario.constraints:
        if constraint.metric == "sulfur" and constraint.upper is not None:
            return float(constraint.upper)
    return 10.0


def _action_assessment(
    state: ProcessState,
    candidate: CandidateAction,
    outcome: ActionOutcome,
    model_id: str,
) -> AgentAssessment:
    reason_codes = tuple(str(item) for item in getattr(outcome, "reason_codes", ()))
    issues = tuple(
        Issue(
            code=code,
            severity=Severity.BLOCKING,
            signal_id=None,
            detail="Action candidate failed a verified action-model guardrail.",
            source_ref=f"action_model:{model_id}",
        )
        for code in reason_codes
    )
    sulfur_upper = outcome.sulfur_upper
    return AgentAssessment(
        agent=AssessmentAgent.OPTIMIZER,
        state_id=state.state_id,
        candidate_id=candidate.id,
        evaluated_for=state.as_of,
        status=AssessmentStatus.DEGRADED if issues else AssessmentStatus.OK,
        metrics={
            "sulfur": _metric_estimate(
                float(outcome.sulfur),
                Unit.MG_KG.value,
                EstimateBasis.FORECAST,
                f"action_model:{model_id}",
                upper=None if sulfur_upper is None else float(sulfur_upper),
                interval_kind=(
                    IntervalKind.NONE if sulfur_upper is None else IntervalKind.EMPIRICAL
                ),
                interval_level=None if sulfur_upper is None else 0.95,
                assumptions=("verified action-effect model", "upper <= 10 mg/kg is required"),
            ),
            "risk_index": _metric_estimate(
                float(outcome.risk_index),
                Unit.RISK_INDEX.value,
                EstimateBasis.PROXY,
                f"action_model:{model_id}",
            ),
            "throughput": _metric_estimate(
                float(outcome.throughput),
                Unit.PROXY.value,
                EstimateBasis.PROXY,
                f"action_model:{model_id}",
            ),
            "cost_proxy": _metric_estimate(
                float(outcome.cost_proxy),
                Unit.PROXY.value,
                EstimateBasis.PROXY,
                f"action_model:{model_id}",
            ),
            "change_size": _metric_estimate(
                float(outcome.change_size),
                Unit.DIMENSIONLESS.value,
                EstimateBasis.FORMULA,
                f"action_model:{model_id}",
            ),
        },
        issues=issues,
    )


def _history_action_evaluations(
    state: ProcessState,
    candidates: tuple[CandidateAction, ...],
    scenario: ScenarioConfig,
    *,
    features: pd.DataFrame,
    model: object,
) -> tuple[CandidateEvaluation, ...]:
    from source.agents.quality import predict_quality

    action_model = _action_model(model)
    if action_model is None:
        raise ValueError("history action evaluation requires a verified action model")
    forecast = predict_quality(state, features, model, scenario)
    sulfur = forecast.metrics.get("sulfur")
    if sulfur is None or sulfur.value is None:
        hold = tuple(item for item in candidates if item.kind is CandidateKind.HOLD)
        return evaluate_candidates(state, hold, scenario, features=features, model=model)

    model_id = str(action_model.model_id)
    evaluations: list[CandidateEvaluation] = []
    for candidate in candidates:
        outcome = action_model.evaluate(
            state,
            candidate,
            baseline_sulfur=float(sulfur.value),
            baseline_sulfur_upper=sulfur.upper,
            sulfur_upper_limit=_sulfur_upper_limit(scenario),
        )
        assessments = (_action_assessment(state, candidate, outcome, model_id),)
        checks = check_constraints(state, candidate, assessments, scenario)
        rank_key = _rank_key(assessments, candidate.id, scenario.active_criteria)
        assessments_available = all(
            assessment.status is not AssessmentStatus.UNAVAILABLE
            and all(issue.severity is not Severity.BLOCKING for issue in assessment.issues)
            for assessment in assessments
        )
        feasible = (
            bool(outcome.feasible)
            and bool(checks)
            and all(item.status is ConstraintStatus.PASS for item in checks)
            and assessments_available
            and rank_key is not None
        )
        evaluations.append(
            CandidateEvaluation(
                candidate=candidate,
                assessments=assessments,
                checks=checks,
                feasible=feasible,
                rank_key=rank_key if feasible else None,
            )
        )
    return tuple(evaluations)


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
    if (
        scenario.mode is OperationMode.HISTORY
        and _supports_actions(model)
        and _action_model(model) is None
    ):
        raise ValueError("supports_actions=true requires a verified action model")
    effective_scenario = _scenario_with_action_controls(scenario, model)
    state = _build_cycle_state(data, as_of, effective_scenario, config)
    features: pd.DataFrame | None = None
    if model is not None and effective_scenario.mode is OperationMode.HISTORY:
        features = build_features(data, as_of, state, model)
    action_model = _action_model(model)
    if effective_scenario.mode is OperationMode.HISTORY and action_model is not None:
        candidates = generate_setpoint_candidates(
            state,
            tuple(action_model.controls),
            supports_actions=True,
            joint_domain=action_model.joint_domain,
            horizon_minutes=config.horizon_minutes,
            max_action_combinations=config.max_candidates,
        )
        if features is None:
            raise ValueError("history action evaluation requires forecast features")
        evaluations = _history_action_evaluations(
            state,
            candidates,
            effective_scenario,
            features=features,
            model=model,
        )
    else:
        candidates = generate_candidates(state, effective_scenario, config)
        evaluations = evaluate_candidates(
            state, candidates, effective_scenario, features=features, model=model
        )
    baseline = next(item for item in evaluations if item.candidate.kind is CandidateKind.HOLD)
    status, selected, reason_codes, selection_reason = _select_result(
        evaluations, effective_scenario, context, as_of
    )
    if effective_scenario.mode is OperationMode.HISTORY and not _supports_actions(model):
        status = RecommendationStatus.ABSTAIN
        selected = None
        reason_codes = tuple(dict.fromkeys((*reason_codes, "ACTION_MODEL_UNAVAILABLE")))
        selection_reason = "action_model_unavailable"
    _recheck_selected(state, selected, effective_scenario)
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
        explanation=build_explanation(status, baseline, selected, reason_codes, effective_scenario),
        assumptions=tuple(
            dict.fromkeys((*effective_scenario.assumptions, "stage-3 deterministic backend cycle"))
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
        effective_scenario,
        context,
        evaluations,
        result,
        selection_reason=selection_reason,
        rejection_summary=_rejection_summary(evaluations),
        features=features,
    )
    return result


__all__ = ["run_cycle"]
