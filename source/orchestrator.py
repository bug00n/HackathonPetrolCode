"""Stage-1 orchestration for one recommendation cycle."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from source.agents.optimizer import evaluate_candidates, generate_candidates
from source.contracts import (
    CandidateEvaluation,
    CandidateKind,
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


def _select_result(
    evaluations: tuple[CandidateEvaluation, ...],
) -> tuple[RecommendationStatus, CandidateEvaluation | None, tuple[str, ...]]:
    baseline = next(item for item in evaluations if item.candidate.kind is CandidateKind.HOLD)
    if baseline.feasible:
        return RecommendationStatus.HOLD, baseline, ()
    feasible = [
        item
        for item in evaluations
        if item.feasible and item.candidate.kind is not CandidateKind.HOLD
    ]
    if not feasible:
        reasons = _blocking_reason_codes(evaluations) or ("NO_FEASIBLE_CANDIDATE",)
        return RecommendationStatus.ABSTAIN, None, reasons
    return (
        RecommendationStatus.RECOMMEND,
        min(feasible, key=_required_rank_key),
        ("QUALITY_LIMIT",),
    )


def _required_rank_key(evaluation: CandidateEvaluation) -> tuple[float, float, float, float, str]:
    if evaluation.rank_key is None:
        raise ValueError("feasible candidate has no rank_key")
    return evaluation.rank_key


def run_cycle(
    data: Any,
    as_of: datetime,
    model: Any,
    scenario: ScenarioConfig,
    config: RuntimeConfig,
    context: DecisionContext,
    run_dir: Path,
) -> Recommendation:
    """Run the stage-1 backend cycle and persist its journal."""
    _ = model
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    as_of = as_of.astimezone(UTC)
    state = _build_cycle_state(data, as_of, scenario, config)
    candidates = generate_candidates(state, scenario, config)
    evaluations = evaluate_candidates(state, candidates, scenario)
    baseline = next(item for item in evaluations if item.candidate.kind is CandidateKind.HOLD)
    status, selected, reason_codes = _select_result(evaluations)
    alternatives = tuple(
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
            dict.fromkeys((*scenario.assumptions, "stage-1 deterministic backend cycle"))
        ),
        model_id=None,
    )
    write_run_journal(run_dir, state, scenario, context, evaluations, result)
    return result


__all__ = ["run_cycle"]
