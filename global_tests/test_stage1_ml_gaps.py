"""Regression tests for honest Stage-1 candidate evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from source.agents import optimizer
from source.config import load_runtime_config, load_scenario
from source.contracts import AssessmentStatus, Issue, ProcessState, Severity

FIXTURES = Path(__file__).parent / "fixtures"


def _demo() -> tuple[ProcessState, object]:
    payload = json.loads((FIXTURES / "model_demo/blend_normal.json").read_text(encoding="utf-8"))
    return ProcessState.model_validate(payload["state"]), load_scenario(
        "config/scenarios/blend_normal.json"
    )


def test_candidate_grid_fails_instead_of_silently_truncating() -> None:
    state, scenario = _demo()
    config = load_runtime_config("config/runtime.toml").model_copy(update={"max_candidates": 20})

    with pytest.raises(
        ValueError,
        match=r"requires 21 candidates including hold, but max_candidates=20",
    ):
        optimizer.generate_candidates(state, scenario, config)


def test_missing_active_metric_makes_candidate_infeasible(monkeypatch) -> None:
    state, scenario = _demo()
    hold = optimizer.generate_candidates(state, scenario)[0]
    assessments = list(optimizer.assess_blend_candidate(state, hold, scenario))
    metrics = dict(assessments[2].metrics)
    metrics.pop("cost_proxy")
    assessments[2] = assessments[2].model_copy(update={"metrics": metrics})
    monkeypatch.setattr(
        optimizer,
        "assess_blend_candidate",
        lambda *_: tuple(assessments),
    )

    evaluation = optimizer.evaluate_candidates(state, (hold,), scenario)[0]

    assert all(check.status.value == "pass" for check in evaluation.checks)
    assert evaluation.feasible is False
    assert evaluation.rank_key is None


@pytest.mark.parametrize("failure_kind", ["unavailable", "blocking_issue"])
def test_unavailable_or_blocking_assessment_makes_candidate_infeasible(
    monkeypatch, failure_kind: str
) -> None:
    state, scenario = _demo()
    hold = optimizer.generate_candidates(state, scenario)[0]
    assessments = list(optimizer.assess_blend_candidate(state, hold, scenario))
    if failure_kind == "unavailable":
        assessments[1] = assessments[1].model_copy(update={"status": AssessmentStatus.UNAVAILABLE})
    else:
        issue = Issue(
            code="TEST_BLOCKING_INPUT",
            severity=Severity.BLOCKING,
            signal_id=None,
            detail="Synthetic blocking issue.",
            source_ref="test",
        )
        assessments[1] = assessments[1].model_copy(
            update={"status": AssessmentStatus.DEGRADED, "issues": (issue,)}
        )
    monkeypatch.setattr(
        optimizer,
        "assess_blend_candidate",
        lambda *_: tuple(assessments),
    )

    evaluation = optimizer.evaluate_candidates(state, (hold,), scenario)[0]

    assert all(check.status.value == "pass" for check in evaluation.checks)
    assert evaluation.feasible is False
    assert evaluation.rank_key is None
