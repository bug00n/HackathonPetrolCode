"""Acceptance tests for Stage-3 control and action-model semantics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from source.contracts import (
    CandidateAction,
    CandidateKind,
    ConstraintBasis,
    ControlSpec,
    OperationMode,
    ProcessState,
)
from source.ml.action_effects import (
    LinearActionEffectModel,
    evaluate_linear_action,
    rank_linear_actions,
)
from source.ml.controls import (
    CONTROL_IDS,
    ActionEffectEvidence,
    assess_action_capability,
    fit_joint_control_domain,
    generate_setpoint_candidates,
    summarize_observed_controls,
    unconfirmed_real_controls,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _training_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    latent = rng.normal(0.0, 1.0, 500)
    return pd.DataFrame(
        {
            "ht:P8": 10.0 + latent + rng.normal(0.0, 0.15, 500),
            "ht:T11": 100.0 + 4.0 * latent + rng.normal(0.0, 0.5, 500),
            "ht:F19": 30.0 - 2.0 * latent + rng.normal(0.0, 0.3, 500),
        }
    )


def _state(values: dict[str, float]) -> ProcessState:
    state = ProcessState.model_validate_json(
        (FIXTURES / "contracts/process_state.json").read_text(encoding="utf-8")
    )
    template = next(iter(state.signals.values()))
    assert template.selected is not None
    signals = {}
    for signal_id, value in values.items():
        selected = template.selected.model_copy(
            update={"id": f"obs:{signal_id}", "signal_id": signal_id, "value": value}
        )
        signals[signal_id] = template.model_copy(update={"selected": selected})
    return state.model_copy(update={"mode": OperationMode.MODEL_DEMO, "signals": signals})


def _controls() -> tuple[ControlSpec, ...]:
    ranges = {
        "ht:P8": (7.0, 13.0, 1.0),
        "ht:T11": (88.0, 112.0, 4.0),
        "ht:F19": (24.0, 36.0, 2.0),
    }
    return tuple(
        ControlSpec(
            signal_id=signal_id,
            unit="scenario_unit",
            lower=lower,
            upper=upper,
            max_step=2 * step,
            step=step,
            evidence_ref="synthetic scenario",
            basis=ConstraintBasis.CONFIRMED,
            enabled=True,
        )
        for signal_id, (lower, upper, step) in ranges.items()
    )


def _effect_model() -> LinearActionEffectModel:
    return LinearActionEffectModel(
        current_setpoints={"ht:P8": 10.0, "ht:T11": 100.0, "ht:F19": 30.0},
        baseline_sulfur=10.5,
        baseline_sulfur_upper=11.0,
        baseline_risk_index=0.2,
        baseline_throughput=100.0,
        baseline_cost_proxy=1.0,
        sulfur_coefficients={"ht:P8": -1.0, "ht:F19": -0.1},
        risk_coefficients={"ht:P8": 0.02},
        throughput_coefficients={"ht:T11": 1.0},
        cost_coefficients={"ht:P8": 0.05, "ht:F19": 0.01},
        evidence_ref="synthetic scenario coefficients",
    )


def test_real_controls_remain_disabled_when_units_and_limits_are_unknown() -> None:
    controls = unconfirmed_real_controls()
    report = assess_action_capability(controls, None)

    assert tuple(control.signal_id for control in controls) == CONTROL_IDS
    assert all(not control.enabled and control.unit == "unknown" for control in controls)
    assert report.supports_actions is False
    assert set(report.reason_codes) == {
        "CONTROL_LIMITS_UNCONFIRMED",
        "CONTROL_UNITS_UNKNOWN",
        "ACTION_EFFECT_VALIDATION_MISSING",
    }


def test_observed_statistics_are_explicitly_not_engineering_limits() -> None:
    stats = summarize_observed_controls(_training_frame())

    assert len(stats) == 3
    assert all(item.q05 < item.median < item.q95 for item in stats)
    assert all(
        item.as_dict()["interpretation"]
        == "observed_training_distribution_not_an_engineering_limit"
        for item in stats
    )


def test_joint_domain_rejects_individually_typical_but_implausible_combination() -> None:
    frame = _training_frame()
    domain = fit_joint_control_domain(frame, coverage=0.99)

    assert domain.contains({"ht:P8": 10.0, "ht:T11": 100.0, "ht:F19": 30.0})
    assert not domain.contains({"ht:P8": 12.0, "ht:T11": 92.0, "ht:F19": 34.0})


def test_action_capability_requires_temporal_gain_on_change_episodes() -> None:
    controls = _controls()
    weak = ActionEffectEvidence("2025-01-01", "2026-01-01", 30, 60, 1.0, 1.1)
    useful = ActionEffectEvidence("2025-01-01", "2026-01-01", 30, 60, 1.0, 0.8)

    assert not assess_action_capability(controls, weak).supports_actions
    assert assess_action_capability(controls, useful).supports_actions


def test_grid_is_deterministic_bounded_and_jointly_filtered() -> None:
    frame = _training_frame()
    domain = fit_joint_control_domain(frame, coverage=0.99)
    state = _state({"ht:P8": 10.0, "ht:T11": 100.0, "ht:F19": 30.0})

    first = generate_setpoint_candidates(
        state, _controls(), supports_actions=True, joint_domain=domain
    )
    second = generate_setpoint_candidates(
        state, _controls(), supports_actions=True, joint_domain=domain
    )

    assert first == second
    assert first[0].kind is CandidateKind.HOLD
    assert 1 < len(first) <= 126
    assert all(
        candidate.kind is CandidateKind.HOLD or domain.contains(candidate.setpoints)
        for candidate in first
    )


def test_no_action_capability_yields_hold_only() -> None:
    state = _state({"ht:P8": 10.0, "ht:T11": 100.0, "ht:F19": 30.0})

    candidates = generate_setpoint_candidates(
        state,
        _controls(),
        supports_actions=False,
        joint_domain=None,
    )

    assert [candidate.id for candidate in candidates] == ["hold"]


def test_linear_scenario_filters_before_documented_ranking() -> None:
    domain = fit_joint_control_domain(_training_frame(), coverage=0.99)
    state = _state({"ht:P8": 10.0, "ht:T11": 100.0, "ht:F19": 30.0})
    candidates = generate_setpoint_candidates(
        state, _controls(), supports_actions=True, joint_domain=domain
    )

    ranked = rank_linear_actions(candidates, _effect_model(), _controls(), domain)

    assert ranked
    assert all(outcome.feasible and outcome.sulfur_upper <= 10.0 for outcome in ranked)
    assert [outcome.rank_key for outcome in ranked] == sorted(
        outcome.rank_key for outcome in ranked if outcome.rank_key is not None
    )


def test_all_invalid_actions_produce_no_ranked_recommendation() -> None:
    domain = fit_joint_control_domain(_training_frame(), coverage=0.99)
    candidate = CandidateAction(
        id="setpoints:unsafe",
        kind=CandidateKind.SETPOINTS,
        setpoints={"ht:P8": 12.0, "ht:T11": 92.0, "ht:F19": 34.0},
        horizon_minutes=60,
        is_model_scenario=True,
    )

    outcome = evaluate_linear_action(candidate, _effect_model(), _controls(), domain)

    assert outcome.feasible is False
    assert "OUT_OF_DOMAIN" in outcome.reason_codes
    assert rank_linear_actions((candidate,), _effect_model(), _controls(), domain) == ()


def test_grid_fails_loudly_instead_of_truncating() -> None:
    domain = fit_joint_control_domain(_training_frame(), coverage=0.99)
    state = _state({"ht:P8": 10.0, "ht:T11": 100.0, "ht:F19": 30.0})

    with pytest.raises(ValueError, match="requires 124 action combinations"):
        generate_setpoint_candidates(
            state,
            _controls(),
            supports_actions=True,
            joint_domain=domain,
            max_action_combinations=100,
        )
