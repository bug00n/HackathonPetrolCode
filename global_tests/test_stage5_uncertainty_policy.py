"""Acceptance tests for isolated Stage-5 ML helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from source.agents.quality import predict_quality
from source.config import load_scenario
from source.constraints import check_constraints
from source.contracts import CandidateAction, CandidateKind, ConstraintStatus, ProcessState
from source.ml.artifacts import (
    LastValueRegressor,
    ModelBundle,
    ModelMetadata,
    feature_schema_hash,
)
from source.ml.policy import (
    PolicyParameters,
    PolicyReplayCase,
    assess_change_policy,
    tune_policy,
)
from source.ml.uncertainty import (
    check_applicability,
    evaluate_robustness_cases,
    evaluate_upper_bounds,
    fit_stage5_uncertainty,
    fit_upper_calibrator,
)

UTC = timezone.utc


class PointUpperFixture:
    def __init__(self, point: float, upper: float, *, in_domain: bool = True) -> None:
        self.point = point
        self.upper = upper
        self.metadata = SimpleNamespace(
            model_id="stage5-fixture",
            target_signal="ht:2:Mg.Sulfur",
            target_source="pak",
            target_unit="mg/kg",
            horizon_minutes=60,
            capabilities=SimpleNamespace(
                supports_forecast=True,
                supports_actions=False,
                supports_uncertainty=True,
            ),
        )
        self.in_domain = in_domain

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray([self.point])

    def predict_upper(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray([self.upper])

    def check_applicability(self, features: pd.DataFrame) -> object:
        return SimpleNamespace(
            available=self.in_domain,
            reason_code=None if self.in_domain else "OUT_OF_DOMAIN",
        )


def _metadata() -> ModelMetadata:
    names = ("baseline", "x")
    return ModelMetadata(
        model_id="stage5-point",
        model_sha256="0" * 64,
        model_type="last_value",
        training_dataset_id="synthetic-stage5",
        git_commit="test",
        python_version="3.11.0",
        sklearn_version=__import__("sklearn").__version__,
        target_signal="ht:2:Mg.Sulfur",
        target_source="pak",
        target_unit="mg/kg",
        horizon_minutes=60,
        feature_names=names,
        baseline_feature="baseline",
        tag_dictionary_sha256="a" * 64,
        feature_schema_hash=feature_schema_hash(names),
        processing={},
        time_boundaries={
            "source_timezone": "UTC",
            "train_end_local": "2025-01-01",
            "validation_end_local": "2026-01-01",
        },
        seed=42,
        capabilities={"supports_forecast": True, "supports_actions": False},
        applicability={},
        reports=(),
    )


def _dataset() -> object:
    as_of = pd.date_range("2023-01-01", "2026-06-01", freq="7D", tz="UTC")
    phase = np.linspace(0.0, 12.0, len(as_of))
    baseline = 9.0 + np.sin(phase)
    frame = pd.DataFrame(
        {
            "as_of": as_of,
            "target_at": as_of + pd.Timedelta(minutes=60),
            "target_available_at": as_of + pd.Timedelta(minutes=60),
            "y": baseline + 0.3 * np.cos(phase),
            "baseline": baseline,
            "x": np.cos(phase),
        }
    )
    return SimpleNamespace(frame=frame, feature_names=("baseline", "x"))


def test_upper_model_selection_and_shift_use_separate_validation_halves() -> None:
    y = np.array([10.0, 10.0, 10.0, 10.0, 12.0, 13.0, 14.0, 15.0])
    calibration = fit_upper_calibrator(
        y,
        {
            "bad-first-half": [7.0, 7.0, 7.0, 7.0, 20.0, 20.0, 20.0, 20.0],
            "selected": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        },
    )

    assert calibration.model_id == "selected"
    assert calibration.selection_size == calibration.calibration_size == 4
    assert calibration.shift == pytest.approx(np.quantile([2.0, 3.0, 4.0, 5.0], 0.95))


def test_upper_evaluation_reports_held_out_coverage_and_width() -> None:
    metrics = evaluate_upper_bounds(
        y_true=[8.0, 10.0, 12.0, 14.0],
        point_prediction=[8.0, 9.0, 10.0, 11.0],
        upper_prediction=[9.0, 11.0, 11.0, 15.0],
    )

    assert metrics.coverage == pytest.approx(0.75)
    assert metrics.average_width == pytest.approx(2.0)
    assert metrics.count == 4


def test_applicability_never_turns_missing_or_regime_shift_into_pass() -> None:
    bounds = {"P8": (300.0, 380.0), "T11": (50.0, 100.0)}

    assert check_applicability({"P8": 340.0}, bounds).reason_code == "FEATURES_UNAVAILABLE"
    shifted = check_applicability({"P8": 430.0, "T11": 75.0}, bounds)
    assert not shifted.available
    assert shifted.reason_code == "OUT_OF_DOMAIN"
    assert shifted.violations == ("P8",)
    assert check_applicability({"P8": 340.0, "T11": 75.0}, bounds).available


def test_robustness_cases_keep_missing_and_invalid_uncertainty_unavailable() -> None:
    results = evaluate_robustness_cases(
        {
            "missing": None,
            "measurement_error": ([10.0], [9.0], [8.0]),
            "nominal": ([10.0], [9.0], [11.0]),
            "regime_shift": ([20.0], [9.0], [11.0]),
        }
    )

    by_name = {result.name: result for result in results}
    assert not by_name["missing"].available
    assert by_name["missing"].reason_code == "UNCERTAINTY_UNAVAILABLE"
    assert not by_name["measurement_error"].available
    assert by_name["regime_shift"].metrics is not None
    assert by_name["regime_shift"].metrics.coverage == 0.0


def test_policy_uses_first_differing_criterion_and_absolute_risk_threshold() -> None:
    now = datetime(2026, 9, 11, tzinfo=UTC)
    decision = assess_change_policy(
        {"risk_index": 0.40, "throughput": 100.0, "cost_proxy": 1.0},
        {"risk_index": 0.385, "throughput": 120.0, "cost_proxy": 0.5},
        ("risk_index", "throughput", "cost_proxy"),
        hold_feasible=True,
        now=now,
        last_recommended_at=None,
    )

    assert not decision.allow_change
    assert decision.reason_code == "NO_MATERIAL_IMPROVEMENT"
    assert decision.first_differing_criterion == "risk_index"


def test_policy_cooldown_only_suppresses_change_when_hold_is_feasible() -> None:
    now = datetime(2026, 9, 11, 12, tzinfo=UTC)
    kwargs = {
        "hold": {"risk_index": 0.4},
        "candidate": {"risk_index": 0.2},
        "active_criteria": ("risk_index",),
        "now": now,
        "last_recommended_at": now - timedelta(minutes=15),
    }

    feasible = assess_change_policy(**kwargs, hold_feasible=True)
    infeasible = assess_change_policy(**kwargs, hold_feasible=False)

    assert feasible.reason_code == "ACTION_COOLDOWN"
    assert not feasible.allow_change
    assert infeasible.reason_code == "HOLD_INFEASIBLE"
    assert infeasible.allow_change


def test_change_size_alone_is_not_a_material_improvement() -> None:
    decision = assess_change_policy(
        {"risk_index": 0.2, "change_size": 0.0},
        {"risk_index": 0.2, "change_size": 1.0},
        ("risk_index", "change_size"),
        hold_feasible=True,
        now=datetime(2026, 9, 11, tzinfo=UTC),
        last_recommended_at=None,
    )

    assert decision.reason_code == "NO_MATERIAL_IMPROVEMENT"
    assert not decision.allow_change


def test_policy_tuning_prioritizes_no_missed_required_changes() -> None:
    now = datetime(2026, 9, 11, tzinfo=UTC)
    cases = (
        PolicyReplayCase(
            now,
            {"risk_index": 0.4},
            {"risk_index": 0.35},
            ("risk_index",),
            True,
            True,
        ),
    )
    strict = PolicyParameters(risk=0.1)
    responsive = PolicyParameters(risk=0.02)

    assert tune_policy(cases, (strict, responsive)) == responsive


def test_policy_tuning_keeps_normal_period_stable_and_selects_cooldown() -> None:
    now = datetime(2026, 9, 11, 12, tzinfo=UTC)
    cases = (
        PolicyReplayCase(
            now,
            {"risk_index": 0.30},
            {"risk_index": 0.29},
            ("risk_index",),
            True,
            False,
        ),
        PolicyReplayCase(
            now,
            {"risk_index": 0.40},
            {"risk_index": 0.35},
            ("risk_index",),
            True,
            True,
        ),
        PolicyReplayCase(
            now,
            {"risk_index": 0.40},
            {"risk_index": 0.35},
            ("risk_index",),
            True,
            False,
            now - timedelta(minutes=15),
        ),
    )
    noisy = PolicyParameters(risk=0.0, cooldown_minutes=0)
    selected = PolicyParameters(risk=0.02, cooldown_minutes=60)
    inert = PolicyParameters(risk=0.1, cooldown_minutes=120)

    assert tune_policy(cases, (noisy, selected, inert)) == selected


def test_stage5_fit_uses_temporal_splits_and_reports_held_out_robustness() -> None:
    dataset = _dataset()
    assert hasattr(dataset, "frame")
    predictor = LastValueRegressor("baseline").fit(dataset.frame[["baseline", "x"]])
    point_model = ModelBundle(predictor, _metadata())

    fitted = fit_stage5_uncertainty(
        dataset,
        point_model,
        source_timezone="UTC",
    )

    assert fitted.calibration.interval_level == 0.95
    assert fitted.report["test_used_for_tuning"] is False
    assert 0.0 <= fitted.report["test_coverage"] <= 1.0
    assert fitted.report["test_average_width"] >= 0.0
    assert set(fitted.report["robustness"]) == {
        "measurement_error_plus_0_5_mg_kg",
        "regime_shift_last_quarter",
        "lims_delay",
    }


def test_stage5_helpers_are_available_from_lazy_ml_facade() -> None:
    import source.ml as ml

    assert ml.fit_upper_calibrator is fit_upper_calibrator
    assert ml.check_applicability is check_applicability
    assert ml.assess_change_policy is assess_change_policy
    assert ml.PolicyParameters is PolicyParameters


def test_quality_agent_uses_upper_bound_and_rejects_ood() -> None:
    state = ProcessState.model_validate_json(
        Path("global_tests/fixtures/contracts/process_state.json").read_text(encoding="utf-8")
    )
    scenario = load_scenario("config/scenarios/history.json")
    features = pd.DataFrame({"baseline": [9.0]})

    assessment = predict_quality(state, features, PointUpperFixture(9.2, 10.4), scenario)
    unavailable = predict_quality(
        state,
        features,
        PointUpperFixture(9.2, 10.4, in_domain=False),
        scenario,
    )

    assert assessment.status.value == "ok"
    assert assessment.metrics["sulfur"].value == pytest.approx(9.2)
    assert assessment.metrics["sulfur"].upper == pytest.approx(10.4)
    assert assessment.metrics["sulfur"].interval_level == 0.95
    checks = check_constraints(
        state,
        CandidateAction(id="hold", kind=CandidateKind.HOLD, horizon_minutes=60),
        (assessment,),
        scenario,
    )
    assert checks[0].actual == pytest.approx(10.4)
    assert checks[0].status is ConstraintStatus.FAIL
    assert unavailable.status.value == "unavailable"
    assert {issue.code for issue in unavailable.issues} == {"OUT_OF_DOMAIN"}
