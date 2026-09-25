"""Stage-6 acceptance, freeze and journal-export tests."""

from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from source.acceptance import (
    JOURNAL_FILES,
    export_journals,
    load_episode_specs,
    run_acceptance_suite,
    sha256_file,
    verify_model_freeze,
)
from source.config import load_runtime_config, load_scenario
from source.contracts import (
    ConstraintBasis,
    ControlSpec,
    DecisionContext,
    Observation,
    OperationMode,
    ProcessState,
    SignalSnapshot,
    SourceKind,
    Stage,
    Validity,
)
from source.ml.action_effects import VerifiedActionEffectModel, combine_forecast_action_model
from source.ml.controls import ActionEffectEvidence, JointControlDomain
from source.orchestrator import run_cycle

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_episode_catalog_is_explicit_and_complete() -> None:
    episodes = load_episode_specs(PROJECT_ROOT / "config/demo_episodes.json")

    assert [episode.id for episode in episodes] == [
        "stable_blend",
        "sulfur_risk",
        "t95_risk",
        "cetane_risk",
        "missing_component_quality",
    ]
    assert [episode.expected_status for episode in episodes] == [
        "hold",
        "recommend",
        "recommend",
        "recommend",
        "abstain",
    ]


def test_acceptance_suite_reproduces_decisions_and_exports_full_journals(
    tmp_path: Path,
) -> None:
    output = tmp_path / "acceptance"
    report = run_acceptance_suite(
        PROJECT_ROOT,
        episodes_path=PROJECT_ROOT / "config/demo_episodes.json",
        output_dir=output,
    )

    assert report["episode_count"] == 5
    assert report["max_cycle_seconds"] < 5
    risk = next(item for item in report["episodes"] if item["episode_id"] == "sulfur_risk")
    assert risk["baseline_quality"]["sulfur_upper"] == pytest.approx(14.058)
    assert risk["selected_quality"] == pytest.approx(
        {"sulfur_upper": 9.306, "t95_upper": 354.0, "cetane_lower": 52.6}
    )
    assert risk["selected_additive_fraction"] == pytest.approx(0.01)
    assert risk["operator_recommendation"] is True
    with zipfile.ZipFile(output / "journals.zip") as archive:
        names = set(archive.namelist())
        assert "manifest.json" in names
        for filename in JOURNAL_FILES:
            assert f"sulfur_risk/{filename}" in names


@pytest.mark.parametrize("name", ("../outside", r"..\outside"))
def test_journal_export_rejects_path_components(name: str, tmp_path: Path) -> None:
    """A catalog id must not create a ZIP entry outside its own journal directory."""
    destination = tmp_path / "journals.zip"
    with pytest.raises(ValueError, match="single path components"):
        export_journals([(name, tmp_path / "source")], destination)
    assert not destination.exists()


def test_acceptance_decision_fingerprints_repeat(tmp_path: Path) -> None:
    reports = [
        run_acceptance_suite(
            PROJECT_ROOT,
            episodes_path=PROJECT_ROOT / "config/demo_episodes.json",
            output_dir=tmp_path / f"run-{index}",
        )
        for index in range(2)
    ]

    assert [item["decision_fingerprint"] for item in reports[0]["episodes"]] == [
        item["decision_fingerprint"] for item in reports[1]["episodes"]
    ]


def test_model_freeze_verifies_hashes_and_rejects_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts/models/frozen-test"
    artifact.mkdir(parents=True)
    model = artifact / "model.joblib"
    metadata = artifact / "metadata.json"
    metrics = artifact / "metrics.json"
    model.write_bytes(b"trusted-local-model")
    model_hash = sha256_file(model)
    metadata.write_text(
        json.dumps(
            {
                "model_id": "frozen-test",
                "model_sha256": model_hash,
                "training_dataset_id": "dataset00001",
                "git_commit": "a" * 40,
                "python_version": "3.12.3",
                "sklearn_version": "1.9.0",
                "capabilities": {"supports_actions": False},
            }
        ),
        encoding="utf-8",
    )
    metrics.write_text(json.dumps({"test_used_for_selection": False}), encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "training_dataset_id": "dataset00001",
                "training_git_commit": "a" * 40,
                "python_version": "3.12.3",
                "sklearn_version": "1.9.0",
                "artifacts": [
                    {
                        "model_id": "frozen-test",
                        "path": "artifacts/models/frozen-test",
                        "model_sha256": model_hash,
                        "metadata_sha256": sha256_file(metadata),
                        "metrics_sha256": sha256_file(metrics),
                    }
                ],
                "evaluation_reports": [],
            }
        ),
        encoding="utf-8",
    )

    assert verify_model_freeze(tmp_path, freeze)["verified_models"] == ["frozen-test"]
    model.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_model_freeze(tmp_path, freeze)


def test_model_freeze_accepts_only_fully_gated_action_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts/models/action-effect-test"
    artifact.mkdir(parents=True)
    model = artifact / "model.joblib"
    metadata_path = artifact / "metadata.json"
    metrics = artifact / "metrics.json"
    model.write_bytes(b"trusted-local-action-model")
    model_hash = sha256_file(model)
    digest = "f" * 64
    gates = {
        "minimum_episodes": True,
        "per_control_episodes": True,
        "mae_gain_10_percent": True,
        "validation_upper_coverage": True,
        "audit_upper_coverage": True,
        "sign_stability": True,
        "engineering_bounds": True,
        "shadow_replay": True,
        "technologist_pilot": True,
    }
    metadata = {
        "model_id": "action-effect-test",
        "model_sha256": model_hash,
        "training_dataset_id": "dataset00001",
        "git_commit": "a" * 40,
        "python_version": "3.12.3",
        "sklearn_version": "1.9.0",
        "artifact_kind": "action_effect",
        "supports_actions": True,
        "dataset_fingerprints": {
            "config_sha256": digest,
            "tag_dictionary_sha256": digest,
            "telemetry_rules_sha256": digest,
        },
        "controls": [
            {
                "signal_id": "ht:P8",
                "unit": "MPa",
                "lower": 0.1,
                "upper": 0.2,
                "max_step": 0.01,
                "step": 0.005,
                "evidence_ref": "engineering-bounds.md#P8",
                "basis": "confirmed",
                "enabled": True,
            },
            {
                "signal_id": "ht:F19",
                "unit": "t/h",
                "lower": 180.0,
                "upper": 230.0,
                "max_step": 5.0,
                "step": 2.5,
                "evidence_ref": "engineering-bounds.md#F19",
                "basis": "confirmed",
                "enabled": True,
            },
        ],
        "horizons_minutes": [60, 120, 180],
        "lag_evidence": {"selected_lag_minutes": 60},
        "gate_report": gates,
        "gate_report_hashes": {"validation": digest, "audit": digest},
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    metrics.write_text(json.dumps({"test_used_for_selection": False}), encoding="utf-8")
    freeze = tmp_path / "freeze.json"

    def write_freeze() -> None:
        freeze.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "training_dataset_id": "dataset00001",
                    "training_git_commit": "a" * 40,
                    "python_version": "3.12.3",
                    "sklearn_version": "1.9.0",
                    "config_sha256": digest,
                    "tag_dictionary_sha256": digest,
                    "telemetry_rules_sha256": digest,
                    "artifacts": [
                        {
                            "model_id": "action-effect-test",
                            "path": "artifacts/models/action-effect-test",
                            "model_sha256": model_hash,
                            "metadata_sha256": sha256_file(metadata_path),
                            "metrics_sha256": sha256_file(metrics),
                        }
                    ],
                    "evaluation_reports": [],
                }
            ),
            encoding="utf-8",
        )

    write_freeze()
    assert verify_model_freeze(tmp_path, freeze)["verified_models"] == ["action-effect-test"]

    metadata["gate_report"]["technologist_pilot"] = False
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    write_freeze()

    with pytest.raises(ValueError, match="action gate failed: technologist_pilot"):
        verify_model_freeze(tmp_path, freeze)


def test_failed_acceptance_does_not_publish_partial_directory(tmp_path: Path) -> None:
    bad_catalog = tmp_path / "episodes.json"
    payload = json.loads((PROJECT_ROOT / "config/demo_episodes.json").read_text(encoding="utf-8"))
    payload["episodes"][0]["expected_status"] = "abstain"
    bad_catalog.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "failed"

    with pytest.raises(ValueError, match="status changed"):
        run_acceptance_suite(PROJECT_ROOT, episodes_path=bad_catalog, output_dir=output)
    assert not output.exists()


def test_history_forecast_without_action_capability_abstains_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = datetime(2026, 1, 15, 9, tzinfo=UTC)
    observation = Observation(
        id="pak-sulfur",
        signal_id="ht:2:Mg.Sulfur",
        stage=Stage.HYDROTREATMENT,
        source=SourceKind.PAK,
        measured_at=as_of,
        available_at=as_of,
        value=8.0,
        unit="mg/kg",
        validity=Validity.VALID,
        source_ref="fixture",
    )
    state = ProcessState(
        state_id="history-state",
        as_of=as_of,
        dataset_id="dataset00001",
        mode=OperationMode.HISTORY,
        signals={
            "ht:2:Mg.Sulfur": SignalSnapshot(
                selected=observation,
                alternatives=(),
                age_seconds=0,
                fresh=True,
                issues=(),
            )
        },
        issues=(),
    )
    monkeypatch.setattr("source.orchestrator._build_cycle_state", lambda *_: state)

    result = run_cycle(
        data=None,
        as_of=as_of,
        model=None,
        scenario=load_scenario(PROJECT_ROOT / "config/scenarios/history.json"),
        config=load_runtime_config(PROJECT_ROOT / "config/runtime.toml"),
        context=DecisionContext(),
        run_dir=tmp_path,
    )

    assert result.status.value == "abstain"
    assert result.selected is None
    assert "ACTION_MODEL_UNAVAILABLE" in result.reason_codes


class _ForecastModel:
    def __init__(self) -> None:
        self.metadata = SimpleNamespace(
            model_id="forecast-fixture",
            capabilities=SimpleNamespace(
                supports_forecast=True,
                supports_actions=False,
                supports_uncertainty=True,
                supports_exceedance_probability=False,
                supports_multi_horizon=False,
            ),
            target_signal="ht:2:Mg.Sulfur",
            target_source="pak",
            target_unit="mg/kg",
            horizon_minutes=60,
            feature_names=("baseline",),
            processing={},
        )
        self.feature_names = ("baseline",)

    def predict(self, features: pd.DataFrame) -> list[float]:
        _ = features
        return [9.8]

    def predict_upper(self, features: pd.DataFrame) -> list[float]:
        _ = features
        return [11.0]

    def check_applicability(self, features: pd.DataFrame) -> SimpleNamespace:
        _ = features
        return SimpleNamespace(available=True, reason_code=None)


def _history_action_state(as_of: datetime) -> ProcessState:
    def observation(signal_id: str, value: float, unit: str) -> Observation:
        return Observation(
            id=f"obs:{signal_id}",
            signal_id=signal_id,
            stage=Stage.HYDROTREATMENT,
            source=SourceKind.TELEMETRY if signal_id != "ht:2:Mg.Sulfur" else SourceKind.PAK,
            measured_at=as_of,
            available_at=as_of,
            value=value,
            unit=unit,
            validity=Validity.VALID,
            source_ref="fixture",
        )

    observations = {
        "ht:2:Mg.Sulfur": observation("ht:2:Mg.Sulfur", 9.8, "mg/kg"),
        "ht:P8": observation("ht:P8", 0.15, "MPa"),
        "ht:F19": observation("ht:F19", 200.0, "t/h"),
    }
    return ProcessState(
        state_id="history-action-state",
        as_of=as_of,
        dataset_id="dataset00001",
        mode=OperationMode.HISTORY,
        signals={
            signal_id: SignalSnapshot(
                selected=item,
                alternatives=(),
                age_seconds=0,
                fresh=True,
                issues=(),
            )
            for signal_id, item in observations.items()
        },
        issues=(),
    )


def _verified_action_model() -> VerifiedActionEffectModel:
    controls = (
        ControlSpec(
            signal_id="ht:P8",
            unit="MPa",
            lower=0.1,
            upper=0.2,
            max_step=0.01,
            step=0.005,
            evidence_ref="engineering-bounds.md#P8",
            basis=ConstraintBasis.CONFIRMED,
            enabled=True,
        ),
        ControlSpec(
            signal_id="ht:F19",
            unit="t/h",
            lower=180.0,
            upper=230.0,
            max_step=5.0,
            step=2.5,
            evidence_ref="engineering-bounds.md#F19",
            basis=ConstraintBasis.CONFIRMED,
            enabled=True,
        ),
    )
    evidence = ActionEffectEvidence(
        "2025-01-01",
        "2026-01-01",
        240,
        60,
        1.0,
        0.8,
        per_control_episode_counts={"ht:P8": 120, "ht:F19": 120},
        conservative_coverage=0.96,
        sign_stable_folds=3,
        shadow_replay_passed=True,
        pilot_approved=True,
    )
    domain = JointControlDomain(
        signal_ids=("ht:P8", "ht:F19"),
        center=(0.15, 200.0),
        scale=(1.0, 100.0),
        inverse_correlation=((1.0, 0.0), (0.0, 1.0)),
        max_distance_squared=100.0,
    )
    return VerifiedActionEffectModel(
        model_id="action-fixture",
        controls=controls,
        joint_domain=domain,
        evidence=evidence,
        sulfur_coefficients={"ht:P8": -200.0},
        risk_coefficients={"ht:P8": -10.0},
        throughput_coefficients={},
        cost_coefficients={},
        evidence_ref="reports/action/fixture-gates.json",
    )


def test_history_with_verified_action_artifact_can_recommend_setpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    as_of = datetime(2026, 1, 15, 9, tzinfo=UTC)
    model = combine_forecast_action_model(_ForecastModel(), _verified_action_model())
    monkeypatch.setattr(
        "source.orchestrator._build_cycle_state",
        lambda *_: _history_action_state(as_of),
    )
    monkeypatch.setattr(
        "source.orchestrator.build_features",
        lambda *_: pd.DataFrame({"baseline": [9.8]}),
    )

    result = run_cycle(
        data=SimpleNamespace(manifest=SimpleNamespace(dataset_id="dataset00001")),
        as_of=as_of,
        model=model,
        scenario=load_scenario(PROJECT_ROOT / "config/scenarios/history.json"),
        config=load_runtime_config(PROJECT_ROOT / "config/runtime.toml"),
        context=DecisionContext(),
        run_dir=tmp_path,
    )

    assert result.status.value == "recommend"
    assert result.selected is not None
    assert result.selected.candidate.kind.value == "setpoints"
    assert "ACTION_MODEL_UNAVAILABLE" not in result.reason_codes
