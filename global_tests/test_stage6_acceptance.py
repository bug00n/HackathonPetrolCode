"""Stage-6 acceptance, freeze and journal-export tests."""

from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from source.acceptance import (
    JOURNAL_FILES,
    load_episode_specs,
    run_acceptance_suite,
    verify_model_freeze,
)
from source.config import load_runtime_config, load_scenario
from source.contracts import (
    DecisionContext,
    Observation,
    OperationMode,
    ProcessState,
    SignalSnapshot,
    SourceKind,
    Stage,
    Validity,
)
from source.ml.artifacts import sha256_file
from source.orchestrator import run_cycle

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_episode_catalog_is_explicit_and_complete() -> None:
    episodes = load_episode_specs(PROJECT_ROOT / "config/demo_episodes.json")

    assert [episode.id for episode in episodes] == [
        "stable_blend",
        "sulfur_risk",
        "missing_component_quality",
    ]
    assert all(episode.expected_status == "abstain" for episode in episodes)


def test_acceptance_suite_reproduces_decisions_and_exports_full_journals(
    tmp_path: Path,
) -> None:
    output = tmp_path / "acceptance"
    report = run_acceptance_suite(
        PROJECT_ROOT,
        episodes_path=PROJECT_ROOT / "config/demo_episodes.json",
        output_dir=output,
    )

    assert report["episode_count"] == 3
    assert report["max_cycle_seconds"] < 5
    risk = next(item for item in report["episodes"] if item["episode_id"] == "sulfur_risk")
    assert risk["baseline_upper_mg_kg"] == pytest.approx(14.2)
    assert risk["sulfur_only_counterfactual"]["upper_mg_kg"] == pytest.approx(9.4)
    assert risk["sulfur_only_counterfactual"]["operator_recommendation"] is False
    with zipfile.ZipFile(output / "journals.zip") as archive:
        names = set(archive.namelist())
        assert "manifest.json" in names
        for filename in JOURNAL_FILES:
            assert f"sulfur_risk/{filename}" in names


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


def test_failed_acceptance_does_not_publish_partial_directory(tmp_path: Path) -> None:
    bad_catalog = tmp_path / "episodes.json"
    payload = json.loads((PROJECT_ROOT / "config/demo_episodes.json").read_text(encoding="utf-8"))
    payload["episodes"][0]["expected_status"] = "hold"
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
