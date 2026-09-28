"""Stage-7 history artifact serving tests."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import pandas as pd
import pytest
import sklearn

from source.config import config_fingerprint, load_runtime_config
from source.contracts import DatasetManifest, Recommendation
from source.data.prepare import PreparedData, load_prepared_dataset, write_prepared_dataset
from source.main import main, run_history_command
from source.ml.artifacts import LastValueRegressor, feature_schema_hash, save_model

TARGET_SIGNAL = "ht:2:Mg.Sulfur"
TARGET_UNIT = "mg/kg"
FEATURE_NAME = f"{TARGET_SIGNAL}__pak_last"


def _prepared_data() -> PreparedData:
    manifest = DatasetManifest.model_validate_json(
        Path("global_tests/fixtures/data/manifest.json").read_text(encoding="utf-8")
    )
    config = load_runtime_config("config/runtime.toml")
    manifest = manifest.model_copy(
        update={
            "config_sha256": hashlib.sha256(config_fingerprint(config).encode()).hexdigest(),
            "tag_dictionary_sha256": hashlib.sha256(
                config.tag_dictionary_path.read_bytes()
            ).hexdigest(),
            "telemetry_rules_sha256": hashlib.sha256(
                config.telemetry_rules_path.read_bytes()
            ).hexdigest(),
        }
    )
    return PreparedData(
        telemetry=pd.DataFrame(
            {
                "timestamp": [
                    "2026-01-15T08:40:00Z",
                    "2026-01-15T08:50:00Z",
                    "2026-01-15T09:00:00Z",
                ],
                "ht:F1": [50.0, 51.0, 52.0],
            }
        ),
        quality=pd.DataFrame(
            {
                "observation_id": ["pak-before", "lims-delayed"],
                "signal_id": [TARGET_SIGNAL, TARGET_SIGNAL],
                "stage": ["ht", "ht"],
                "source": ["pak", "lims"],
                "measured_at": ["2026-01-15T08:00:00Z", "2026-01-15T08:00:00Z"],
                "available_at": ["2026-01-15T08:00:00Z", "2026-01-15T12:00:00Z"],
                "value": [8.8, 8.4],
                "unit": [TARGET_UNIT, TARGET_UNIT],
                "validity": ["valid", "valid"],
                "source_ref": ["fixture-pak", "fixture-lims"],
            }
        ),
        issues=pd.DataFrame(columns=["source_ref", "code", "detail"]),
        manifest=manifest,
        feature_order=("ht:F1", TARGET_SIGNAL),
    )


def _write_point_model(directory: Path, tag_dictionary_sha256: str) -> Path:
    model_id = "stage7-point"
    feature_definition = {
        "horizon_minutes": 60,
        "quality_history_source": "pak",
    }
    x = pd.DataFrame({FEATURE_NAME: [8.8, 9.1]})
    predictor = LastValueRegressor(FEATURE_NAME).fit(x)
    metadata = {
        "model_id": model_id,
        "schema_version": "1.0",
        "model_sha256": "0" * 64,
        "model_type": "last_value",
        "training_dataset_id": "0123456789ab",
        "git_commit": "test-commit",
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "target_signal": TARGET_SIGNAL,
        "target_source": "pak",
        "target_unit": TARGET_UNIT,
        "horizon_minutes": 60,
        "feature_names": (FEATURE_NAME,),
        "baseline_feature": FEATURE_NAME,
        "tag_dictionary_sha256": tag_dictionary_sha256,
        "feature_schema_hash": feature_schema_hash((FEATURE_NAME,), feature_definition),
        "processing": {"feature_definition": feature_definition},
        "time_boundaries": {
            "source_timezone": "UTC",
            "train_end_local": "2025-01-01",
            "validation_end_local": "2026-01-01",
        },
        "seed": 42,
        "capabilities": {
            "supports_forecast": True,
            "supports_actions": False,
            "supports_uncertainty": False,
        },
        "applicability": {"action_comparison": "forbidden"},
        "reports": ("metrics.json",),
    }
    artifact_dir = directory / model_id
    save_model(artifact_dir, predictor, metadata, {"selected_model": "last_value"})
    return artifact_dir


def test_reused_feature_sources_match_uncached_batches(tmp_path: Path) -> None:
    from source.ml.artifacts import load_model
    from source.ml.features import prepare_feature_batch, prepare_feature_sources

    data = _prepared_data()
    model = load_model(
        _write_point_model(tmp_path / "models", data.manifest.tag_dictionary_sha256),
        trusted=True,
    )
    queries = pd.date_range("2026-01-15T08:40:00Z", periods=4, freq="10min")
    sources = prepare_feature_sources(data, model)
    for chunk in (queries[:2], queries[2:]):
        expected = prepare_feature_batch(data, chunk, model).frame
        actual = prepare_feature_batch(data, chunk, model, sources=sources).frame
        pd.testing.assert_frame_equal(expected, actual)


def test_feature_recipe_mismatch_fails_before_forecast(tmp_path: Path) -> None:
    from source.ml.artifacts import load_model
    from source.ml.features import prepare_feature_sources

    data = _prepared_data()
    model = load_model(
        _write_point_model(tmp_path / "models", data.manifest.tag_dictionary_sha256),
        trusted=True,
    )
    metadata = model.metadata.model_dump(mode="python")
    metadata["processing"]["feature_definition"]["lags_minutes"] = [0, 15]
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="feature recipe"):
        prepare_feature_sources(data, SimpleNamespace(metadata=metadata))


def test_feature_recipe_accepts_telemetry_order_from_model() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from source.ml.features import prepare_feature_sources

    data = _prepared_data()
    telemetry = data.telemetry.copy()
    telemetry["ht:F2"] = telemetry["ht:F1"] + 1
    data = replace(data, telemetry=telemetry, feature_order=("ht:F2", "ht:F1", TARGET_SIGNAL))
    metadata = {
        "feature_names": ("ht:F1__lag_0m", "ht:F2__lag_0m", FEATURE_NAME),
        "target_signal": TARGET_SIGNAL,
        "target_source": "pak",
        "target_unit": TARGET_UNIT,
        "horizon_minutes": 60,
        "processing": {"feature_definition": {"telemetry_signals": ["ht:F1", "ht:F2"]}},
    }
    sources = prepare_feature_sources(data, SimpleNamespace(metadata=metadata))
    assert sources.feature_spec[1] == ("ht:F2", "ht:F1")


def test_runtime_output_path_does_not_invalidate_new_prepared_data(tmp_path: Path) -> None:
    from dataclasses import replace

    from source.main import PROJECT_ROOT, _validate_current_prepared_dataset

    config = load_runtime_config(Path("config/runtime.toml"))
    data = _prepared_data()
    manifest = data.manifest.model_copy(
        update={
            "preparation_version": "2",
            "config_sha256": hashlib.sha256(
                config_fingerprint(config, preparation_only=True).encode()
            ).hexdigest(),
        }
    )
    data = replace(data, manifest=manifest)
    path = write_prepared_dataset(data, tmp_path / "processed")
    data = load_prepared_dataset(path)
    changed_output = config.model_copy(update={"runs_dir": Path("other-runs")})
    _validate_current_prepared_dataset(data, changed_output, PROJECT_ROOT, path)
    changed_preparation = config.model_copy(update={"source_timezone": "UTC"})
    with pytest.raises(ValueError, match="config_sha256"):
        _validate_current_prepared_dataset(data, changed_preparation, PROJECT_ROOT, path)


def test_pinned_legacy_dataset_tolerates_output_path_change(tmp_path: Path) -> None:
    from source.main import _validate_current_prepared_dataset

    config = load_runtime_config(Path("config/runtime.toml"))
    for relative in ("config/tags.csv", "config/telemetry_rules.json"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(relative).read_bytes())
    path = write_prepared_dataset(_prepared_data(), tmp_path / "data/processed")
    data = load_prepared_dataset(path)
    release = {
        "prepared_dataset": f"data/processed/{path.name}",
        "legacy_config_sha256": data.manifest.config_sha256,
        "preparation_config_sha256": hashlib.sha256(
            config_fingerprint(config, preparation_only=True).encode()
        ).hexdigest(),
    }
    (tmp_path / "config/release_manifest.json").write_text(json.dumps(release), encoding="utf-8")
    changed_output = config.model_copy(update={"runs_dir": Path("other-runs")})
    _validate_current_prepared_dataset(data, changed_output, tmp_path, path)
    changed_preparation = config.model_copy(update={"source_timezone": "UTC"})
    with pytest.raises(ValueError, match="config_sha256"):
        _validate_current_prepared_dataset(data, changed_preparation, tmp_path, path)


def test_run_history_cli_uses_trusted_artifact_and_writes_journal(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = _prepared_data()
    dataset_path = write_prepared_dataset(data, tmp_path / "processed")
    model_dir = _write_point_model(tmp_path / "models", data.manifest.tag_dictionary_sha256)
    run_dir = tmp_path / "runs"

    exit_code = main(
        [
            "run-history",
            "--dataset",
            str(dataset_path),
            "--model",
            str(model_dir),
            "--trusted-model",
            "--as-of",
            "2026-01-15T09:00:00Z",
            "--run-dir",
            str(run_dir),
        ]
    )

    result = Recommendation.model_validate_json(capsys.readouterr().out)
    assert exit_code == 0
    assert result.mode.value == "history"
    assert result.model_id == model_dir.name
    assert result.status.value == "abstain"
    assert result.baseline is not None
    quality = result.baseline.assessments[0]
    assert quality.metrics["sulfur"].value == pytest.approx(8.8)
    assert quality.metrics["sulfur"].upper is None
    assert {issue.code for issue in quality.issues} == {"UNCERTAINTY_UNAVAILABLE"}

    journal_dir = run_dir / result.run_id
    for name in ("result.json", "input.json", "features.json", "trace.jsonl", "candidates.jsonl"):
        assert (journal_dir / name).is_file()
    features = json.loads((journal_dir / "features.json").read_text(encoding="utf-8"))
    assert features["features"][0][FEATURE_NAME] == pytest.approx(8.8)


def test_run_history_requires_explicit_trust_for_joblib_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = _prepared_data()
    dataset_path = write_prepared_dataset(data, tmp_path / "processed")
    model_dir = _write_point_model(tmp_path / "models", data.manifest.tag_dictionary_sha256)

    exit_code = main(
        [
            "run-history",
            "--dataset",
            str(dataset_path),
            "--model",
            str(model_dir),
            "--as-of",
            "2026-01-15T09:00:00Z",
        ]
    )

    assert exit_code == 1
    assert "trusted" in capsys.readouterr().err


def test_run_history_rejects_tag_dictionary_mismatch(tmp_path: Path) -> None:
    data = _prepared_data()
    dataset_path = write_prepared_dataset(data, tmp_path / "processed")
    model_dir = _write_point_model(tmp_path / "models", "f" * 64)

    with pytest.raises(ValueError, match="tag_dictionary_sha256"):
        run_history_command(
            dataset_path,
            model_dir,
            pd.Timestamp("2026-01-15T09:00:00Z").to_pydatetime(),
            trusted_model=True,
            run_dir=tmp_path / "runs",
        )
