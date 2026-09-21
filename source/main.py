"""Small CLI surface implemented for the current backend stages."""

from __future__ import annotations

import argparse
import hashlib
import json
<<<<<<< HEAD
import platform
=======
import os
>>>>>>> 1527107 (Extend UI and ML analysis materials)
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

import pandas as pd

from source.config import (
    config_fingerprint,
    load_runtime_config,
    load_scenario,
    load_tag_dictionary,
)
from source.contracts import (
    DecisionContext,
    ProcessState,
    Recommendation,
    RuntimeConfig,
    SourceKind,
)
from source.data import (
    PreparedData,
    build_state,
    load_prepared_dataset,
    prepare_dataset,
    write_prepared_dataset,
)

if TYPE_CHECKING:
    from source.ml.artifacts import ModelBundle
    from source.ml.features import SupervisedDataset

SplitName = Literal["train", "validation", "test"]
# Curated, dictionary-confirmed 24-2000 context.  P8/F19 remain historical
# action candidates; no signal is enabled as a real setpoint control.
TRAINING_TELEMETRY_SIGNALS = ("ht:P8", "ht:F19", "ht:T11")
MODEL_DEMO_SCENARIOS = (
    "blend_normal",
    "blend_risk",
    "blend_t95_risk",
    "blend_cetane_risk",
    "blend_missing",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DEMO_SCENARIOS: tuple[str, ...] = (
    "blend_normal",
    "blend_risk",
    "blend_t95_risk",
    "blend_cetane_risk",
    "blend_missing",
)
STAGE6_SCENARIOS = MODEL_DEMO_SCENARIOS
STAGE6_EXPECTED_STATUSES: dict[str, str] = {
    "blend_normal": "hold",
    "blend_risk": "recommend",
    "blend_t95_risk": "recommend",
    "blend_cetane_risk": "recommend",
    "blend_missing": "abstain",
}
STAGE6_JOURNAL_FILES: tuple[str, ...] = (
    "metadata.json",
    "input.json",
    "features.json",
    "trace.jsonl",
    "candidates.jsonl",
    "result.json",
)
STAGE6_LIMITATIONS: tuple[str, ...] = (
    "Stage 6 is an acceptance and demonstration layer, not a production action model.",
    "Real setpoint recommendations remain disabled until a validated action model exists.",
    "Model-demo counterfactuals are synthetic; history artifact serving remains forecast-only.",
)


def _write_json_report(path: str | Path, payload: dict[str, object], root: Path) -> Path:
    """Write a report atomically and refuse accidental overwrite."""
    destination = _resolve_path(path, root)
    if destination.exists():
        raise FileExistsError(f"report already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def validate_stage0(root: Path = PROJECT_ROOT) -> dict[str, int]:
    """Validate shared configs and serialized contract fixtures."""
    load_runtime_config(root / "config/runtime.toml")
    tags = load_tag_dictionary(root / "config/tags.csv")
    scenarios = [
        load_scenario(path) for path in sorted((root / "config/scenarios").glob("*.json"))
    ]
    contract_dir = root / "global_tests/fixtures/contracts"
    ProcessState.model_validate_json(
        (contract_dir / "process_state.json").read_text(encoding="utf-8")
    )
    Recommendation.model_validate_json(
        (contract_dir / "recommendation.json").read_text(encoding="utf-8")
    )
    model_demo_dir = root / "global_tests/fixtures/model_demo"
    demo_fixtures = list(sorted(model_demo_dir.glob("*.json")))
    for path in demo_fixtures:
        payload = json.loads(path.read_text(encoding="utf-8"))
        ProcessState.model_validate(payload["state"])
    return {
        "tags": len(tags),
        "scenarios": len(scenarios),
        "model_demo_fixtures": len(demo_fixtures),
    }


def run_model_demo(
    scenario_id: str,
    root: Path = PROJECT_ROOT,
    run_dir: str | Path | None = None,
) -> Recommendation:
    """Run one deterministic stage-1 model-demo scenario on fixture data."""
    from source.orchestrator import run_cycle

    config = load_runtime_config(root / "config/runtime.toml")
    scenario = load_scenario(root / f"config/scenarios/{scenario_id}.json")
    fixture = json.loads(
        (root / f"global_tests/fixtures/model_demo/{scenario_id}.json").read_text(encoding="utf-8")
    )
    state = ProcessState.model_validate(fixture["state"])
    return run_cycle(
        data=None,
        as_of=state.as_of,
        model=None,
        scenario=scenario,
        config=config,
        context=DecisionContext(),
        run_dir=_resolve_path(run_dir if run_dir is not None else config.runs_dir, root),
    )


def _resolve_path(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """Resolve CLI paths relative to the project root unless they are absolute."""
    value = Path(path)
    return value if value.is_absolute() else root / value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_current_prepared_dataset(
    data: PreparedData,
    config: RuntimeConfig,
    root: Path,
) -> None:
    """Fail early when a prepared dataset was built from older runtime inputs."""
    expected = {
        "config_sha256": hashlib.sha256(config_fingerprint(config).encode()).hexdigest(),
        "tag_dictionary_sha256": _sha256_file(_resolve_path(config.tag_dictionary_path, root)),
        "telemetry_rules_sha256": _sha256_file(
            _resolve_path(config.telemetry_rules_path, root)
        ),
    }
    actual = {
        "config_sha256": data.manifest.config_sha256,
        "tag_dictionary_sha256": data.manifest.tag_dictionary_sha256,
        "telemetry_rules_sha256": data.manifest.telemetry_rules_sha256,
    }
    mismatches = [
        f"{name}: manifest={actual[name] or '<missing>'}, current={expected[name]}"
        for name in expected
        if actual[name] != expected[name]
    ]
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(
            "prepared dataset is not current for this runtime/tags/rules: "
            f"{details}. Run `python -m source.main prepare` and use the new dataset_id."
        )


def _load_current_prepared_dataset(
    dataset: str | Path,
    config: RuntimeConfig,
    root: Path,
) -> PreparedData:
    data = load_prepared_dataset(_resolve_path(dataset, root))
    _validate_current_prepared_dataset(data, config, root)
    return data


def _parse_as_of(value: str) -> datetime:
    """Parse an ISO datetime and accept a trailing Z as UTC."""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO datetime: {value}") from exc


def prepare_command(
    materials: str | Path,
    config_path: str | Path,
    output: str | Path | None = None,
    root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Prepare original source files into a reproducible local dataset."""
    config = load_runtime_config(_resolve_path(config_path, root))
    data = prepare_dataset(_resolve_path(materials, root), config)
    output_root = _resolve_path(output if output is not None else config.data_dir, root)
    dataset_path = write_prepared_dataset(data, output_root)
    manifest = data.manifest
    return {
        "dataset_id": manifest.dataset_id,
        "dataset_path": dataset_path.as_posix(),
        "row_counts": manifest.row_counts,
        "time_ranges": {
            key: None
            if value is None
            else [item.isoformat().replace("+00:00", "Z") for item in value]
            for key, value in manifest.time_ranges.items()
        },
        "issues_count": manifest.row_counts.get("issues", 0),
    }


def build_state_command(
    dataset: str | Path,
    scenario: str | Path,
    as_of: datetime,
    config_path: str | Path = "config/runtime.toml",
    root: Path = PROJECT_ROOT,
) -> ProcessState:
    """Build a ProcessState from a prepared dataset and a checked-in scenario."""
    config = load_runtime_config(_resolve_path(config_path, root))
    scenario_path = Path(scenario)
    if not scenario_path.suffix:
        scenario_path = Path("config/scenarios") / f"{scenario}.json"
    scenario_config = load_scenario(_resolve_path(scenario_path, root))
    data = _load_current_prepared_dataset(dataset, config, root)
    return build_state(data, as_of, scenario_config, config)


def _git_revision(root: Path) -> str:
    """Return the exact clean revision used to create a model artifact."""
    revision = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise ValueError("training requires a clean worktree so the model can be frozen")
    return revision


<<<<<<< HEAD
def _source_or_output(
    value: SourceKind | str | Path | None,
    output: str | Path | None,
) -> tuple[SourceKind, str | Path | None]:
    """Keep the old positional train API while accepting the Stage-6 CLI shape."""
    if value is None:
        return SourceKind.PAK, output
    try:
        return SourceKind(value), output
    except ValueError:
        if output is None:
            return SourceKind.PAK, value
        raise


def _supervised_dataset(
    data: PreparedData,
    target_source: SourceKind,
    target_signal: str = "ht:2:Mg.Sulfur",
) -> SupervisedDataset:
=======
def _shadow_git_revision(root: Path, *, allow_dirty: bool) -> str:
    """Return a reproducible label for a shadow fit, including an explicit dirty marker."""
    if not allow_dirty:
        return _git_revision(root)
    revision = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=False,
    ).stdout
    diff = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "diff", "--binary", "HEAD", "--"],
        cwd=root,
        check=True,
        capture_output=True,
        text=False,
    ).stdout
    untracked = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={root.as_posix()}",
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=False,
    ).stdout
    digest = hashlib.sha256()
    digest.update(status)
    digest.update(diff)
    for encoded_path in sorted(path for path in untracked.split(b"\0") if path):
        digest.update(b"\0" + encoded_path + b"\0")
        candidate = root / os.fsdecode(encoded_path)
        try:
            digest.update(candidate.read_bytes())
        except OSError:
            digest.update(b"<unreadable>")
    return f"working-tree:{revision}:{digest.hexdigest()[:12]}"


def _supervised_dataset(data: PreparedData, target_source: SourceKind) -> SupervisedDataset:
>>>>>>> 1527107 (Extend UI and ML analysis materials)
    from source.ml.features import build_supervised_dataset

    return build_supervised_dataset(
        data,
        target_signal_id=target_signal,
        target_source=target_source,
        feature_source=SourceKind.PAK,
        horizon_minutes=60,
        telemetry_signals=TRAINING_TELEMETRY_SIGNALS,
    )


def train_command(
    dataset: str | Path,
    target_source: SourceKind | str | Path | None = SourceKind.PAK,
    output: str | Path | None = None,
    *,
    target_signal: str = "ht:2:Mg.Sulfur",
    with_uncertainty: bool = False,
    with_safety: bool = False,
    config_path: str | Path = "config/runtime.toml",
    root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Train the selected point model and optionally its frozen upper model."""
    from source.ml.safety import save_safety_model
    from source.ml.train import train_model
    from source.ml.uncertainty import save_stage5_model

<<<<<<< HEAD
    source, target_output = _source_or_output(target_source, output)
=======
    if with_safety and not with_uncertainty:
        raise ValueError("--with-safety requires --with-uncertainty")

>>>>>>> f0ad14f (Complete ML stages and safety diagnostics)
    config = load_runtime_config(_resolve_path(config_path, root))
    data = _load_current_prepared_dataset(dataset, config, root)
    revision = _git_revision(root)
<<<<<<< HEAD
    data = load_prepared_dataset(_resolve_path(dataset, root))
    supervised = _supervised_dataset(data, source, target_signal)
    models_root = _resolve_path(target_output if target_output is not None else config.models_dir, root)
=======
    supervised = _supervised_dataset(data, SourceKind.PAK)
    models_root = _resolve_path(output if output is not None else config.models_dir, root)
>>>>>>> e70cafe (fix)
    point = train_model(
        supervised,
        models_root=models_root,
        training_dataset_id=data.manifest.dataset_id,
        tag_dictionary_sha256=data.manifest.tag_dictionary_sha256,
        git_commit=revision,
        source_timezone=config.source_timezone,
        horizon_minutes=config.horizon_minutes,
        seed=config.seed,
    )
    result: dict[str, object] = {
        "dataset_id": data.manifest.dataset_id,
        "point_model_id": point.bundle.metadata.model_id,
        "point_model_path": point.artifact_dir.as_posix(),
        "selected_model": point.metrics["selected_model"],
        "target_signal": point.bundle.metadata.target_signal,
        "target_source": point.bundle.metadata.target_source,
    }
    if with_uncertainty:
        recipe = f"{data.manifest.dataset_id}:{point.bundle.metadata.model_id}:upper:0.95"
        upper_id = f"sulfur-upper-{hashlib.sha256(recipe.encode()).hexdigest()[:12]}"
        upper_path = models_root / upper_id
        upper, fitted = save_stage5_model(
            upper_path,
            supervised,
            point.bundle,
            source_timezone=config.source_timezone,
            seed=config.seed,
        )
        result.update(
            {
                "upper_model_id": upper.metadata.model_id,
                "upper_model_path": upper_path.as_posix(),
                "test_coverage": fitted.report["test_coverage"],
                "test_applicability_rate": fitted.report["test_applicability_rate"],
            }
        )
        if with_safety:
            safety_recipe = f"{data.manifest.dataset_id}:{upper.metadata.model_id}:safety:1.1"
            safety_id = f"sulfur-safety-{hashlib.sha256(safety_recipe.encode()).hexdigest()[:12]}"
            safety_path = models_root / safety_id
            safety, safety_fit = save_safety_model(
                safety_path,
                supervised,
                upper,
                source_timezone=config.source_timezone,
                seed=config.seed,
                include_lightgbm=True,
            )
            result.update(
                {
                    "safety_model_id": safety.metadata.model_id,
                    "safety_model_path": safety_path.as_posix(),
                    "alarm_threshold": safety_fit.report["alarm_threshold"],
                    "validation_safety": safety_fit.report["validation_policy"],
                    "test_safety": safety_fit.report["test"],
                    "test_transition_recall": safety_fit.report["test_transition_recall"],
                    "test_joint_applicability_rate": safety_fit.report["test_applicability_rate"],
                }
            )
    return result


def diagnose_ml_command(
    dataset: str | Path,
    *,
    config_path: str | Path = "config/runtime.toml",
    root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Run read-only alignment, drift and telemetry diagnostics."""
    config = load_runtime_config(_resolve_path(config_path, root))
    data = _load_current_prepared_dataset(dataset, config, root)
    from source.ml.diagnostics import build_diagnostic_report

    supervised = _supervised_dataset(data, SourceKind.PAK)
    return build_diagnostic_report(data, supervised)


def train_v2_shadow_command(
    dataset: str | Path,
    output: str | Path | None = None,
    *,
    allow_dirty_shadow: bool = False,
    config_path: str | Path = "config/runtime.toml",
    root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Train and persist a schema-1.2 artifact that is restricted to shadow use."""
    from source.ml.v2 import save_episode_safety_model

    config = load_runtime_config(_resolve_path(config_path, root))
    data = _load_current_prepared_dataset(dataset, config, root)
    revision = _shadow_git_revision(root, allow_dirty=allow_dirty_shadow)
    supervised = _supervised_dataset(data, SourceKind.PAK)
    models_root = _resolve_path(output if output is not None else config.models_dir, root)
    recipe = f"{data.manifest.dataset_id}:episode-multihorizon:1.2:{revision}"
    model_id = f"sulfur-v2-shadow-{hashlib.sha256(recipe.encode()).hexdigest()[:12]}"
    path = models_root / model_id
    bundle, fitted = save_episode_safety_model(
        path,
        data,
        supervised,
        git_commit=revision,
        seed=config.seed,
    )
    return {
        "dataset_id": data.manifest.dataset_id,
        "model_id": bundle.metadata.model_id,
        "model_path": path.as_posix(),
        "schema_version": bundle.metadata.schema_version,
        "production_status": "shadow_only",
        "selected_family": fitted.report["selected_family"],
        "threshold_metrics": fitted.report["threshold_metrics"],
        "audit_2026": fitted.report["audit_2026"],
    }


def replay_v2_shadow_command(
    dataset: str | Path,
    model_path: str | Path,
    as_of: datetime,
    *,
    config_path: str | Path = "config/runtime.toml",
    root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Serve one schema-1.2 PAK episode forecast without enabling controls."""
    from source.ml.artifacts import load_model
    from source.ml.features import SupervisedDataset, _feature_matrix
    from source.ml.v2 import build_episode_dataset

    config = load_runtime_config(_resolve_path(config_path, root))
    data = _load_current_prepared_dataset(dataset, config, root)
    bundle = load_model(
        _resolve_path(model_path, root),
        trusted=True,
        expected_schema_version="1.2",
        expected_horizon_minutes=60,
        expected_tag_dictionary_sha256=data.manifest.tag_dictionary_sha256,
        expected_target_signal="ht:2:Mg.Sulfur",
        expected_target_source="pak",
        expected_target_unit="mg/kg",
    )
    definition = bundle.metadata.processing.get("feature_definition", {})
    raw_signals = definition.get("telemetry_signals", TRAINING_TELEMETRY_SIGNALS)
    if not isinstance(raw_signals, (list, tuple)):
        raise ValueError("v2 artifact has invalid telemetry_signals definition")
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    cutoff = cutoff.tz_convert("UTC")
    signals = tuple(str(signal) for signal in raw_signals)
    feature_matrix = _feature_matrix(
        data,
        pd.DatetimeIndex([cutoff]),
        signals,
        "ht:2:Mg.Sulfur",
        SourceKind.PAK,
        "mg/kg",
    )
    if feature_matrix.empty or pd.isna(feature_matrix.iloc[0][bundle.metadata.baseline_feature]):
        raise ValueError("no PAK state is available at or before as_of")
    base_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "observation_id": ["serving-v2"],
                    "as_of": [cutoff],
                    "target_at": [pd.NaT],
                    "target_available_at": [pd.NaT],
                    "target_source": ["pak"],
                    "target_signal": ["ht:2:Mg.Sulfur"],
                    "target_unit": ["mg/kg"],
                    "y": [float("nan")],
                    "baseline": [float(feature_matrix.iloc[0][bundle.metadata.baseline_feature])],
                }
            ),
            feature_matrix.reset_index(drop=True),
        ],
        axis="columns",
    )
    base = SupervisedDataset(
        frame=base_frame,
        feature_names=tuple(feature_matrix.columns),
        target_signal_id="ht:2:Mg.Sulfur",
        target_unit="mg/kg",
        target_source=SourceKind.PAK,
        feature_source=SourceKind.PAK,
        baseline_feature=bundle.metadata.baseline_feature,
        feature_definition=dict(definition),
        excluded_counts={},
    )
    episode = build_episode_dataset(data, base)
    features = episode.frame.loc[:, list(bundle.feature_names)]
    row = episode.frame.iloc[[0]]
    forecast = bundle.predict_v2(features)[0]
    return {
        "model_id": bundle.metadata.model_id,
        "schema_version": bundle.metadata.schema_version,
        "as_of": pd.Timestamp(row.iloc[0]["as_of"]).isoformat(),
        "production_status": "shadow_only",
        "git_revision_label": bundle.metadata.git_commit,
        "supports_actions": False,
        "forecast": forecast,
    }


def train_action_shadow_command(
    dataset: str | Path,
    *,
    output: str | Path | None = None,
    config_path: str | Path = "config/runtime.toml",
    root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Fit the matched historical action study without enabling recommendations."""
    from source.ml.action_effects import (
        build_historical_action_dataset,
        fit_historical_action_model,
        save_historical_action_model,
    )

    config = load_runtime_config(_resolve_path(config_path, root))
    data = _load_current_prepared_dataset(dataset, config, root)
    research = build_historical_action_dataset(data)
    model = fit_historical_action_model(research, seed=config.seed)
    models_root = _resolve_path(output if output is not None else config.models_dir, root)
    model_id = f"action-shadow-{data.manifest.dataset_id}-v2"
    artifact_path = models_root / model_id
    if not artifact_path.exists():
        save_historical_action_model(
            artifact_path, model, training_dataset_id=data.manifest.dataset_id
        )
    return {
        "dataset_id": data.manifest.dataset_id,
        "model_id": model_id,
        "model_path": artifact_path.as_posix(),
        "supports_actions": False,
        "evidence_gate_passed": model.report["evidence_gate_passed"],
        "report": dict(model.report),
    }


def action_shadow_estimate_command(
    dataset: str | Path,
    model_path: str | Path,
    control: str,
    delta: float,
    as_of: datetime,
    *,
    config_path: str | Path = "config/runtime.toml",
    root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Calculate one research-only historical action scenario at an available time."""
    from source.ml.action_effects import _action_timeline, load_historical_action_model

    config = load_runtime_config(_resolve_path(config_path, root))
    data = _load_current_prepared_dataset(dataset, config, root)
    model = load_historical_action_model(
        _resolve_path(model_path, root),
        trusted=True,
        expected_dataset_id=data.manifest.dataset_id,
    )
    timeline = _action_timeline(data)
    timestamp = timeline["timestamp"]
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    selected = timeline.loc[timestamp <= cutoff.tz_convert("UTC")]
    if selected.empty:
        raise ValueError("no complete PAK/telemetry action state is available at as_of")
    state_row = selected.iloc[-1]
    state = {
        name: float(state_row[name])
        for name in (
            "baseline_sulfur",
            "sulfur_slope_60m",
            "ht:P8",
            "ht:F19",
            "ht:T11",
            "ht:F26",
        )
    }
    estimate = model.estimate(state, control, delta)
    payload = estimate.as_ui_payload()
    payload.update(
        {
            "as_of": pd.Timestamp(state_row["timestamp"]).isoformat(),
            "state": state,
            "supports_actions": False,
        }
    )
    return payload


def evaluate_lims_correction_command(
    dataset: str | Path,
    pak_model_path: str | Path,
    *,
    output: str | Path | None = None,
    config_path: str | Path = "config/runtime.toml",
    root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Evaluate the delayed LIMS correction as a separate, non-operational layer."""
    from source.ml.safety import fit_lims_correction

    config = load_runtime_config(_resolve_path(config_path, root))
    data = _load_current_prepared_dataset(dataset, config, root)
    pak_model = _load_trusted_model(pak_model_path, data, root)
    correction = fit_lims_correction(
        _supervised_dataset(data, SourceKind.LIMS),
        pak_model,
        data=data,
        source_timezone=config.source_timezone,
        seed=config.seed,
    )
    result: dict[str, object] = {
        "dataset_id": data.manifest.dataset_id,
        "pak_model_id": pak_model.metadata.model_id,
        "production_status": (
            "eligible_for_shadow" if correction.report["promotion_eligible"] else "research_only"
        ),
        "report": correction.report,
    }
    if output is not None:
        result["report_path"] = _write_json_report(output, result, root).as_posix()
    return result


def evaluate_action_residualization_command(
    dataset: str | Path,
    *,
    output: str | Path | None = None,
    config_path: str | Path = "config/runtime.toml",
    root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Run the cross-fitted historical action association benchmark."""
    from source.ml.action_effects import (
        build_historical_action_dataset,
        evaluate_temporal_residualization,
    )

    config = load_runtime_config(_resolve_path(config_path, root))
    data = _load_current_prepared_dataset(dataset, config, root)
    report = evaluate_temporal_residualization(build_historical_action_dataset(data))
    result: dict[str, object] = {
        "dataset_id": data.manifest.dataset_id,
        "production_status": "research_only",
        "report": report,
    }
    if output is not None:
        result["report_path"] = _write_json_report(output, result, root).as_posix()
    return result


def ablate_v2_features_command(
    dataset: str | Path,
    *,
    groups: tuple[str, ...],
    output: str | Path | None = None,
    config_path: str | Path = "config/runtime.toml",
    root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Run the fixed 2024 PAK/HT/AVT feature ablation without audit selection."""
    from source.ml.v2 import ablate_episode_features

    config = load_runtime_config(_resolve_path(config_path, root))
    data = _load_current_prepared_dataset(dataset, config, root)
    report = ablate_episode_features(data, groups=groups)
    result = {"dataset_id": data.manifest.dataset_id, **report}
    if output is not None:
        result["report_path"] = _write_json_report(output, result, root).as_posix()
    return result


def _load_trusted_model(model_path: str | Path, data: PreparedData, root: Path) -> ModelBundle:
    from source.ml.artifacts import load_model

    manifest = data.manifest
    return load_model(
        _resolve_path(model_path, root),
        trusted=True,
        expected_horizon_minutes=60,
        expected_tag_dictionary_sha256=manifest.tag_dictionary_sha256,
        expected_target_signal="ht:2:Mg.Sulfur",
        expected_target_unit="mg/kg",
    )


def _coerce_split_or_source(
    value: SourceKind | str | None,
    split: SplitName,
) -> tuple[SourceKind | None, SplitName]:
    if isinstance(value, str) and value in {"train", "validation", "test"}:
        return None, value
    if value is None:
        return None, split
    return SourceKind(value), split


def evaluate_command(
    dataset: str | Path,
    model_path: str | Path,
    target_source: SourceKind | str | None = SourceKind.PAK,
    output: str | Path | None = None,
    *,
    split: SplitName = "test",
    config_path: str | Path = "config/runtime.toml",
    root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Evaluate a frozen model and baseline on identical historical timestamps."""
    from source.ml.evaluate import evaluate_model, write_evaluation

    config = load_runtime_config(_resolve_path(config_path, root))
    data = _load_current_prepared_dataset(dataset, config, root)
    model = _load_trusted_model(model_path, data, root)
    source, selected_split = _coerce_split_or_source(target_source, split)
    evaluation_source = source or SourceKind(model.metadata.target_source)
    supervised = _supervised_dataset(data, evaluation_source, model.metadata.target_signal)
    evaluation = evaluate_model(
        supervised,
        model,
        split=selected_split,
        source_timezone=config.source_timezone,
    )
    destination = (
        _resolve_path(output, root)
        if output is not None
        else root
        / config.reports_dir
        / "final"
        / model.metadata.model_id
        / f"{evaluation_source.value}_{selected_split}"
    )
    write_evaluation(destination, evaluation)
    metrics = evaluation.report["metrics"]
    return {
        "model_id": model.metadata.model_id,
        "target_source": evaluation_source.value,
        "split": selected_split,
        "report_path": (destination / "metrics.json").as_posix(),
        "baseline_mae": metrics["baseline"]["mae"],
        "model_mae": metrics["model"]["mae"],
        "coverage": evaluation.report["coverage"],
    }


def replay_command(
    dataset: str | Path,
    model_path: str | Path,
    scenario: str | Path,
    as_of: datetime,
    *,
    action_model_path: str | Path | None = None,
    config_path: str | Path = "config/runtime.toml",
    run_dir: str | Path | None = None,
    root: Path = PROJECT_ROOT,
) -> Recommendation:
    """Replay one historical point with a compatible frozen forecast model."""
    from source.orchestrator import run_cycle

    config = load_runtime_config(_resolve_path(config_path, root))
    data = _load_current_prepared_dataset(dataset, config, root)
    model = _load_trusted_model(model_path, data, root)
    if action_model_path is not None:
        from source.ml.action_effects import (
            combine_forecast_action_model,
            load_verified_action_model,
        )

        action_model = load_verified_action_model(
            _resolve_path(action_model_path, root),
            trusted=True,
            expected_dataset_id=data.manifest.dataset_id,
            expected_config_sha256=data.manifest.config_sha256,
            expected_tag_dictionary_sha256=data.manifest.tag_dictionary_sha256,
            expected_telemetry_rules_sha256=data.manifest.telemetry_rules_sha256,
        )
        model = combine_forecast_action_model(model, action_model)
    scenario_path = Path(scenario)
    if not scenario_path.suffix:
        scenario_path = Path("config/scenarios") / f"{scenario}.json"
    scenario_config = load_scenario(_resolve_path(scenario_path, root))
    return run_cycle(
        data=data,
        as_of=as_of,
        model=model,
        scenario=scenario_config,
        config=config,
        context=DecisionContext(),
        run_dir=_resolve_path(run_dir if run_dir is not None else config.runs_dir, root),
    )


def run_history_command(
    dataset: str | Path,
    model: str | Path,
    as_of: datetime,
    *,
    trusted_model: bool = False,
    scenario: str | Path = "history",
    config_path: str | Path = "config/runtime.toml",
    run_dir: str | Path | None = None,
    root: Path = PROJECT_ROOT,
) -> Recommendation:
    """Compatibility wrapper for the older ``run-history`` command."""
    if not trusted_model:
        raise ValueError("run-history requires --trusted-model for local joblib artifacts")
    return replay_command(
        dataset,
        model,
        scenario,
        as_of,
        config_path=config_path,
        run_dir=run_dir,
        root=root,
    )


def accept_stage6(
    root: Path = PROJECT_ROOT,
    run_dir: str | Path | None = None,
) -> dict[str, object]:
    """Compatibility wrapper around the Stage-6 acceptance package."""
    from source.acceptance import run_acceptance_suite

    output_dir = _resolve_path(run_dir if run_dir is not None else "runs/stage6", root)
    report = run_acceptance_suite(
        root,
        episodes_path=root / "config/demo_episodes.json",
        output_dir=output_dir,
    )
    return {
        "passed": True,
        "report": report,
        "environment": {
            "python_version": platform.python_version(),
        },
        "limitations": list(STAGE6_LIMITATIONS),
        "issues": [],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the CLI command and run the requested backend command."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Нефтекод recommendation prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-stage0", help="validate contracts, configs and fixtures")
    demo = subparsers.add_parser("run-model-demo", help="run a stage-1 model-demo scenario")
    demo.add_argument(
        "scenario",
        choices=MODEL_DEMO_SCENARIOS,
        help="scenario id from config/scenarios",
    )
    demo_alias = subparsers.add_parser("demo", help="run a deterministic model-demo episode")
    demo_alias.add_argument(
        "scenario",
        choices=MODEL_DEMO_SCENARIOS,
    )
    prepare = subparsers.add_parser(
        "prepare", help="prepare original materials into data/processed"
    )
    prepare.add_argument("--materials", default="materials", help="directory with original files")
    prepare.add_argument("--config", default="config/runtime.toml", help="runtime config path")
    prepare.add_argument("--output", default=None, help="prepared dataset root")
    state = subparsers.add_parser("build-state", help="build a ProcessState from prepared data")
    state.add_argument("--dataset", required=True, help="prepared dataset directory")
    state.add_argument("--scenario", default="history", help="scenario id or JSON path")
    state.add_argument(
        "--as-of", required=True, type=_parse_as_of, help="timezone-aware ISO datetime"
    )
    state.add_argument("--config", default="config/runtime.toml", help="runtime config path")
    train = subparsers.add_parser("train", help="train and persist the frozen sulfur models")
    train.add_argument("--dataset", required=True, help="prepared dataset directory")
    train.add_argument(
        "--target-source",
        choices=(SourceKind.PAK.value, SourceKind.LIMS.value),
        default=SourceKind.PAK.value,
        help="quality source used as supervised target",
    )
    train.add_argument("--target-signal", default="ht:2:Mg.Sulfur", help="target signal id")
    train.add_argument("--output", default=None, help="model artifacts root")
    train.add_argument("--config", default="config/runtime.toml", help="runtime config path")
    train.add_argument(
        "--with-uncertainty",
        action="store_true",
        help="also fit the Stage-5 empirical upper model",
    )
    train.add_argument(
        "--with-safety",
        action="store_true",
        help="also fit the calibrated safety alarm and joint applicability model",
    )
    diagnose = subparsers.add_parser(
        "diagnose-ml", help="inspect sulfur alignment, temporal drift and candidate signals"
    )
    diagnose.add_argument("--dataset", required=True, help="prepared dataset directory")
    diagnose.add_argument("--config", default="config/runtime.toml", help="runtime config path")
    v2 = subparsers.add_parser(
        "train-v2-shadow", help="train the episode-aware schema-1.2 shadow artifact"
    )
    v2.add_argument("--dataset", required=True, help="prepared dataset directory")
    v2.add_argument("--output", default=None, help="model artifacts root")
    v2.add_argument("--config", default="config/runtime.toml", help="runtime config path")
    v2.add_argument(
        "--allow-dirty-shadow",
        action="store_true",
        help="allow a non-clean worktree; artifact remains shadow_only and cannot be frozen",
    )
    v2_replay = subparsers.add_parser(
        "replay-v2-shadow", help="serve one schema-1.2 PAK episode forecast"
    )
    v2_replay.add_argument("--dataset", required=True, help="prepared dataset directory")
    v2_replay.add_argument("--model", required=True, help="schema-1.2 shadow artifact")
    v2_replay.add_argument(
        "--at", required=True, type=_parse_as_of, help="timezone-aware ISO time"
    )
    v2_replay.add_argument("--config", default="config/runtime.toml", help="runtime config path")
    action_shadow = subparsers.add_parser(
        "evaluate-action-shadow", help="evaluate matched P8/F19 sulfur effects"
    )
    action_shadow.add_argument("--dataset", required=True, help="prepared dataset directory")
    action_shadow.add_argument("--output", default=None, help="action artifact root")
    action_shadow.add_argument(
        "--config", default="config/runtime.toml", help="runtime config path"
    )
    action_estimate = subparsers.add_parser(
        "action-shadow-estimate", help="calculate one non-advisory P8/F19 historical scenario"
    )
    action_estimate.add_argument("--dataset", required=True, help="prepared dataset directory")
    action_estimate.add_argument(
        "--model", required=True, help="trusted action artifact directory"
    )
    action_estimate.add_argument("--control", choices=("ht:P8", "ht:F19"), required=True)
    action_estimate.add_argument("--delta", type=float, required=True)
    action_estimate.add_argument(
        "--at", required=True, type=_parse_as_of, help="timezone-aware ISO time"
    )
    action_estimate.add_argument(
        "--config", default="config/runtime.toml", help="runtime config path"
    )
    lims_correction = subparsers.add_parser(
        "evaluate-lims-correction", help="evaluate a delayed PAK-to-LIMS correction"
    )
    lims_correction.add_argument("--dataset", required=True, help="prepared dataset directory")
    lims_correction.add_argument("--pak-model", required=True, help="trusted local PAK model")
    lims_correction.add_argument(
        "--config", default="config/runtime.toml", help="runtime config path"
    )
    lims_correction.add_argument("--output", default=None, help="new JSON report path")
    residualization = subparsers.add_parser(
        "evaluate-action-residualization",
        help="cross-fitted residual P8/F19 association benchmark",
    )
    residualization.add_argument("--dataset", required=True, help="prepared dataset directory")
    residualization.add_argument("--output", default=None, help="new JSON report path")
    residualization.add_argument(
        "--config", default="config/runtime.toml", help="runtime config path"
    )
    ablation = subparsers.add_parser(
        "ablate-v2-features", help="compare PAK/HT/AVT groups on 2024 temporal folds"
    )
    ablation.add_argument("--dataset", required=True, help="prepared dataset directory")
    ablation.add_argument("--output", default=None, help="new JSON report path")
    ablation.add_argument("--config", default="config/runtime.toml", help="runtime config path")
    ablation.add_argument(
        "--group",
        action="append",
        choices=("pak_only", "ht_context", "k2_state", "k2_circulation", "diesel_cut"),
        required=True,
        help="one fixed group; repeat this argument for a deliberate comparison",
    )
    evaluate = subparsers.add_parser(
        "evaluate", help="compare a frozen model with persistence on one temporal split"
    )
    evaluate.add_argument("--dataset", required=True, help="prepared dataset directory")
    evaluate.add_argument("--model", required=True, help="trusted local model directory")
    evaluate.add_argument("--source", choices=("pak", "lims"), required=True)
    evaluate.add_argument("--split", choices=("train", "validation", "test"), default="test")
    evaluate.add_argument("--output", default=None, help="new evaluation directory")
    evaluate.add_argument("--config", default="config/runtime.toml", help="runtime config path")
    replay = subparsers.add_parser("replay", help="replay one historical forecast point")
    replay.add_argument("--dataset", required=True, help="prepared dataset directory")
    replay.add_argument("--model", required=True, help="trusted local model directory")
    replay.add_argument(
        "--action-model",
        default=None,
        help="optional verified production action artifact directory",
    )
    replay.add_argument("--scenario", default="history", help="scenario id or JSON path")
    replay.add_argument("--at", required=True, type=_parse_as_of, help="timezone-aware ISO time")
    replay.add_argument("--config", default="config/runtime.toml", help="runtime config path")
    replay.add_argument("--run-dir", default=None, help="journal directory override")
    history = subparsers.add_parser(
        "run-history", help="legacy alias for replay with a trusted forecast artifact"
    )
    history.add_argument("--dataset", required=True, help="prepared dataset directory")
    history.add_argument("--model", required=True, help="trusted local model directory")
    history.add_argument(
        "--trusted-model",
        action="store_true",
        help="explicitly trust this local joblib artifact after metadata checks",
    )
    history.add_argument(
        "--as-of", required=True, type=_parse_as_of, help="timezone-aware ISO datetime"
    )
    history.add_argument("--scenario", default="history", help="scenario id or JSON path")
    history.add_argument("--config", default="config/runtime.toml", help="runtime config path")
    history.add_argument("--run-dir", default=None, help="journal directory override")
    acceptance = subparsers.add_parser(
        "acceptance", help="run fixed Stage-6 episodes and export their journals"
    )
    acceptance.add_argument("--episodes", default="config/demo_episodes.json")
    acceptance.add_argument("--output", required=True, help="new acceptance output directory")
    accept = subparsers.add_parser(
        "accept-stage6", help="legacy alias for Stage-6 acceptance checks"
    )
    accept.add_argument(
        "--run-dir",
        default="runs/stage6",
        help="fresh output directory for acceptance journals",
    )
    freeze = subparsers.add_parser("verify-model-freeze", help="verify frozen artifact hashes")
    freeze.add_argument("--manifest", default="config/model_freeze.json")
    export = subparsers.add_parser("export-journal", help="export complete journals to ZIP")
    export.add_argument("--run", action="append", required=True, help="run id or directory")
    export.add_argument("--output", required=True, help="new ZIP path")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-stage0":
            validation_result = validate_stage0()
            print(json.dumps(validation_result, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command in {"run-model-demo", "demo"}:
            recommendation = run_model_demo(args.scenario)
            print(recommendation.model_dump_json(indent=2))
            return 0
        if args.command == "prepare":
            preparation_result = prepare_command(args.materials, args.config, args.output)
            print(json.dumps(preparation_result, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "build-state":
            state_result = build_state_command(
                args.dataset, args.scenario, args.as_of, args.config
            )
            print(state_result.model_dump_json(indent=2))
            return 0
        if args.command == "train":
            training_result = train_command(
                args.dataset,
                SourceKind(args.target_source),
                args.output,
                target_signal=args.target_signal,
                with_uncertainty=args.with_uncertainty,
                with_safety=args.with_safety,
                config_path=args.config,
            )
            print(json.dumps(training_result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "diagnose-ml":
            diagnostic_result = diagnose_ml_command(args.dataset, config_path=args.config)
            print(json.dumps(diagnostic_result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "train-v2-shadow":
            v2_result = train_v2_shadow_command(
                args.dataset,
                args.output,
                allow_dirty_shadow=args.allow_dirty_shadow,
                config_path=args.config,
            )
            print(json.dumps(v2_result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "replay-v2-shadow":
            v2_replay_result = replay_v2_shadow_command(
                args.dataset, args.model, args.at, config_path=args.config
            )
            print(json.dumps(v2_replay_result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "evaluate-action-shadow":
            action_result = train_action_shadow_command(
                args.dataset, output=args.output, config_path=args.config
            )
            print(json.dumps(action_result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "action-shadow-estimate":
            action_result = action_shadow_estimate_command(
                args.dataset,
                args.model,
                args.control,
                args.delta,
                args.at,
                config_path=args.config,
            )
            print(json.dumps(action_result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "evaluate-lims-correction":
            lims_result = evaluate_lims_correction_command(
                args.dataset, args.pak_model, output=args.output, config_path=args.config
            )
            print(json.dumps(lims_result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "evaluate-action-residualization":
            residual_result = evaluate_action_residualization_command(
                args.dataset, output=args.output, config_path=args.config
            )
            print(json.dumps(residual_result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "ablate-v2-features":
            ablation_result = ablate_v2_features_command(
                args.dataset,
                groups=tuple(args.group),
                output=args.output,
                config_path=args.config,
            )
            print(json.dumps(ablation_result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "evaluate":
            evaluation_result = evaluate_command(
                args.dataset,
                args.model,
                SourceKind(args.source),
                args.output,
                split=args.split,
                config_path=args.config,
            )
            print(json.dumps(evaluation_result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "replay":
            replay_result = replay_command(
                args.dataset,
                args.model,
                args.scenario,
                args.at,
                action_model_path=args.action_model,
                config_path=args.config,
                run_dir=args.run_dir,
            )
            print(replay_result.model_dump_json(indent=2))
            return 0
        if args.command == "run-history":
            history_result = run_history_command(
                args.dataset,
                args.model,
                args.as_of,
                trusted_model=args.trusted_model,
                scenario=args.scenario,
                config_path=args.config,
                run_dir=args.run_dir,
            )
            print(history_result.model_dump_json(indent=2))
            return 0
        if args.command == "acceptance":
            from source.acceptance import run_acceptance_suite

            acceptance_result = run_acceptance_suite(
                PROJECT_ROOT,
                episodes_path=_resolve_path(args.episodes),
                output_dir=_resolve_path(args.output),
            )
            print(json.dumps(acceptance_result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "accept-stage6":
            acceptance_result = accept_stage6(run_dir=args.run_dir)
            print(json.dumps(acceptance_result, ensure_ascii=False, indent=2))
            return 0 if acceptance_result["passed"] else 1
        if args.command == "verify-model-freeze":
            from source.acceptance import verify_model_freeze

            freeze_result = verify_model_freeze(PROJECT_ROOT, _resolve_path(args.manifest))
            print(json.dumps(freeze_result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "export-journal":
            from source.acceptance import export_journals

            config = load_runtime_config(PROJECT_ROOT / "config/runtime.toml")
            journals = []
            for value in args.run:
                requested = Path(value)
                directory = (
                    _resolve_path(requested)
                    if requested.is_absolute() or len(requested.parts) > 1
                    else PROJECT_ROOT / config.runs_dir / requested
                )
                journals.append((directory.name, directory))
            archive = export_journals(journals, _resolve_path(args.output))
            print(json.dumps({"journal_archive": archive.as_posix()}, ensure_ascii=False))
            return 0
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
