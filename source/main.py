"""Small CLI surface implemented for the current backend stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

from source.config import load_runtime_config, load_scenario, load_tag_dictionary
from source.contracts import DecisionContext, ProcessState, Recommendation, SourceKind
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def validate_stage0(root: Path = PROJECT_ROOT) -> dict[str, int]:
    """Validate shared configs and serialized contract fixtures."""
    load_runtime_config(root / "config/runtime.toml")
    tags = load_tag_dictionary(root / "config/tags.csv")
    scenarios = [load_scenario(path) for path in sorted((root / "config/scenarios").glob("*.json"))]
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


def run_model_demo(scenario_id: str, root: Path = PROJECT_ROOT) -> Recommendation:
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
        run_dir=root / config.runs_dir,
    )


def _resolve_path(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """Resolve CLI paths relative to the project root unless they are absolute."""
    value = Path(path)
    return value if value.is_absolute() else root / value


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
    data = load_prepared_dataset(_resolve_path(dataset, root))
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


def _supervised_dataset(data: PreparedData, target_source: SourceKind) -> SupervisedDataset:
    from source.ml.features import build_supervised_dataset

    return build_supervised_dataset(
        data,
        target_signal_id="ht:2:Mg.Sulfur",
        target_source=target_source,
        feature_source=SourceKind.PAK,
        horizon_minutes=60,
    )


def train_command(
    dataset: str | Path,
    output: str | Path | None = None,
    *,
    with_uncertainty: bool = False,
    config_path: str | Path = "config/runtime.toml",
    root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Train the selected point model and optionally its frozen upper model."""
    from source.ml.train import train_model
    from source.ml.uncertainty import save_stage5_model

    config = load_runtime_config(_resolve_path(config_path, root))
    revision = _git_revision(root)
    data = load_prepared_dataset(_resolve_path(dataset, root))
    supervised = _supervised_dataset(data, SourceKind.PAK)
    models_root = _resolve_path(output if output is not None else config.models_dir, root)
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


def evaluate_command(
    dataset: str | Path,
    model_path: str | Path,
    target_source: SourceKind,
    output: str | Path | None = None,
    *,
    split: SplitName = "test",
    config_path: str | Path = "config/runtime.toml",
    root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Evaluate a frozen model and baseline on identical historical timestamps."""
    from source.ml.evaluate import evaluate_model, write_evaluation

    config = load_runtime_config(_resolve_path(config_path, root))
    data = load_prepared_dataset(_resolve_path(dataset, root))
    model = _load_trusted_model(model_path, data, root)
    supervised = _supervised_dataset(data, target_source)
    evaluation = evaluate_model(
        supervised,
        model,
        split=split,
        source_timezone=config.source_timezone,
    )
    destination = (
        _resolve_path(output, root)
        if output is not None
        else root
        / config.reports_dir
        / "final"
        / model.metadata.model_id
        / f"{target_source.value}_{split}"
    )
    write_evaluation(destination, evaluation)
    metrics = evaluation.report["metrics"]
    return {
        "model_id": model.metadata.model_id,
        "target_source": target_source.value,
        "split": split,
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
    config_path: str | Path = "config/runtime.toml",
    root: Path = PROJECT_ROOT,
) -> Recommendation:
    """Replay one historical point with a compatible frozen forecast model."""
    from source.orchestrator import run_cycle

    config = load_runtime_config(_resolve_path(config_path, root))
    data = load_prepared_dataset(_resolve_path(dataset, root))
    model = _load_trusted_model(model_path, data, root)
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
        run_dir=root / config.runs_dir,
    )


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
        choices=("blend_normal", "blend_risk", "blend_missing"),
        help="scenario id from config/scenarios",
    )
    demo_alias = subparsers.add_parser("demo", help="run a deterministic model-demo episode")
    demo_alias.add_argument(
        "scenario",
        choices=("blend_normal", "blend_risk", "blend_missing"),
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
    train.add_argument("--output", default=None, help="model artifacts root")
    train.add_argument("--config", default="config/runtime.toml", help="runtime config path")
    train.add_argument(
        "--with-uncertainty",
        action="store_true",
        help="also fit the Stage-5 empirical upper model",
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
    replay.add_argument("--scenario", default="history", help="scenario id or JSON path")
    replay.add_argument("--at", required=True, type=_parse_as_of, help="timezone-aware ISO time")
    replay.add_argument("--config", default="config/runtime.toml", help="runtime config path")
    acceptance = subparsers.add_parser(
        "acceptance", help="run fixed Stage-6 episodes and export their journals"
    )
    acceptance.add_argument("--episodes", default="config/demo_episodes.json")
    acceptance.add_argument("--output", required=True, help="new acceptance output directory")
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
            state_result = build_state_command(args.dataset, args.scenario, args.as_of, args.config)
            print(state_result.model_dump_json(indent=2))
            return 0
        if args.command == "train":
            training_result = train_command(
                args.dataset,
                args.output,
                with_uncertainty=args.with_uncertainty,
                config_path=args.config,
            )
            print(json.dumps(training_result, ensure_ascii=False, indent=2))
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
                config_path=args.config,
            )
            print(replay_result.model_dump_json(indent=2))
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
