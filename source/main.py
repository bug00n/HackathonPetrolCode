"""Small CLI surface implemented for the current backend stages."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast

from source.config import load_runtime_config, load_scenario, load_tag_dictionary
from source.contracts import (
    DecisionContext,
    ProcessState,
    Recommendation,
    RecommendationStatus,
    ScenarioConfig,
    SourceKind,
)
from source.data import build_state, load_prepared_dataset, prepare_dataset, write_prepared_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE6_SCENARIOS: tuple[str, ...] = ("blend_normal", "blend_risk", "blend_missing")
STAGE6_EXPECTED_STATUSES: dict[str, str] = {
    "blend_normal": RecommendationStatus.HOLD.value,
    "blend_risk": RecommendationStatus.RECOMMEND.value,
    "blend_missing": RecommendationStatus.ABSTAIN.value,
}
STAGE6_JOURNAL_FILES: tuple[str, ...] = (
    "result.json",
    "input.json",
    "trace.jsonl",
    "candidates.jsonl",
)
STAGE6_LIMITATIONS: tuple[str, ...] = (
    "Stage 6 is an acceptance and demonstration layer, not a new ML or action model.",
    "Real setpoint recommendations remain disabled until a validated action model exists.",
    "Model-demo recommendations are synthetic; history artifact serving remains forecast-only.",
)


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


def _git_commit(root: Path) -> str:
    """Return a best-effort commit id for model metadata."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def train_command(
    dataset: str | Path,
    target_source: str,
    output: str | Path | None = None,
    target_signal: str = "ht:2:Mg.Sulfur",
    config_path: str | Path = "config/runtime.toml",
    root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Train a reproducible local sulfur forecast artifact from prepared data."""
    from source.ml.features import build_supervised_dataset
    from source.ml.train import train_model

    config = load_runtime_config(_resolve_path(config_path, root))
    data = load_prepared_dataset(_resolve_path(dataset, root))
    supervised = build_supervised_dataset(
        data,
        target_signal_id=target_signal,
        target_source=SourceKind(target_source),
        horizon_minutes=config.horizon_minutes,
    )
    result = train_model(
        cast(Any, supervised),
        models_root=_resolve_path(output if output is not None else config.models_dir, root),
        training_dataset_id=data.manifest.dataset_id,
        tag_dictionary_sha256=data.manifest.tag_dictionary_sha256,
        git_commit=_git_commit(root),
        source_timezone=config.source_timezone,
        horizon_minutes=config.horizon_minutes,
        seed=config.seed,
    )
    return {
        "model_id": result.bundle.metadata.model_id,
        "artifact_dir": result.artifact_dir.as_posix(),
        "selected_model": result.metrics.get("selected_model"),
        "target_signal": result.bundle.metadata.target_signal,
        "target_source": result.bundle.metadata.target_source,
    }


def evaluate_command(
    dataset: str | Path,
    model: str | Path,
    split: Literal["validation", "test"] = "test",
    output: str | Path | None = None,
    config_path: str | Path = "config/runtime.toml",
    root: Path = PROJECT_ROOT,
) -> dict[str, object]:
    """Evaluate a trusted local forecast artifact on a temporal split."""
    from source.ml.artifacts import load_model
    from source.ml.evaluate import evaluate_model, write_evaluation
    from source.ml.features import build_supervised_dataset

    config = load_runtime_config(_resolve_path(config_path, root))
    data = load_prepared_dataset(_resolve_path(dataset, root))
    bundle = load_model(
        _resolve_path(model, root),
        trusted=True,
        expected_horizon_minutes=config.horizon_minutes,
        expected_tag_dictionary_sha256=data.manifest.tag_dictionary_sha256,
    )
    supervised = build_supervised_dataset(
        data,
        target_signal_id=bundle.metadata.target_signal,
        target_source=SourceKind(bundle.metadata.target_source),
        horizon_minutes=bundle.metadata.horizon_minutes,
    )
    result = evaluate_model(
        cast(Any, supervised), bundle, split=split, source_timezone=config.source_timezone
    )
    output_root = _resolve_path(output if output is not None else config.reports_dir, root)
    report_dir = write_evaluation(output_root / f"{bundle.metadata.model_id}-{split}", result)
    return {
        "model_id": bundle.metadata.model_id,
        "split": split,
        "report_dir": report_dir.as_posix(),
        "metrics": result.report["metrics"],
    }


def _scenario_target_signal(scenario: ScenarioConfig) -> str | None:
    """Return the single forecast target expected by the current history scenario."""
    return scenario.required_signals[0] if len(scenario.required_signals) == 1 else None


def _scenario_target_unit(scenario: ScenarioConfig) -> str | None:
    """Return the sulfur unit that the history artifact must serve, when configured."""
    sulfur_constraints = [item for item in scenario.constraints if item.metric == "sulfur"]
    return sulfur_constraints[0].unit if sulfur_constraints else None


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
    """Run the history serving path with an explicitly trusted local model artifact."""
    from source.ml.artifacts import load_model
    from source.orchestrator import run_cycle

    config = load_runtime_config(_resolve_path(config_path, root))
    scenario_path = Path(scenario)
    if not scenario_path.suffix:
        scenario_path = Path("config/scenarios") / f"{scenario}.json"
    scenario_config = load_scenario(_resolve_path(scenario_path, root))
    data = load_prepared_dataset(_resolve_path(dataset, root))
    model_bundle = load_model(
        _resolve_path(model, root),
        trusted=trusted_model,
        expected_horizon_minutes=config.horizon_minutes,
        expected_tag_dictionary_sha256=data.manifest.tag_dictionary_sha256,
        expected_target_signal=_scenario_target_signal(scenario_config),
        expected_target_unit=_scenario_target_unit(scenario_config),
    )
    target_run_dir = _resolve_path(run_dir if run_dir is not None else config.runs_dir, root)
    return run_cycle(
        data=data,
        as_of=as_of,
        model=model_bundle,
        scenario=scenario_config,
        config=config,
        context=DecisionContext(),
        run_dir=target_run_dir,
    )


def accept_stage6(
    root: Path = PROJECT_ROOT,
    run_dir: str | Path | None = None,
    expected_statuses: Mapping[str, str] = STAGE6_EXPECTED_STATUSES,
) -> dict[str, object]:
    """Run the Stage-6 reproducibility and demonstration acceptance checks."""
    try:
        import sklearn

        sklearn_version = sklearn.__version__
    except ImportError:
        sklearn_version = "unavailable"
    acceptance_run_dir = _resolve_path(run_dir if run_dir is not None else "runs/stage6", root)
    issues: list[str] = []
    validation: dict[str, object] = {}
    try:
        validation.update(validate_stage0(root))
    except Exception as exc:
        validation = {"error": str(exc)}
        issues.append(f"validate_stage0 failed: {exc}")

    scenario_results: list[dict[str, object]] = []
    for scenario_id in STAGE6_SCENARIOS:
        expected = expected_statuses[scenario_id]
        try:
            result = run_model_demo(scenario_id, root=root, run_dir=acceptance_run_dir)
            journal_dir = acceptance_run_dir / result.run_id
            missing = [name for name in STAGE6_JOURNAL_FILES if not (journal_dir / name).is_file()]
            status = result.status.value
            passed = status == expected and not missing
            if status != expected:
                issues.append(f"{scenario_id}: expected status {expected!r}, received {status!r}")
            if missing:
                issues.append(f"{scenario_id}: missing journal files {', '.join(missing)}")
            scenario_results.append(
                {
                    "scenario_id": scenario_id,
                    "expected_status": expected,
                    "status": status,
                    "passed": passed,
                    "run_id": result.run_id,
                    "journal_dir": journal_dir.as_posix(),
                    "journal_files": {
                        name: (journal_dir / name).as_posix() for name in STAGE6_JOURNAL_FILES
                    },
                    "missing_journal_files": missing,
                }
            )
        except Exception as exc:
            issues.append(f"{scenario_id}: {exc}")
            scenario_results.append(
                {
                    "scenario_id": scenario_id,
                    "expected_status": expected,
                    "status": None,
                    "passed": False,
                    "error": str(exc),
                }
            )

    passed = not issues and all(bool(item.get("passed")) for item in scenario_results)
    return {
        "passed": passed,
        "validation": validation,
        "scenarios": scenario_results,
        "environment": {
            "python_version": platform.python_version(),
            "sklearn_version": sklearn_version,
        },
        "limitations": list(STAGE6_LIMITATIONS),
        "issues": issues,
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
        choices=("blend_normal", "blend_risk", "blend_missing"),
        help="scenario id from config/scenarios",
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
    train = subparsers.add_parser("train", help="train a local forecast artifact")
    train.add_argument("--dataset", required=True, help="prepared dataset directory")
    train.add_argument(
        "--target-source",
        required=True,
        choices=(SourceKind.PAK.value, SourceKind.LIMS.value),
        help="quality source used as supervised target",
    )
    train.add_argument("--target-signal", default="ht:2:Mg.Sulfur", help="target signal id")
    train.add_argument("--config", default="config/runtime.toml", help="runtime config path")
    train.add_argument("--output", default=None, help="model artifact root")
    evaluate = subparsers.add_parser("evaluate", help="evaluate a local forecast artifact")
    evaluate.add_argument("--dataset", required=True, help="prepared dataset directory")
    evaluate.add_argument("--model", required=True, help="local model artifact directory")
    evaluate.add_argument(
        "--split", choices=("validation", "test"), default="test", help="temporal split"
    )
    evaluate.add_argument("--config", default="config/runtime.toml", help="runtime config path")
    evaluate.add_argument("--output", default=None, help="report root")
    accept = subparsers.add_parser(
        "accept-stage6", help="run Stage-6 acceptance and demonstration checks"
    )
    accept.add_argument(
        "--run-dir",
        default="runs/stage6",
        help="directory for acceptance journals, relative to project root unless absolute",
    )
    history = subparsers.add_parser(
        "run-history", help="run history mode with a trusted local forecast artifact"
    )
    history.add_argument("--dataset", required=True, help="prepared dataset directory")
    history.add_argument("--model", required=True, help="local model artifact directory")
    history.add_argument(
        "--trusted-model",
        action="store_true",
        help="allow loading the local joblib artifact after path and metadata checks",
    )
    history.add_argument(
        "--as-of", required=True, type=_parse_as_of, help="timezone-aware ISO datetime"
    )
    history.add_argument("--scenario", default="history", help="scenario id or JSON path")
    history.add_argument("--config", default="config/runtime.toml", help="runtime config path")
    history.add_argument("--run-dir", default=None, help="journal directory override")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-stage0":
            validation_result = validate_stage0()
            print(json.dumps(validation_result, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "run-model-demo":
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
                args.target_source,
                args.output,
                target_signal=args.target_signal,
                config_path=args.config,
            )
            print(json.dumps(training_result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "evaluate":
            evaluation_result = evaluate_command(
                args.dataset,
                args.model,
                split=cast(Literal["validation", "test"], args.split),
                output=args.output,
                config_path=args.config,
            )
            print(json.dumps(evaluation_result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "accept-stage6":
            acceptance_result = accept_stage6(run_dir=args.run_dir)
            print(json.dumps(acceptance_result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if acceptance_result["passed"] else 1
        if args.command == "run-history":
            recommendation = run_history_command(
                args.dataset,
                args.model,
                args.as_of,
                trusted_model=args.trusted_model,
                scenario=args.scenario,
                config_path=args.config,
                run_dir=args.run_dir,
            )
            print(recommendation.model_dump_json(indent=2))
            return 0
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
