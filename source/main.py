"""Small CLI surface implemented for the current backend stages."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from source.config import load_runtime_config, load_scenario, load_tag_dictionary
from source.contracts import DecisionContext, ProcessState, Recommendation, RecommendationStatus
from source.data import build_state, load_prepared_dataset, prepare_dataset, write_prepared_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE6_SCENARIOS: tuple[str, ...] = ("blend_normal", "blend_risk", "blend_missing")
STAGE6_EXPECTED_STATUSES: dict[str, str] = {
    scenario_id: RecommendationStatus.ABSTAIN.value for scenario_id in STAGE6_SCENARIOS
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
    "The desktop UI still demonstrates model-demo scenarios; "
    "history artifact serving is next work.",
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
    accept = subparsers.add_parser(
        "accept-stage6", help="run Stage-6 acceptance and demonstration checks"
    )
    accept.add_argument(
        "--run-dir",
        default="runs/stage6",
        help="directory for acceptance journals, relative to project root unless absolute",
    )
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
        if args.command == "accept-stage6":
            acceptance_result = accept_stage6(run_dir=args.run_dir)
            print(json.dumps(acceptance_result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if acceptance_result["passed"] else 1
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
