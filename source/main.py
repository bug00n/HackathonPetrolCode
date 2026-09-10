"""Small CLI surface implemented at stage 0."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from source.config import load_runtime_config, load_scenario, load_tag_dictionary
from source.contracts import DecisionContext, ProcessState, Recommendation

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


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the CLI command and run the requested stage-0 validation."""
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
    args = parser.parse_args(argv)
    if args.command == "validate-stage0":
        summary = validate_stage0()
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "run-model-demo":
        recommendation = run_model_demo(args.scenario)
        print(recommendation.model_dump_json(indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
