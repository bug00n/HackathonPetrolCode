"""Small CLI surface implemented at stage 0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from source.config import load_runtime_config, load_scenario, load_tag_dictionary
from source.contracts import ProcessState, Recommendation

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


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the CLI command and run the requested stage-0 validation."""
    parser = argparse.ArgumentParser(description="Нефтекод recommendation prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-stage0", help="validate contracts, configs and fixtures")
    args = parser.parse_args(argv)
    if args.command == "validate-stage0":
        result = validate_stage0()
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
