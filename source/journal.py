"""Decision-run journal writer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd
from pydantic import BaseModel

from source.contracts import (
    CandidateEvaluation,
    DecisionContext,
    ProcessState,
    Recommendation,
    ScenarioConfig,
)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    temporary.replace(path)


def _json_line(payload: object) -> str:
    if isinstance(payload, BaseModel):
        return payload.model_dump_json() + "\n"
    return json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"


def write_run_journal(
    run_dir: Path,
    state: ProcessState,
    scenario: ScenarioConfig,
    context: DecisionContext,
    candidates: Iterable[CandidateEvaluation],
    result: Recommendation,
    *,
    selection_reason: str = "unknown",
    rejection_summary: dict[str, int] | None = None,
    features: pd.DataFrame | None = None,
) -> Path:
    """Write a compact reproducible record of one successful run."""
    rejection_summary = rejection_summary or {}
    target = run_dir / result.run_id
    target.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        target / "metadata.json",
        json.dumps(
            {
                "schema_version": result.schema_version,
                "run_id": result.run_id,
                "dataset_id": state.dataset_id,
                "scenario_id": scenario.id,
                "model_id": result.model_id,
                "selection_reason": selection_reason,
                "rejection_summary": rejection_summary,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
    )
    _atomic_write(
        target / "input.json",
        json.dumps(
            {
                "state": state.model_dump(mode="json"),
                "scenario": scenario.model_dump(mode="json"),
                "context": context.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
    )
    feature_records: list[dict[str, object]] = []
    if features is not None:
        feature_records = json.loads(features.to_json(orient="records", date_format="iso"))
    _atomic_write(
        target / "features.json",
        json.dumps({"features": feature_records}, ensure_ascii=False, indent=2, allow_nan=False),
    )
    _atomic_write(
        target / "trace.jsonl",
        _json_line(
            {
                "event": "run_cycle_completed",
                "status": result.status.value,
                "selection_reason": selection_reason,
                "rejection_summary": rejection_summary,
            }
        ),
    )
    _atomic_write(target / "candidates.jsonl", "".join(_json_line(item) for item in candidates))
    _atomic_write(target / "result.json", result.model_dump_json(indent=2))
    return target


__all__ = ["write_run_journal"]
