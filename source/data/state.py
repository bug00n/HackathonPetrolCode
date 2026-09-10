"""Build a leakage-safe state from prepared observations."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pandas as pd

from source.contracts import (
    Issue,
    Observation,
    ProcessState,
    RuntimeConfig,
    ScenarioConfig,
    Severity,
    SignalSnapshot,
    SourceKind,
    Validity,
)
from source.data.prepare import PreparedData

_SOURCE_PRIORITY = {
    SourceKind.LIMS: 0,
    SourceKind.PAK: 1,
    SourceKind.VAK: 2,
    SourceKind.TELEMETRY: 3,
    SourceKind.SCENARIO: 4,
}


def _observation(row: pd.Series) -> Observation:
    """Convert one prepared quality-table row into a validated observation."""
    return Observation(
        id=str(row["observation_id"]),
        signal_id=str(row["signal_id"]),
        stage=row["stage"],
        source=row["source"],
        measured_at=row["measured_at"],
        available_at=row["available_at"],
        value=row["value"] if pd.notna(row["value"]) else None,
        unit=str(row["unit"]),
        validity=row["validity"],
        source_ref=str(row["source_ref"]),
    )


def build_state(
    data: PreparedData,
    as_of: datetime,
    scenario: ScenarioConfig,
    config: RuntimeConfig,
) -> ProcessState:
    """Select only observations measured and available by ``as_of``."""
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    as_of = as_of.astimezone(UTC)
    frame = data.quality.copy()
    frame["measured_at"] = pd.to_datetime(frame["measured_at"], utc=True)
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    visible = frame[(frame["measured_at"] <= as_of) & (frame["available_at"] <= as_of)]

    snapshots: dict[str, SignalSnapshot] = {}
    state_issues: list[Issue] = []
    for signal_id in scenario.required_signals:
        candidates = [
            _observation(row) for _, row in visible[visible["signal_id"] == signal_id].iterrows()
        ]
        candidates.sort(
            key=lambda item: (item.measured_at, -_SOURCE_PRIORITY[item.source]), reverse=True
        )
        usable = [
            item
            for item in candidates
            if item.validity is Validity.VALID and item.value is not None
        ]
        selected = usable[0] if usable else None
        issues: list[Issue] = []
        if selected is None:
            issues.append(
                Issue(
                    code="MISSING_REQUIRED_SIGNAL",
                    severity=Severity.BLOCKING,
                    signal_id=signal_id,
                    detail="no valid observation was available at as_of",
                    source_ref=None,
                )
            )
            age_seconds = None
            fresh = False
        else:
            age_seconds = (as_of - selected.measured_at).total_seconds()
            limit_seconds = config.freshness_minutes.get(selected.source, 0) * 60
            fresh = age_seconds <= limit_seconds
            if not fresh:
                issues.append(
                    Issue(
                        code="STALE_REQUIRED_SIGNAL",
                        severity=Severity.BLOCKING,
                        signal_id=signal_id,
                        detail=f"age {age_seconds:.0f}s exceeds {limit_seconds}s",
                        source_ref=selected.source_ref,
                    )
                )
        snapshots[signal_id] = SignalSnapshot(
            selected=selected,
            alternatives=tuple(item for item in candidates if item != selected),
            age_seconds=age_seconds,
            fresh=fresh,
            issues=tuple(issues),
        )
        state_issues.extend(issues)

    payload = {
        "schema_version": "1.0",
        "as_of": as_of.isoformat(),
        "dataset_id": data.manifest.dataset_id,
        "mode": scenario.mode.value,
        "signals": {key: value.model_dump(mode="json") for key, value in sorted(snapshots.items())},
        "issues": [item.model_dump(mode="json") for item in state_issues],
    }
    state_id = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return ProcessState(state_id=state_id, **payload)


__all__ = ["build_state"]
