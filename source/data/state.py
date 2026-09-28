"""Build a leakage-safe state from prepared observations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
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
    Stage,
    Unit,
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


def _stage_from_signal_id(signal_id: str) -> Stage:
    if signal_id.startswith("avt:"):
        return Stage.AVT
    if signal_id.startswith("ht:"):
        return Stage.HYDROTREATMENT
    return Stage.HYDROTREATMENT


def _utc_frame(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Normalize timestamp columns once; reuse frames already normalized to UTC."""
    for column in columns:
        dtype = frame[column].dtype
        if not isinstance(dtype, pd.DatetimeTZDtype) or dtype.tz != UTC:
            break
    else:
        return frame
    normalized = frame.copy()
    for column in columns:
        normalized[column] = pd.to_datetime(normalized[column], utc=True)
    return normalized


def prepare_state_data(data: PreparedData) -> PreparedData:
    """Return a per-request dataset with UTC timestamps for repeated state builds."""
    quality = _utc_frame(data.quality, ("measured_at", "available_at"))
    telemetry = (
        _utc_frame(data.telemetry, ("timestamp",))
        if "timestamp" in data.telemetry.columns
        else data.telemetry
    )
    return replace(data, quality=quality, telemetry=telemetry)


def _telemetry_candidates(
    frame: pd.DataFrame,
    signal_id: str,
    as_of: datetime,
    unit: str,
    dataset_id: str,
) -> list[Observation]:
    if signal_id not in frame.columns or "timestamp" not in frame.columns:
        return []
    visible = frame[frame["timestamp"] <= pd.Timestamp(as_of)]
    values = pd.to_numeric(visible[signal_id], errors="coerce")
    valid = visible[values.notna()].copy()
    if valid.empty:
        return []
    row = valid.iloc[-1]
    measured_at = pd.Timestamp(row["timestamp"]).to_pydatetime()
    return [
        Observation(
            id=f"telemetry:{signal_id}:{measured_at.isoformat()}",
            signal_id=signal_id,
            stage=_stage_from_signal_id(signal_id),
            source=SourceKind.TELEMETRY,
            measured_at=measured_at,
            available_at=measured_at,
            value=float(row[signal_id]),
            unit=unit,
            validity=Validity.VALID,
            source_ref=f"prepared:{dataset_id}:telemetry.csv.gz",
        )
    ]


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
    prepared = prepare_state_data(data)
    frame = prepared.quality
    visible = frame[(frame["measured_at"] <= as_of) & (frame["available_at"] <= as_of)]
    telemetry = prepared.telemetry
    control_units = {control.signal_id: control.unit for control in scenario.controls}

    snapshots: dict[str, SignalSnapshot] = {}
    state_issues: list[Issue] = []
    for signal_id in scenario.required_signals:
        signal_rows = visible[visible["signal_id"] == signal_id].sort_values(
            "measured_at", ascending=False, kind="stable"
        )
        # A state is a snapshot, not a second copy of the historical dataset.
        # Keep the latest reading and latest usable reading of each source:
        # an invalid new sample must not hide an older valid observation.
        usable_rows = signal_rows[
            signal_rows["validity"].eq(Validity.VALID.value) & signal_rows["value"].notna()
        ]
        snapshot_rows = pd.concat(
            [
                signal_rows.groupby("source", sort=False).head(1),
                usable_rows.groupby("source", sort=False).head(1),
            ]
        ).drop_duplicates("observation_id")
        candidates = [_observation(row) for _, row in snapshot_rows.iterrows()]
        candidates.extend(
            _telemetry_candidates(
                telemetry,
                signal_id,
                as_of,
                control_units.get(signal_id, Unit.UNKNOWN.value),
                data.manifest.dataset_id,
            )
        )
        usable = [
            item
            for item in candidates
            if item.validity is Validity.VALID and item.value is not None
        ]

        def is_fresh(item: Observation) -> bool:
            age_seconds = (as_of - item.measured_at).total_seconds()
            return age_seconds <= config.freshness_minutes.get(item.source, 0) * 60

        fresh_candidates = [item for item in usable if is_fresh(item)]
        pool = fresh_candidates or usable
        selected = (
            min(
                pool,
                key=lambda item: (_SOURCE_PRIORITY[item.source], -item.measured_at.timestamp()),
            )
            if pool
            else None
        )
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
            selected_is_fresh = False
        else:
            age_seconds = (as_of - selected.measured_at).total_seconds()
            limit_seconds = config.freshness_minutes.get(selected.source, 0) * 60
            selected_is_fresh = age_seconds <= limit_seconds
            if not selected_is_fresh:
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
            fresh=selected_is_fresh,
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
