"""Headless UI projections for stage dashboards and history replay.

The Tkinter shell imports this module for display-ready data.  Keeping the
queries here makes AVT/hydrotreatment/history screens testable without opening
Tk and avoids importing optional ML stacks during ordinary UI startup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from source.config import PROJECT_ROOT, load_runtime_config, load_scenario, load_tag_dictionary
from source.contracts import (
    CandidateEvaluation,
    ConstraintStatus,
    EstimateBasis,
    IntervalKind,
    MetricEstimate,
    Recommendation,
    RecommendationStatus,
    SourceKind,
    TagMeta,
    Unit,
)
from source.data.prepare import PreparedData, load_prepared_dataset
from source.main import replay_command
from source.ml.blending import (
    HybridComponentForecast,
    apply_hydrotreater_forecast,
    calculate_mass_blend,
    sulfur_constraint_status,
)

AVT_SIGNAL_GROUPS: dict[str, tuple[str, ...]] = {
    "K-2 state": ("avt:F65", "avt:T20", "avt:T33", "avt:P21", "avt:P22", "avt:P23", "avt:P67"),
    "K-2 circulation": (
        "avt:F14",
        "avt:T13",
        "avt:T18",
        "avt:F12",
        "avt:T17",
        "avt:F64",
        "avt:T11",
        "avt:T15",
    ),
    "Diesel cut": ("avt:T66", "avt:F28", "avt:F32", "avt:T71", "avt:F30", "avt:W70"),
}

HYDROTREATING_SIGNAL_GROUPS: dict[str, tuple[str, ...]] = {
    "Quality": ("ht:2:Mg.Sulfur", "ht:density_15c"),
    "Action readiness": ("ht:P8", "ht:F19"),
    "Process context": ("ht:T11", "ht:F26", "ht:F14", "ht:F15", "ht:F17"),
    "Gas context": ("ht:F2", "ht:F22", "ht:F25"),
}

ACTION_CONTROL_IDS = {"ht:P8", "ht:F19"}
CONTEXT_ONLY_IDS = {"ht:T11", "ht:F26", "ht:F2", "ht:F22", "ht:F25"}
HISTORY_SULFUR_SIGNAL_ID = "ht:2:Mg.Sulfur"
SULFUR_LIMIT_MG_KG = 10.0

_SOURCE_PRIORITY = {
    "lims": 0,
    "pak": 1,
    "vak": 2,
    "telemetry": 3,
    "scenario": 4,
}


@dataclass(frozen=True)
class UiArtifactOption:
    path: str
    model_id: str
    schema_version: str
    artifact_kind: str
    supports_actions: bool
    supports_multi_horizon: bool


@dataclass(frozen=True)
class UiSignalRow:
    group: str
    signal_id: str
    label: str
    value: float | None
    value_text: str
    unit: str
    source: str
    measured_at: str | None
    available_at: str | None
    age_minutes: float | None
    freshness: Literal["fresh", "stale", "missing"]
    issue: str
    evidence_ref: str
    read_only_reason: str


@dataclass(frozen=True)
class UiStageSnapshot:
    page: str
    title: str
    dataset_path: str | None
    dataset_id: str | None
    as_of: str | None
    status: Literal["ready", "empty", "error"]
    message: str
    fresh_count: int
    stale_count: int
    missing_count: int
    rows: tuple[UiSignalRow, ...]


@dataclass(frozen=True)
class UiHistoryReplayView:
    status: Literal["ready", "empty", "error"]
    message: str
    recommendation_status: str | None
    scenario_id: str | None
    model_id: str | None
    as_of: str | None
    sulfur_point: float | None
    sulfur_upper: float | None
    upper_status: Literal["pass", "fail", "unknown"]
    selected_kind: str | None
    action_state: Literal["not_actionable", "actionable", "unavailable"]
    reason_codes: tuple[str, ...]
    issues: tuple[str, ...]
    journal_path: str | None
    raw: dict[str, Any] | None


@dataclass(frozen=True)
class UiHybridBlendView:
    status: Literal["ready", "empty", "error"]
    message: str
    scenario_id: str
    source_state_id: str | None
    model_id: str | None
    component_sulfur_point: float | None
    component_sulfur_upper: float | None
    blend_sulfur_point: float | None
    blend_sulfur_upper: float | None
    constraint_status: str
    assumptions: tuple[str, ...]
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class UiContext:
    latest_dataset: str | None
    forecast_artifacts: tuple[UiArtifactOption, ...]
    action_artifacts: tuple[UiArtifactOption, ...]
    v2_artifacts: tuple[UiArtifactOption, ...]


def list_prepared_datasets(root: Path = PROJECT_ROOT) -> tuple[Path, ...]:
    """Return prepared dataset directories sorted by modification time."""
    config = load_runtime_config(root / "config/runtime.toml")
    data_root = _resolve(config.data_dir, root)
    manifests = [path for path in data_root.glob("*/manifest.json") if path.is_file()]
    return tuple(path.parent for path in sorted(manifests, key=lambda item: item.stat().st_mtime))


def latest_prepared_dataset(root: Path = PROJECT_ROOT) -> Path | None:
    datasets = list_prepared_datasets(root)
    return datasets[-1] if datasets else None


def list_model_artifacts(root: Path = PROJECT_ROOT) -> tuple[UiArtifactOption, ...]:
    """Return model metadata summaries without loading executable artifacts."""
    config = load_runtime_config(root / "config/runtime.toml")
    models_root = _resolve(config.models_dir, root)
    result: list[UiArtifactOption] = []
    for metadata_path in sorted(models_root.glob("*/metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        capabilities = metadata.get("capabilities")
        if not isinstance(capabilities, dict):
            capabilities = {}
        artifact_kind = str(metadata.get("artifact_kind", "forecast"))
        model_id = str(metadata.get("model_id") or metadata_path.parent.name)
        result.append(
            UiArtifactOption(
                path=str(metadata_path.parent),
                model_id=model_id,
                schema_version=str(metadata.get("schema_version", "")),
                artifact_kind=artifact_kind,
                supports_actions=bool(
                    metadata.get("supports_actions") is True
                    or capabilities.get("supports_actions") is True
                ),
                supports_multi_horizon=bool(capabilities.get("supports_multi_horizon")),
            )
        )
    return tuple(result)


def discover_ui_context(root: Path = PROJECT_ROOT) -> UiContext:
    artifacts = list_model_artifacts(root)
    forecast = tuple(
        item
        for item in artifacts
        if item.schema_version == "1.0"
        and not item.supports_multi_horizon
        and item.artifact_kind != "action_effect"
    )
    action = tuple(
        item
        for item in artifacts
        if item.artifact_kind == "action_effect" and item.supports_actions
    )
    v2 = tuple(
        item for item in artifacts if item.schema_version == "1.2" and item.supports_multi_horizon
    )
    latest = latest_prepared_dataset(root)
    return UiContext(
        latest_dataset=str(latest) if latest is not None else None,
        forecast_artifacts=forecast,
        action_artifacts=action,
        v2_artifacts=v2,
    )


def default_as_of_for_dataset(dataset: str | Path | None, root: Path = PROJECT_ROOT) -> str:
    """Return an ISO timestamp inside the available prepared data when possible."""
    data_path = _dataset_path(dataset, root)
    if data_path is None:
        return datetime.now(UTC).isoformat()
    try:
        data = load_prepared_dataset(data_path)
        return _default_as_of(data).isoformat()
    except Exception:
        return datetime.now(UTC).isoformat()


def ui_stage_snapshot(
    page: str,
    dataset: str | Path | None = None,
    as_of: datetime | str | None = None,
    *,
    root: Path = PROJECT_ROOT,
) -> UiStageSnapshot:
    """Build a read-only AVT or hydrotreating dashboard snapshot."""
    canonical_page = _canonical_stage_page(page)
    title = "АВТ" if canonical_page == "avt" else "Гидроочистка"
    data_path = _dataset_path(dataset, root)
    if data_path is None:
        return UiStageSnapshot(
            page=canonical_page,
            title=title,
            dataset_path=None,
            dataset_id=None,
            as_of=None,
            status="empty",
            message=(
                "Prepared dataset is not available. Run: python -m source.main prepare "
                "--materials materials --config config/runtime.toml"
            ),
            fresh_count=0,
            stale_count=0,
            missing_count=0,
            rows=(),
        )
    try:
        config = load_runtime_config(root / "config/runtime.toml")
        data = load_prepared_dataset(data_path)
        timestamp = _parse_as_of(as_of) if as_of is not None else _default_as_of(data)
        tags = _tags_by_signal_id(root)
        groups = AVT_SIGNAL_GROUPS if canonical_page == "avt" else HYDROTREATING_SIGNAL_GROUPS
        rows = tuple(
            _signal_row(data, timestamp, config, tags, group, signal_id)
            for group, signal_ids in groups.items()
            for signal_id in signal_ids
        )
    except Exception as exc:
        return UiStageSnapshot(
            page=canonical_page,
            title=title,
            dataset_path=str(data_path),
            dataset_id=None,
            as_of=None,
            status="error",
            message=str(exc),
            fresh_count=0,
            stale_count=0,
            missing_count=0,
            rows=(),
        )
    fresh = sum(row.freshness == "fresh" for row in rows)
    stale = sum(row.freshness == "stale" for row in rows)
    missing = sum(row.freshness == "missing" for row in rows)
    return UiStageSnapshot(
        page=canonical_page,
        title=title,
        dataset_path=str(data_path),
        dataset_id=data.manifest.dataset_id,
        as_of=timestamp.isoformat(),
        status="ready",
        message=(
            "Read-only process context. Real setpoint changes require a verified action artifact."
        ),
        fresh_count=fresh,
        stale_count=stale,
        missing_count=missing,
        rows=rows,
    )


def ui_history_snapshot(
    dataset: str | Path | None,
    model: str | Path | None,
    as_of: datetime | str,
    action_model: str | Path | None = None,
    *,
    root: Path = PROJECT_ROOT,
) -> UiHistoryReplayView:
    """Replay history and convert the recommendation into UI-facing fields."""
    if not dataset or not model:
        return UiHistoryReplayView(
            status="empty",
            message="Prepared dataset and forecast artifact are required for history replay.",
            recommendation_status=None,
            scenario_id=None,
            model_id=None,
            as_of=None,
            sulfur_point=None,
            sulfur_upper=None,
            upper_status="unknown",
            selected_kind=None,
            action_state="unavailable",
            reason_codes=(),
            issues=(),
            journal_path=None,
            raw=None,
        )
    try:
        timestamp = _parse_as_of(as_of)
        result = replay_command(
            dataset,
            model,
            "history",
            timestamp,
            action_model_path=str(action_model) if action_model else None,
            root=root,
        )
        config = load_runtime_config(root / "config/runtime.toml")
        journal = root / config.runs_dir / result.run_id / "result.json"
        return history_replay_to_view(result, journal if journal.is_file() else None)
    except Exception as exc:
        return UiHistoryReplayView(
            status="error",
            message=str(exc),
            recommendation_status=None,
            scenario_id=None,
            model_id=None,
            as_of=_parse_as_of(as_of).isoformat() if as_of else None,
            sulfur_point=None,
            sulfur_upper=None,
            upper_status="unknown",
            selected_kind=None,
            action_state="unavailable",
            reason_codes=(),
            issues=(str(exc),),
            journal_path=None,
            raw=None,
        )


def history_replay_to_view(
    result: Recommendation, journal_path: str | Path | None = None
) -> UiHistoryReplayView:
    """Project a strict Recommendation into a compact history replay view."""
    carrier = result.selected or result.baseline
    sulfur_point, _, sulfur_upper = _metric(carrier, "sulfur")
    issues = tuple(_evaluation_issues(carrier))
    if sulfur_upper is None:
        upper_status: Literal["pass", "fail", "unknown"] = "unknown"
    else:
        upper_status = "pass" if sulfur_upper <= SULFUR_LIMIT_MG_KG else "fail"
    selected_kind = result.selected.candidate.kind.value if result.selected else None
    action_state: Literal["not_actionable", "actionable", "unavailable"]
    if result.status is RecommendationStatus.RECOMMEND and selected_kind == "setpoints":
        action_state = "actionable"
    elif "ACTION_MODEL_UNAVAILABLE" in result.reason_codes:
        action_state = "not_actionable"
    else:
        action_state = (
            "unavailable" if result.status is RecommendationStatus.ABSTAIN else "not_actionable"
        )
    return UiHistoryReplayView(
        status="ready",
        message=result.explanation,
        recommendation_status=result.status.value,
        scenario_id=result.scenario_id,
        model_id=result.model_id,
        as_of=result.as_of.isoformat(),
        sulfur_point=sulfur_point,
        sulfur_upper=sulfur_upper,
        upper_status=upper_status,
        selected_kind=selected_kind,
        action_state=action_state,
        reason_codes=tuple(result.reason_codes),
        issues=issues,
        journal_path=str(journal_path) if journal_path is not None else None,
        raw=result.model_dump(mode="json"),
    )


def ui_hybrid_snapshot(
    dataset: str | Path | None,
    model: str | Path | None,
    as_of: datetime | str,
    *,
    root: Path = PROJECT_ROOT,
) -> UiHybridBlendView:
    """Use a history sulfur forecast as component A in the hybrid blend scenario."""
    if not dataset or not model:
        return UiHybridBlendView(
            status="empty",
            message="Hybrid needs a prepared dataset and a trusted sulfur forecast artifact.",
            scenario_id="hybrid_blend",
            source_state_id=None,
            model_id=None,
            component_sulfur_point=None,
            component_sulfur_upper=None,
            blend_sulfur_point=None,
            blend_sulfur_upper=None,
            constraint_status="unknown",
            assumptions=(),
            reason_codes=("FORECAST_UNAVAILABLE",),
        )
    try:
        timestamp = _parse_as_of(as_of)
        result = replay_command(dataset, model, "history", timestamp, root=root)
        carrier = result.baseline
        sulfur_point, _, sulfur_upper = _metric(carrier, "sulfur")
        if sulfur_point is None and sulfur_upper is None:
            return UiHybridBlendView(
                status="empty",
                message="History replay did not produce a sulfur forecast for component A.",
                scenario_id="hybrid_blend",
                source_state_id=result.state_id,
                model_id=result.model_id,
                component_sulfur_point=None,
                component_sulfur_upper=None,
                blend_sulfur_point=None,
                blend_sulfur_upper=None,
                constraint_status="unknown",
                assumptions=tuple(result.assumptions),
                reason_codes=tuple(result.reason_codes) + ("FORECAST_UNAVAILABLE",),
            )
        scenario = load_scenario(root / "config/scenarios/hybrid_blend.json")
        estimate = MetricEstimate(
            value=sulfur_point,
            lower=None,
            upper=sulfur_upper,
            unit=Unit.MG_KG.value,
            basis=EstimateBasis.FORECAST,
            interval_kind=(
                IntervalKind.EMPIRICAL if sulfur_upper is not None else IntervalKind.NONE
            ),
            interval_level=0.95 if sulfur_upper is not None else None,
            reference=f"history replay {result.run_id}",
            assumptions=("Component A is populated from a trusted history sulfur forecast.",),
        )
        forecast = HybridComponentForecast(
            component_id="A",
            sulfur=estimate,
            source_state_id=result.state_id,
            upstream_quality_reference=f"history:{result.state_id}",
            lag_min_minutes=0,
            lag_max_minutes=180,
            link_confirmed=False,
            evidence_ref="config/scenarios/hybrid_blend.json",
        )
        components = apply_hydrotreater_forecast(scenario.blend_components, forecast)
        if scenario.total_mass_t is None:
            raise ValueError("hybrid scenario needs a batch mass")
        blend = calculate_mass_blend(
            dict(scenario.current_blend_mass_fractions),
            components,
            scenario.total_mass_t,
            additive_mass_fraction=scenario.current_additive_mass_fraction,
            additive=scenario.cetane_additive,
        )
        status = sulfur_constraint_status(blend, SULFUR_LIMIT_MG_KG).value
        return UiHybridBlendView(
            status="ready",
            message=(
                "Hybrid is sulfur-only and keeps AVT-to-hydrotreatment linkage as an "
                "explicit assumption."
            ),
            scenario_id=scenario.id,
            source_state_id=result.state_id,
            model_id=result.model_id,
            component_sulfur_point=sulfur_point,
            component_sulfur_upper=sulfur_upper,
            blend_sulfur_point=blend.sulfur.value,
            blend_sulfur_upper=blend.sulfur.upper,
            constraint_status=status,
            assumptions=tuple(dict.fromkeys((*scenario.assumptions, *blend.assumptions))),
            reason_codes=tuple(result.reason_codes),
        )
    except Exception as exc:
        return UiHybridBlendView(
            status="error",
            message=str(exc),
            scenario_id="hybrid_blend",
            source_state_id=None,
            model_id=None,
            component_sulfur_point=None,
            component_sulfur_upper=None,
            blend_sulfur_point=None,
            blend_sulfur_upper=None,
            constraint_status="unknown",
            assumptions=(),
            reason_codes=(type(exc).__name__,),
        )


def _canonical_stage_page(page: str) -> Literal["avt", "hydrotreating"]:
    normalized = page.strip().lower()
    if normalized in {"avt", "авт"}:
        return "avt"
    if normalized in {"hydrotreating", "hydro", "ht", "гидроочистка"}:
        return "hydrotreating"
    raise ValueError(f"unsupported UI stage page: {page}")


def _resolve(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _dataset_path(dataset: str | Path | None, root: Path) -> Path | None:
    if dataset:
        return _resolve(dataset, root)
    return latest_prepared_dataset(root)


def _tags_by_signal_id(root: Path) -> dict[str, TagMeta]:
    config = load_runtime_config(root / "config/runtime.toml")
    tags = load_tag_dictionary(_resolve(config.tag_dictionary_path, root))
    result: dict[str, TagMeta] = {}
    for tag in tags.values():
        result.setdefault(tag.signal_id, tag)
    return result


def _parse_as_of(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        timestamp = value
    else:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("as_of must include timezone")
    return timestamp.astimezone(UTC)


def _default_as_of(data: PreparedData) -> datetime:
    candidates: list[pd.Timestamp] = []
    if "timestamp" in data.telemetry.columns:
        timestamps = pd.to_datetime(data.telemetry["timestamp"], utc=True, errors="coerce")
        if timestamps.notna().any():
            candidates.append(timestamps.max())
    if "available_at" in data.quality.columns:
        available = pd.to_datetime(data.quality["available_at"], utc=True, errors="coerce")
        if available.notna().any():
            candidates.append(available.max())
    if not candidates:
        return datetime.now(UTC)
    return max(candidates).to_pydatetime()


def _signal_row(
    data: PreparedData,
    as_of: datetime,
    config: Any,
    tags: dict[str, TagMeta],
    group: str,
    signal_id: str,
) -> UiSignalRow:
    tag = tags.get(signal_id)
    unit = tag.canonical_unit if tag is not None else Unit.UNKNOWN.value
    selected = _select_observation(data, signal_id, as_of, unit)
    label = tag.meaning if tag is not None else signal_id
    evidence = tag.evidence_ref if tag is not None else ""
    read_only_reason = _read_only_reason(signal_id, unit)
    if selected is None:
        return UiSignalRow(
            group=group,
            signal_id=signal_id,
            label=label,
            value=None,
            value_text="—",
            unit=unit,
            source="—",
            measured_at=None,
            available_at=None,
            age_minutes=None,
            freshness="missing",
            issue="no valid observation at as_of",
            evidence_ref=evidence,
            read_only_reason=read_only_reason,
        )
    measured_at = selected["measured_at"]
    available_at = selected["available_at"]
    source = str(selected["source"])
    value = float(selected["value"])
    age_minutes = max(0.0, (as_of - measured_at).total_seconds() / 60.0)
    limit = _freshness_limit_minutes(config, source)
    fresh = age_minutes <= limit
    return UiSignalRow(
        group=group,
        signal_id=signal_id,
        label=label,
        value=value,
        value_text=_format_value(value, unit),
        unit=unit,
        source=source,
        measured_at=measured_at.isoformat(),
        available_at=available_at.isoformat(),
        age_minutes=age_minutes,
        freshness="fresh" if fresh else "stale",
        issue="" if fresh else f"age {age_minutes:.0f} min exceeds {limit:.0f} min",
        evidence_ref=evidence,
        read_only_reason=read_only_reason,
    )


def _select_observation(
    data: PreparedData, signal_id: str, as_of: datetime, unit: str
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    if signal_id in data.quality.get("signal_id", pd.Series(dtype=str)).astype(str).values:
        frame = data.quality.copy()
        frame["measured_at"] = pd.to_datetime(frame["measured_at"], utc=True, errors="coerce")
        frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")
        visible = frame[
            (frame["signal_id"].astype(str) == signal_id)
            & (frame["measured_at"] <= pd.Timestamp(as_of))
            & (frame["available_at"] <= pd.Timestamp(as_of))
            & (frame["validity"].astype(str) == "valid")
            & frame["value"].notna()
        ]
        for _, row in visible.iterrows():
            candidates.append(
                {
                    "source": str(row["source"]),
                    "value": float(row["value"]),
                    "unit": str(row["unit"]),
                    "measured_at": pd.Timestamp(row["measured_at"]).to_pydatetime(),
                    "available_at": pd.Timestamp(row["available_at"]).to_pydatetime(),
                }
            )
    if signal_id in data.telemetry.columns and "timestamp" in data.telemetry.columns:
        frame = data.telemetry.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        visible = frame[frame["timestamp"] <= pd.Timestamp(as_of)]
        values = pd.to_numeric(visible[signal_id], errors="coerce")
        valid = visible[values.notna()]
        if not valid.empty:
            row = valid.iloc[-1]
            measured = pd.Timestamp(row["timestamp"]).to_pydatetime()
            candidates.append(
                {
                    "source": SourceKind.TELEMETRY.value,
                    "value": float(row[signal_id]),
                    "unit": unit,
                    "measured_at": measured,
                    "available_at": measured,
                }
            )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            _SOURCE_PRIORITY.get(str(item["source"]), 99),
            -item["measured_at"].timestamp(),
        ),
    )


def _freshness_limit_minutes(config: Any, source: str) -> float:
    try:
        key = SourceKind(source)
    except ValueError:
        return 0.0
    return float(config.freshness_minutes.get(key, 0))


def _read_only_reason(signal_id: str, unit: str) -> str:
    if signal_id in ACTION_CONTROL_IDS:
        return (
            "candidate control; disabled until a verified action artifact supplies bounds and gates"
        )
    if signal_id in CONTEXT_ONLY_IDS:
        return "context-only signal; not an enabled action control"
    if unit == Unit.UNKNOWN.value:
        return "read-only context; unit is unknown and cannot define a control"
    return "read-only process context"


def _format_value(value: float | None, unit: str) -> str:
    if value is None:
        return "—"
    rendered = f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{rendered} {unit}".strip()


def _metric(
    evaluation: CandidateEvaluation | None, name: str
) -> tuple[float | None, float | None, float | None]:
    if evaluation is None:
        return None, None, None
    for assessment in evaluation.assessments:
        estimate = assessment.metrics.get(name)
        if estimate is not None:
            return estimate.value, estimate.lower, estimate.upper
    return None, None, None


def _evaluation_issues(evaluation: CandidateEvaluation | None) -> tuple[str, ...]:
    if evaluation is None:
        return ()
    result: list[str] = []
    for assessment in evaluation.assessments:
        result.extend(f"{issue.code}: {issue.detail}" for issue in assessment.issues)
    for check in evaluation.checks:
        if check.status is not ConstraintStatus.PASS:
            result.append(f"{check.reason_code}: {check.constraint_id}")
    return tuple(result)


__all__ = [
    "ACTION_CONTROL_IDS",
    "AVT_SIGNAL_GROUPS",
    "HYDROTREATING_SIGNAL_GROUPS",
    "UiArtifactOption",
    "UiContext",
    "UiHistoryReplayView",
    "UiHybridBlendView",
    "UiSignalRow",
    "UiStageSnapshot",
    "default_as_of_for_dataset",
    "discover_ui_context",
    "history_replay_to_view",
    "latest_prepared_dataset",
    "list_model_artifacts",
    "list_prepared_datasets",
    "ui_history_snapshot",
    "ui_hybrid_snapshot",
    "ui_stage_snapshot",
]
