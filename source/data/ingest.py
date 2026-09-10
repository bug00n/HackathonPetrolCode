"""Read supplied CSV/Excel sources without changing them."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pandas as pd

from source.contracts import (
    Issue,
    MappingStatus,
    Observation,
    Severity,
    SourceKind,
    Stage,
    TagMeta,
    Unit,
    Validity,
)
from source.data.prepare import canonical_column, resolve_unit, tag_stage, to_utc

NS_LIMS = "ЛИМС"
_ROW_SECTION = 0
_ROW_PARAMETER = 1
_ROW_UNIT = 2
_ROW_DATA_START = 4


@dataclass(frozen=True)
class TelemetryRead:
    frame: pd.DataFrame
    issues: tuple[Issue, ...]


@dataclass(frozen=True)
class QualityRead:
    observations: tuple[Observation, ...]
    issues: tuple[Issue, ...]


def _issue(code: str, detail: str, source_ref: str, signal_id: str | None = None) -> Issue:
    """Create a non-blocking ingestion issue with a common shape."""
    return Issue(
        code=code,
        severity=Severity.WARNING,
        signal_id=signal_id,
        detail=detail,
        source_ref=source_ref,
    )


def _observation_id(source_ref: str) -> str:
    """Create a deterministic identifier from the source location."""
    return str(uuid5(NAMESPACE_URL, source_ref))


def _is_service_column(name: str) -> bool:
    """Identify empty and pandas-generated index columns in raw CSV data."""
    return not name.strip() or name.strip().startswith("Unnamed:")


def read_telemetry_csv(
    path: str | Path,
    namespace: str,
    tags: dict[str, TagMeta],
    source_timezone: str = "Europe/Moscow",
    nrows: int | None = None,
) -> TelemetryRead:
    """Normalize one telemetry CSV to UTC and canonical signal columns."""
    path = Path(path)
    raw = pd.read_csv(path, nrows=nrows)
    service = [column for column in raw.columns if _is_service_column(str(column))]
    raw = raw.drop(columns=service)
    if "date" not in raw.columns:
        raise ValueError(f"{path}: required date column is missing")

    issues: list[Issue] = []
    parsed_time = pd.to_datetime(raw["date"], errors="coerce")
    bad_time = raw["date"].notna() & parsed_time.isna()
    for row_index in raw.index[bad_time]:
        source_ref = f"{path.as_posix()}#row={row_index + 2};column=date"
        issues.append(_issue("INVALID_TIMESTAMP", "timestamp cannot be parsed", source_ref))

    valid = raw.loc[~parsed_time.isna()].copy()
    valid["timestamp"] = [to_utc(value, source_timezone) for value in parsed_time.dropna()]
    valid = valid.drop(columns=["date"])

    renamed: dict[str, str] = {}
    for column in [name for name in valid.columns if name != "timestamp"]:
        raw_key = canonical_column(namespace, str(column))
        meta = tags.get(raw_key)
        renamed[column] = meta.signal_id if meta else canonical_column(namespace, str(column))
        if meta is None or meta.mapping_status is not MappingStatus.CONFIRMED:
            issues.append(
                _issue(
                    "TAG_UNCONFIRMED",
                    "telemetry tag is absent or unconfirmed in config/tags.csv",
                    f"{path.as_posix()}#column={column}",
                    renamed[column],
                )
            )
    valid = valid.rename(columns=renamed)

    for column in [name for name in valid.columns if name != "timestamp"]:
        original = valid[column]
        numeric = pd.to_numeric(original, errors="coerce")
        bad_value = original.notna() & numeric.isna()
        for row_index in valid.index[bad_value]:
            source_ref = f"{path.as_posix()}#row={row_index + 2};column={column}"
            issues.append(
                _issue("INVALID_VALUE", "telemetry value is not finite numeric data", source_ref)
            )
        valid[column] = numeric

    valid = _collapse_telemetry_duplicates(valid, path, issues)
    valid = valid.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return TelemetryRead(valid, tuple(issues))


def _collapse_telemetry_duplicates(
    frame: pd.DataFrame, path: Path, issues: list[Issue]
) -> pd.DataFrame:
    """Collapse equal duplicate timestamps and expose conflicting values."""
    duplicate_mask = frame.duplicated("timestamp", keep=False)
    if not duplicate_mask.any():
        return frame
    rows = [frame.loc[~duplicate_mask]]
    value_columns = [column for column in frame.columns if column != "timestamp"]
    for timestamp, group in frame.loc[duplicate_mask].groupby("timestamp", sort=False):
        row: dict[str, object] = {"timestamp": timestamp}
        for column in value_columns:
            values = group[column].dropna().unique()
            row[column] = values[0] if len(values) == 1 else pd.NA
            if len(values) > 1:
                issues.append(
                    _issue(
                        "SOURCE_CONFLICT",
                        "different values share one source timestamp",
                        f"{path.as_posix()}#timestamp={timestamp};column={column}",
                        column,
                    )
                )
        rows.append(pd.DataFrame([row]))
    return pd.concat(rows, ignore_index=True)


def normalize_section(section: str) -> str:
    """Convert a verbose LIMS section heading to a stable namespace."""
    section = section.strip().rstrip(".").replace("..", ".")
    unit_match = re.search(r"Установка\s+'([^']+)'", section)
    point_match = re.search(r"Точка отбора\s+'([^']+)'", section)
    if not unit_match:
        return section
    namespace = unit_match.group(1)
    return f"{namespace}.{point_match.group(1)}" if point_match else namespace


def _mapping(
    raw_name: str, raw_unit: object, stage: Stage, tags: dict[str, TagMeta], source_ref: str
) -> tuple[str, str, Validity, list[Issue]]:
    """Resolve a raw tag and unit while recording every mapping violation."""
    meta = tags.get(raw_name)
    unit = resolve_unit(raw_unit)
    issues: list[Issue] = []
    validity = Validity.VALID
    signal_id = meta.signal_id if meta else raw_name
    if meta is None or meta.mapping_status is not MappingStatus.CONFIRMED:
        validity = Validity.INVALID
        issues.append(
            _issue(
                "TAG_UNCONFIRMED",
                "source tag is absent or unconfirmed in config/tags.csv",
                source_ref,
                signal_id,
            )
        )
    elif meta.stage is not stage:
        validity = Validity.INVALID
        issues.append(
            _issue("TAG_STAGE_MISMATCH", "tag stage differs from source", source_ref, signal_id)
        )
    if unit == Unit.UNKNOWN.value or (meta and meta.canonical_unit != unit):
        validity = Validity.INVALID
        issues.append(
            _issue(
                "UNIT_UNCONFIRMED",
                "source unit is unknown or differs from the canonical dictionary",
                source_ref,
                signal_id,
            )
        )
    return signal_id, unit, validity, issues


def read_pak(
    path: str | Path,
    tags: dict[str, TagMeta],
    source_timezone: str = "Europe/Moscow",
) -> QualityRead:
    """Read each PAK timestamp/value pair independently."""
    path = Path(path)
    raw = pd.read_excel(path, header=None)
    observations: list[Observation] = []
    issues: list[Issue] = []
    col = 0
    while col < raw.shape[1] - 1:
        tag_raw = raw.iloc[0, col]
        if pd.isna(tag_raw):
            col += 1
            continue
        raw_name = str(tag_raw).strip()
        raw_unit = raw.iloc[1, col]
        mapping_ref = f"{path.as_posix()}#rows=1:2;columns={col + 1},{col + 2}"
        signal_id, unit, mapping_validity, mapping_issues = _mapping(
            raw_name, raw_unit, Stage.HYDROTREATMENT, tags, mapping_ref
        )
        issues.extend(mapping_issues)
        for row_index in range(2, raw.shape[0]):
            timestamp_raw, value_raw = raw.iloc[row_index, col], raw.iloc[row_index, col + 1]
            if pd.isna(timestamp_raw) and pd.isna(value_raw):
                continue
            source_ref = f"{path.as_posix()}#row={row_index + 1};columns={col + 1},{col + 2}"
            try:
                measured_at = to_utc(timestamp_raw, source_timezone)
            except (TypeError, ValueError):
                issues.append(
                    _issue("INVALID_TIMESTAMP", "PAK timestamp cannot be parsed", source_ref)
                )
                continue
            numeric = pd.to_numeric(pd.Series([value_raw]), errors="coerce").iloc[0]
            validity = mapping_validity
            value = None if pd.isna(numeric) else float(numeric)
            if value is None:
                validity = Validity.INVALID
                issues.append(
                    _issue("INVALID_VALUE", "PAK value is not numeric", source_ref, signal_id)
                )
            observations.append(
                Observation(
                    id=_observation_id(source_ref),
                    signal_id=signal_id,
                    stage=Stage.HYDROTREATMENT,
                    source=SourceKind.PAK,
                    measured_at=measured_at,
                    available_at=measured_at,
                    value=value,
                    unit=unit,
                    validity=validity,
                    source_ref=source_ref,
                )
            )
        col += 2
    return QualityRead(tuple(observations), tuple(issues))


def read_lims(
    path: str | Path,
    tags: dict[str, TagMeta],
    source_timezone: str = "Europe/Moscow",
    lims_delay_hours: float = 6.0,
) -> QualityRead:
    """Read each LIMS timestamp/value pair and apply publication delay."""
    path = Path(path)
    raw = pd.read_excel(path, header=None)
    observations: list[Observation] = []
    issues: list[Issue] = []
    current_section = ""
    col = 0
    while col < raw.shape[1] - 1:
        section_raw = raw.iloc[_ROW_SECTION, col]
        if pd.notna(section_raw):
            current_section = str(section_raw)
        parameter_raw = raw.iloc[_ROW_PARAMETER, col]
        if pd.isna(parameter_raw):
            col += 1
            continue
        parameter = str(parameter_raw).strip()
        namespace = normalize_section(current_section)
        raw_name = f"{NS_LIMS}:{namespace}:{parameter}"
        stage = tag_stage(namespace.split(".", 1)[0])
        raw_unit = raw.iloc[_ROW_UNIT, col]
        mapping_ref = f"{path.as_posix()}#rows=1:3;columns={col + 1},{col + 2}"
        signal_id, unit, mapping_validity, mapping_issues = _mapping(
            raw_name, raw_unit, stage, tags, mapping_ref
        )
        issues.extend(mapping_issues)
        for row_index in range(_ROW_DATA_START, raw.shape[0]):
            timestamp_raw, value_raw = raw.iloc[row_index, col], raw.iloc[row_index, col + 1]
            if pd.isna(timestamp_raw) and pd.isna(value_raw):
                continue
            source_ref = f"{path.as_posix()}#row={row_index + 1};columns={col + 1},{col + 2}"
            try:
                measured_at = to_utc(timestamp_raw, source_timezone)
            except (TypeError, ValueError):
                issues.append(
                    _issue("INVALID_TIMESTAMP", "LIMS timestamp cannot be parsed", source_ref)
                )
                continue
            numeric = pd.to_numeric(pd.Series([value_raw]), errors="coerce").iloc[0]
            validity = mapping_validity
            value = None if pd.isna(numeric) else float(numeric)
            if value is None:
                validity = Validity.INVALID
                issues.append(
                    _issue("INVALID_VALUE", "LIMS value is not numeric", source_ref, signal_id)
                )
            observations.append(
                Observation(
                    id=_observation_id(source_ref),
                    signal_id=signal_id,
                    stage=stage,
                    source=SourceKind.LIMS,
                    measured_at=measured_at,
                    available_at=measured_at + timedelta(hours=lims_delay_hours),
                    value=value,
                    unit=unit,
                    validity=validity,
                    source_ref=source_ref,
                )
            )
        col += 2
    return QualityRead(tuple(observations), tuple(issues))


__all__ = [
    "NS_LIMS",
    "QualityRead",
    "TelemetryRead",
    "normalize_section",
    "read_lims",
    "read_pak",
    "read_telemetry_csv",
]
