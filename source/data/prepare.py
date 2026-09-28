"""Stage-0 normalization helpers and the in-memory prepared dataset."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from source.contracts import (
    DatasetManifest,
    Issue,
    Observation,
    RuntimeConfig,
    Severity,
    SourceArtifact,
    Stage,
    TagMeta,
    Unit,
    Validity,
)

PREPARATION_VERSION = "2"
_ARCHIVE_MEMBERS = {"data/avt_tags.csv", "data/242000_tags.csv"}
_PUBLISH_ATTEMPTS = 5
_PUBLISH_RETRY_SECONDS = 0.05
_PREPARED_CACHE_LOCK = threading.Lock()
_PREPARED_FILES = (
    "manifest.json",
    "telemetry.csv.gz",
    "quality.csv.gz",
    "issues.csv.gz",
    "feature_order.json",
)

RAW_UNIT_MAP: dict[str, Unit] = {
    "°С": Unit.CELSIUS,
    "°C": Unit.CELSIUS,
    "кг/м3": Unit.DENSITY,
    "% об.": Unit.PERCENT_VOLUME,
    "% масс.": Unit.PERCENT_MASS,
    "мг/кг": Unit.MG_KG,
    "ppm": Unit.PPM,
    "ед.цет.ч.": Unit.CETANE,
    "мм2/с": Unit.MM2_S,
    "кПа": Unit.KPA,
    "МПа": Unit.MPA,
    "тыс.м3/ч": Unit.THOUSAND_M3_H,
    "м3/ч": Unit.M3_H,
    "т/ч": Unit.T_H,
    "Nm3/h": Unit.NM3_H,
    "Нм3/ч": Unit.NM3_H,
    "нм3/ч": Unit.NM3_H,
    "%": Unit.PERCENT,
}


@dataclass(frozen=True)
class PreparedData:
    """Local container passed to state construction and offline ML."""

    telemetry: pd.DataFrame
    quality: pd.DataFrame
    issues: pd.DataFrame
    manifest: DatasetManifest
    feature_order: tuple[str, ...]


def resolve_unit(raw: Any) -> str:
    """Map only explicitly known spellings; an unknown unit stays unknown."""
    if raw is None or pd.isna(raw):
        return Unit.UNKNOWN.value
    return RAW_UNIT_MAP.get(str(raw).strip(), Unit.UNKNOWN).value


def to_utc(value: Any, source_timezone: str) -> datetime:
    """Interpret naive source timestamps in the configured zone, then use UTC."""
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("timestamp is missing or invalid")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(ZoneInfo(source_timezone))
    return timestamp.tz_convert(UTC).to_pydatetime()


def canonical_column(namespace: str, raw_name: str) -> str:
    """Prefix a raw telemetry tag exactly once with its canonical stage."""
    prefix = f"{namespace}:"
    return raw_name if raw_name.startswith(prefix) else f"{prefix}{raw_name}"


def tag_stage(namespace: str) -> Stage:
    """Map the two supplied installation namespaces to design stages."""
    if namespace in {"АВТ", "avt"}:
        return Stage.AVT
    if namespace in {"24-2000", "Гидроочистка", "ht"}:
        return Stage.HYDROTREATMENT
    raise ValueError(f"unknown namespace: {namespace}")


def issue_frame(issues: tuple[Issue, ...] | list[Issue]) -> pd.DataFrame:
    """Create the exact persisted issues table from validated objects."""
    return pd.DataFrame(
        (
            {"source_ref": item.source_ref, "code": item.code, "detail": item.detail}
            for item in issues
        ),
        columns=["source_ref", "code", "detail"],
    )


def quality_frame(observations: list[dict[str, object]]) -> pd.DataFrame:
    """Create the exact persisted quality table in a stable column order."""
    columns = [
        "observation_id",
        "signal_id",
        "stage",
        "source",
        "measured_at",
        "available_at",
        "value",
        "unit",
        "validity",
        "source_ref",
    ]
    return pd.DataFrame(observations, columns=columns)


def known_feature_order(tags: dict[str, TagMeta]) -> tuple[str, ...]:
    """Freeze the order shared by training and serving."""
    return tuple(sorted({tag.signal_id for tag in tags.values()}))


def _sha256(path: Path) -> str:
    """Calculate a streaming SHA-256 digest for a source file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_artifact(path: Path, root: Path) -> SourceArtifact:
    """Describe one input file for the reproducibility manifest."""
    try:
        artifact_path = path.relative_to(root).as_posix()
    except ValueError:
        artifact_path = path.as_posix()
    return SourceArtifact(path=artifact_path, sha256=_sha256(path), size_bytes=path.stat().st_size)


def _replace_path(source: Path, target: Path) -> None:
    """Wrap platform-specific atomic replace for retry tests."""
    source.replace(target)


def _publish_directory(temporary: Path, target: Path) -> None:
    """Publish a prepared directory with a bounded retry for transient Windows locks."""
    for attempt in range(_PUBLISH_ATTEMPTS):
        try:
            _replace_path(temporary, target)
            return
        except PermissionError:
            if attempt == _PUBLISH_ATTEMPTS - 1:
                raise
            time.sleep(_PUBLISH_RETRY_SECONDS * (2**attempt))


def _validate_archive_entries(entries: list[tuple[str, bool, bool]]) -> None:
    """Accept only the two CSV files and their real parent directory."""
    expected = _ARCHIVE_MEMBERS | {"data"}
    seen: set[str] = set()
    for name, is_directory, is_file in entries:
        # A single leading ./ is harmless; stripping arbitrary dots would hide ../.
        normalized = name.removeprefix("./")
        if normalized == "data/":
            normalized = "data"
        if (
            "\\" in name
            or normalized not in expected
            or normalized in seen
            or (normalized == "data" and not is_directory)
            or (normalized != "data" and not is_file)
        ):
            raise ValueError(f"unsafe archive member: {name}")
        seen.add(normalized)
    if seen != expected:
        raise ValueError(f"unexpected archive members: {sorted(seen)}")


def _extract_telemetry(archive: Path, destination: Path) -> Path:
    """Validate archive members before extraction to a temporary directory."""
    try:
        with tarfile.open(archive) as stream:
            members = stream.getmembers()
            _validate_archive_entries(
                [(member.name, member.isdir(), member.isfile()) for member in members]
            )
            destination_root = destination.resolve()
            for member in members:
                target = (destination / member.name).resolve()
                try:
                    target.relative_to(destination_root)
                except ValueError as exc:
                    raise ValueError(f"unsafe archive member: {member.name}") from exc
            stream.extractall(destination, members=members)
            return destination / "data"
    except tarfile.TarError:
        pass

    listing = subprocess.run(
        ["tar", "-tf", str(archive)], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    details = subprocess.run(
        ["tar", "-tvf", str(archive)], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    if len(listing) != len(details):
        raise ValueError("archive member listings disagree")
    _validate_archive_entries(
        [
            (name, detail.startswith("d"), detail.startswith("-"))
            for name, detail in zip(listing, details, strict=True)
        ]
    )
    subprocess.run(["tar", "-xf", str(archive), "-C", str(destination)], check=True)
    return destination / "data"


def _deduplicate_quality(
    observations: tuple[Observation, ...], issues: list[Issue]
) -> tuple[Observation, ...]:
    """Collapse equal quality observations and mark conflicting duplicates."""
    grouped: dict[tuple[object, ...], list[Observation]] = {}
    for item in observations:
        grouped.setdefault((item.source, item.signal_id, item.measured_at), []).append(item)
    result: list[Observation] = []
    for group in grouped.values():
        first = group[0]
        values = {(item.value, item.unit, item.available_at, item.validity) for item in group}
        if len(values) == 1:
            result.append(first)
            continue
        source_ref = ";".join(item.source_ref for item in group)
        result.append(
            first.model_copy(
                update={"value": None, "validity": Validity.CONFLICT, "source_ref": source_ref}
            )
        )
        issues.append(
            Issue(
                code="SOURCE_CONFLICT",
                severity=Severity.WARNING,
                signal_id=first.signal_id,
                detail="conflicting values share source, signal and measured_at",
                source_ref=source_ref,
            )
        )
    return tuple(sorted(result, key=lambda item: (item.measured_at, item.signal_id, item.id)))


def prepare_dataset(materials_dir: Path, config: RuntimeConfig) -> PreparedData:
    """Build normalized in-memory tables and a reproducibility manifest."""
    from source.config import config_fingerprint, load_tag_dictionary, load_telemetry_rules
    from source.data.ingest import read_lims, read_pak, read_telemetry_csv

    materials_dir = materials_dir.resolve()
    archive = materials_dir / "data.rar"
    pak_path = materials_dir / "Выгрузка ПАК 01.01.2023 - н.в_.xlsx"
    lims_path = materials_dir / "ЛИМСы 01.01.2023 - н.в_ (2).xlsx"
    tag_path = config.tag_dictionary_path.resolve()
    rules_path = config.telemetry_rules_path.resolve()
    for path in (archive, pak_path, lims_path, tag_path, rules_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    tags = load_tag_dictionary(tag_path)
    telemetry_rules = load_telemetry_rules(rules_path)
    issues: list[Issue] = []
    with tempfile.TemporaryDirectory(prefix="neftekod-stage0-") as temp_dir:
        telemetry_dir = _extract_telemetry(archive, Path(temp_dir))
        avt = read_telemetry_csv(
            telemetry_dir / "avt_tags.csv",
            "avt",
            tags,
            config.source_timezone,
            telemetry_rules=telemetry_rules,
        )
        ht = read_telemetry_csv(
            telemetry_dir / "242000_tags.csv",
            "ht",
            tags,
            config.source_timezone,
            telemetry_rules=telemetry_rules,
        )
        issues.extend(avt.issues)
        issues.extend(ht.issues)
        telemetry = avt.frame.merge(ht.frame, how="outer", on="timestamp", validate="one_to_one")

    pak = read_pak(pak_path, tags, config.source_timezone)
    lims = read_lims(lims_path, tags, config.source_timezone, config.lims_delay_hours)
    issues.extend(pak.issues)
    issues.extend(lims.issues)
    observations = _deduplicate_quality(pak.observations + lims.observations, issues)
    observation_rows = [
        {
            "observation_id": item.id,
            **item.model_dump(mode="python", exclude={"id"}),
        }
        for item in observations
    ]
    quality = quality_frame(observation_rows)

    sources = tuple(
        _source_artifact(path, materials_dir.parent)
        for path in (archive, pak_path, lims_path, tag_path, rules_path)
    )
    config_hash = hashlib.sha256(
        config_fingerprint(config, preparation_only=True).encode()
    ).hexdigest()
    tag_hash = _sha256(tag_path)
    telemetry_rules_hash = _sha256(rules_path)
    identity_parts = sorted(item.sha256 for item in sources) + [
        config_hash,
        tag_hash,
        telemetry_rules_hash,
        PREPARATION_VERSION,
    ]
    dataset_id = hashlib.sha256("\n".join(identity_parts).encode()).hexdigest()[:12]
    time_ranges: dict[str, tuple[datetime, datetime] | None] = {
        "telemetry": (
            telemetry["timestamp"].min().to_pydatetime(),
            telemetry["timestamp"].max().to_pydatetime(),
        ),
        "quality": None
        if quality.empty
        else (
            pd.Timestamp(quality["measured_at"].min()).to_pydatetime(),
            pd.Timestamp(quality["measured_at"].max()).to_pydatetime(),
        ),
    }
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        preparation_version=PREPARATION_VERSION,
        created_at=datetime.now(UTC),
        source_timezone=config.source_timezone,
        assumptions=(
            f"naive source timestamps interpreted as {config.source_timezone}",
            f"LIMS availability delayed by {config.lims_delay_hours:g} hours",
        ),
        sources=sources,
        config_sha256=config_hash,
        tag_dictionary_sha256=tag_hash,
        telemetry_rules_sha256=telemetry_rules_hash,
        row_counts={
            "telemetry": len(telemetry),
            "quality": len(quality),
            "issues": len(issues),
        },
        time_ranges=time_ranges,
    )
    return PreparedData(
        telemetry=telemetry.sort_values("timestamp", kind="stable").reset_index(drop=True),
        quality=quality,
        issues=issue_frame(issues),
        manifest=manifest,
        feature_order=known_feature_order(tags),
    )


def write_prepared_dataset(data: PreparedData, output_root: Path) -> Path:
    """Atomically publish the stage-2 prepared files under ``dataset_id``."""
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / data.manifest.dataset_id
    manifest_path = target / "manifest.json"
    feature_order_path = target / "feature_order.json"
    files = {
        "telemetry.csv.gz": data.telemetry,
        "quality.csv.gz": data.quality,
        "issues.csv.gz": data.issues,
    }
    if manifest_path.exists():
        existing = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if existing.model_dump(
            exclude={"created_at", "schema_version", "prepared_sha256"}
        ) != data.manifest.model_dump(exclude={"created_at", "schema_version", "prepared_sha256"}):
            raise FileExistsError(f"dataset_id collision at {target}")
        missing = [
            path.name
            for path in (feature_order_path, *(target / name for name in files))
            if not path.is_file()
        ]
        if missing:
            raise FileExistsError(f"incomplete prepared dataset at {target}: missing {missing}")
        _verify_prepared_files(target, existing)
        return target

    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=output_root))
    try:
        for name, frame in files.items():
            frame.to_csv(temporary / name, index=False, compression="gzip")
        (temporary / "feature_order.json").write_text(
            json.dumps(list(data.feature_order), ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        hashes = {name: _sha256(temporary / name) for name in _PREPARED_FILES[1:]}
        published_manifest = data.manifest.model_copy(
            update={
                "schema_version": "1.1" if data.manifest.preparation_version == "2" else "1.0",
                "prepared_sha256": hashes,
            }
        )
        (temporary / "manifest.json").write_text(
            published_manifest.model_dump_json(indent=2), encoding="utf-8", newline="\n"
        )
        required = (
            temporary / "manifest.json",
            temporary / "feature_order.json",
            *(temporary / name for name in files),
        )
        if any(not path.is_file() for path in required):
            raise OSError("prepared dataset publication is incomplete")
        _publish_directory(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def _verify_prepared_files(dataset_path: Path, manifest: DatasetManifest) -> None:
    expected = manifest.prepared_sha256
    if expected is None:
        release_root = dataset_path.parent.parent.parent
        release_file = release_root / "config/release_manifest.json"
        if release_file.is_file():
            release = json.loads(release_file.read_text(encoding="utf-8"))
            if (release_root / release.get("prepared_dataset", "")).resolve() == dataset_path:
                expected = release.get("prepared_sha256")
                if not isinstance(expected, dict):
                    raise ValueError("pinned legacy dataset has no trusted prepared hashes")
    if expected is None:
        return  # Unpinned legacy development fixtures cannot be retroactively authenticated.
    if set(expected) != set(_PREPARED_FILES[1:]):
        raise ValueError("prepared file hash manifest is incomplete")
    for name, digest in expected.items():
        if not isinstance(digest, str) or _sha256(dataset_path / name) != digest:
            raise ValueError(f"prepared dataset checksum mismatch: {name}")


@lru_cache(maxsize=1)
def _read_prepared_dataset(dataset_path: Path, file_hashes: tuple[str, ...]) -> PreparedData:
    """Keep one verified, unmodified parsed dataset in process memory."""
    required = tuple(dataset_path / name for name in _PREPARED_FILES)
    manifest_path, telemetry_path, quality_path, issues_path, feature_order_path = required
    data = PreparedData(
        telemetry=pd.read_csv(telemetry_path),
        quality=pd.read_csv(quality_path),
        issues=pd.read_csv(issues_path),
        manifest=DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8")),
        feature_order=tuple(json.loads(feature_order_path.read_text(encoding="utf-8"))),
    )
    if tuple(_sha256(path) for path in required) != file_hashes:
        raise OSError(f"prepared dataset changed while loading: {dataset_path}")
    return data


def load_prepared_dataset(dataset_path: Path) -> PreparedData:
    """Load prepared data, reusing a content-checked in-memory parse when possible."""
    dataset_path = dataset_path.resolve()
    required = tuple(dataset_path / name for name in _PREPARED_FILES)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{dataset_path}: missing prepared dataset files: {missing}")
    file_hashes = tuple(_sha256(path) for path in required)
    with _PREPARED_CACHE_LOCK:
        cached = _read_prepared_dataset(dataset_path, file_hashes)
    _verify_prepared_files(dataset_path, cached.manifest)
    return PreparedData(
        telemetry=cached.telemetry.copy(),
        quality=cached.quality.copy(),
        issues=cached.issues.copy(),
        manifest=cached.manifest.model_copy(deep=True),
        feature_order=cached.feature_order,
    )


__all__ = [
    "PreparedData",
    "canonical_column",
    "issue_frame",
    "known_feature_order",
    "load_prepared_dataset",
    "quality_frame",
    "prepare_dataset",
    "resolve_unit",
    "tag_stage",
    "to_utc",
    "write_prepared_dataset",
]
