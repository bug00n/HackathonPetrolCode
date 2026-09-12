"""Stage-6 reproducible demonstrations, model freeze checks and journal export."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from source.config import load_runtime_config, load_scenario
from source.contracts import (
    CandidateEvaluation,
    DecisionContext,
    ProcessState,
    Recommendation,
)
from source.ml.artifacts import sha256_file
from source.orchestrator import run_cycle

JOURNAL_FILES = (
    "metadata.json",
    "input.json",
    "features.json",
    "trace.jsonl",
    "candidates.jsonl",
    "result.json",
)


@dataclass(frozen=True)
class EpisodeSpec:
    """One checked-in acceptance episode and its stable expectations."""

    id: str
    scenario: str
    expected_status: str
    expected_reason_codes: tuple[str, ...]
    expected_baseline_quality: dict[str, float | None]
    expected_selected_quality: dict[str, float | None] | None
    expected_recipe: dict[str, float] | None
    expected_additive_fraction: float | None


def load_episode_specs(path: Path) -> tuple[EpisodeSpec, ...]:
    """Read and strictly validate the small Stage-6 episode catalog."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("unsupported episode catalog schema_version")
    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError("episode catalog must contain a non-empty episodes list")
    episodes: list[EpisodeSpec] = []
    for raw in raw_episodes:
        if not isinstance(raw, Mapping):
            raise ValueError("each episode must be an object")
        required = {
            "id",
            "scenario",
            "expected_status",
            "expected_reason_codes",
            "expected_baseline_quality",
            "expected_selected_quality",
            "expected_recipe",
            "expected_additive_fraction",
        }
        missing = required.difference(raw)
        if missing:
            raise ValueError(f"episode is missing fields: {sorted(missing)}")
        selected_quality = raw["expected_selected_quality"]
        recipe = raw["expected_recipe"]
        episodes.append(
            EpisodeSpec(
                id=str(raw["id"]),
                scenario=str(raw["scenario"]),
                expected_status=str(raw["expected_status"]),
                expected_reason_codes=tuple(str(item) for item in raw["expected_reason_codes"]),
                expected_baseline_quality=_quality_expectation(raw["expected_baseline_quality"]),
                expected_selected_quality=(
                    None if selected_quality is None else _quality_expectation(selected_quality)
                ),
                expected_recipe=(
                    None
                    if recipe is None
                    else {str(key): float(value) for key, value in dict(recipe).items()}
                ),
                expected_additive_fraction=_optional_float(raw["expected_additive_fraction"]),
            )
        )
    if len({episode.id for episode in episodes}) != len(episodes):
        raise ValueError("episode ids must be unique")
    return tuple(episodes)


def recommendation_fingerprint(result: Recommendation) -> str:
    """Hash stable decision content while excluding the random run identifier."""
    payload = result.model_dump(mode="json")
    payload.pop("run_id", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_acceptance_suite(
    root: Path,
    *,
    episodes_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run checked-in episodes and export their complete journals."""
    root = Path(root)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"acceptance output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        run_root = temporary / "runs"
        config = load_runtime_config(root / "config/runtime.toml")
        summaries: list[dict[str, Any]] = []
        journal_sources: list[tuple[str, Path]] = []
        started = time.perf_counter()
        for episode in load_episode_specs(episodes_path):
            scenario = load_scenario(root / f"config/scenarios/{episode.scenario}.json")
            fixture = json.loads(
                (root / f"global_tests/fixtures/model_demo/{episode.scenario}.json").read_text(
                    encoding="utf-8"
                )
            )
            state = ProcessState.model_validate(fixture["state"])
            cycle_started = time.perf_counter()
            result = run_cycle(
                data=None,
                as_of=state.as_of,
                model=None,
                scenario=scenario,
                config=config,
                context=DecisionContext(),
                run_dir=run_root,
            )
            cycle_seconds = time.perf_counter() - cycle_started
            journal_dir = run_root / result.run_id
            _check_episode(episode, result, scenario)
            recipe, additive_fraction = _effective_selected_recipe(result, scenario)
            summaries.append(
                {
                    "episode_id": episode.id,
                    "scenario": episode.scenario,
                    "status": result.status.value,
                    "reason_codes": list(result.reason_codes),
                    "baseline_quality": _quality_snapshot(result.baseline),
                    "selected_quality": _quality_snapshot(result.selected),
                    "selected_recipe": recipe,
                    "selected_additive_fraction": additive_fraction,
                    "operator_recommendation": result.status.value in {"hold", "recommend"},
                    "decision_fingerprint": recommendation_fingerprint(result),
                    "cycle_seconds": cycle_seconds,
                    "run_id": result.run_id,
                }
            )
            journal_sources.append((episode.id, journal_dir))
        archive = export_journals(journal_sources, temporary / "journals.zip")
        report = {
            "schema_version": "1.0",
            "episodes_file": Path(episodes_path).relative_to(root).as_posix(),
            "episode_count": len(summaries),
            "elapsed_seconds": time.perf_counter() - started,
            "max_cycle_seconds": max(item["cycle_seconds"] for item in summaries),
            "episodes": summaries,
            "journal_archive": archive.name,
            "evidence_boundary": (
                "Historical metrics assess forecast accuracy; model-demo recommendations prove "
                "only the declared synthetic sulfur/T95/cetane and additive model."
            ),
        }
        _write_json(temporary / "summary.json", report)
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def export_journals(
    journals: Iterable[tuple[str, Path]],
    destination: Path,
) -> Path:
    """Export complete journals with checksums and stable archive metadata."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"journal archive already exists: {destination}")
    sources = tuple(journals)
    if not sources:
        raise ValueError("at least one journal is required")
    if len({name for name, _ in sources}) != len(sources):
        raise ValueError("journal export names must be unique")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"schema_version": "1.0", "journals": []}
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, directory in sorted(sources):
                files = []
                for filename in JOURNAL_FILES:
                    path = Path(directory) / filename
                    if not path.is_file():
                        raise FileNotFoundError(
                            f"incomplete journal {directory}: missing {filename}"
                        )
                    archive_name = f"{name}/{filename}"
                    _write_zip_bytes(archive, archive_name, path.read_bytes())
                    files.append({"path": archive_name, "sha256": sha256_file(path)})
                manifest["journals"].append({"name": name, "files": files})
            _write_zip_bytes(
                archive,
                "manifest.json",
                (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
                    "utf-8"
                ),
            )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def verify_model_freeze(root: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify every frozen local artifact and evaluation report by SHA-256."""
    root = Path(root)
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("unsupported model freeze schema_version")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("model freeze must contain artifacts")
    verified_ids: list[str] = []
    for frozen in artifacts:
        if not isinstance(frozen, Mapping):
            raise ValueError("frozen artifact must be an object")
        directory = root / str(frozen["path"])
        metadata_path = directory / "metadata.json"
        metrics_path = directory / "metrics.json"
        model_path = directory / "model.joblib"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        _require_hash(model_path, str(frozen["model_sha256"]))
        _require_hash(metadata_path, str(frozen["metadata_sha256"]))
        _require_hash(metrics_path, str(frozen["metrics_sha256"]))
        if metadata.get("model_id") != frozen.get("model_id"):
            raise ValueError("frozen model_id does not match metadata")
        if metadata.get("model_sha256") != frozen.get("model_sha256"):
            raise ValueError("frozen model hash does not match metadata")
        if metadata.get("training_dataset_id") != payload.get("training_dataset_id"):
            raise ValueError("frozen model uses another training dataset")
        if metadata.get("git_commit") != payload.get("training_git_commit"):
            raise ValueError("frozen model uses another training code revision")
        if metadata.get("python_version") != payload.get("python_version"):
            raise ValueError("frozen model uses another Python version")
        if metadata.get("sklearn_version") != payload.get("sklearn_version"):
            raise ValueError("frozen model uses another scikit-learn version")
        if metadata.get("capabilities", {}).get("supports_actions") is not False:
            raise ValueError("Stage-6 forecast artifacts must not enable action control")
        if str(metadata.get("git_commit", "")).endswith("-dirty"):
            raise ValueError("frozen model was trained from a dirty worktree")
        if "test_used_for_selection" in metrics and metrics["test_used_for_selection"] is not False:
            raise ValueError("point model used final test for selection")
        if "test_used_for_tuning" in metrics and metrics["test_used_for_tuning"] is not False:
            raise ValueError("upper model used final test for tuning")
        verified_ids.append(str(frozen["model_id"]))
    for report in payload.get("evaluation_reports", []):
        _require_hash(root / str(report["path"]), str(report["sha256"]))
    return {
        "training_dataset_id": payload.get("training_dataset_id"),
        "verified_models": verified_ids,
        "evaluation_reports": len(payload.get("evaluation_reports", [])),
    }


def _check_episode(
    episode: EpisodeSpec,
    result: Recommendation,
    scenario: Any,
) -> None:
    if result.status.value != episode.expected_status:
        raise ValueError(
            f"episode {episode.id} status changed: {result.status.value} != "
            f"{episode.expected_status}"
        )
    missing_reasons = set(episode.expected_reason_codes).difference(result.reason_codes)
    if missing_reasons:
        raise ValueError(f"episode {episode.id} lost reasons: {sorted(missing_reasons)}")
    _check_quality(
        _quality_snapshot(result.baseline),
        episode.expected_baseline_quality,
        f"episode {episode.id} baseline",
    )
    actual_selected = _quality_snapshot(result.selected)
    if episode.expected_selected_quality is None:
        if actual_selected is not None:
            raise ValueError(f"episode {episode.id} unexpectedly selected a result")
    else:
        _check_quality(
            actual_selected,
            episode.expected_selected_quality,
            f"episode {episode.id} selected",
        )
    recipe, additive_fraction = _effective_selected_recipe(result, scenario)
    if recipe != episode.expected_recipe:
        raise ValueError(f"episode {episode.id} selected recipe changed")
    _check_optional_number(
        additive_fraction,
        episode.expected_additive_fraction,
        f"episode {episode.id} additive fraction",
    )


def _quality_snapshot(evaluation: CandidateEvaluation | None) -> dict[str, float | None] | None:
    if evaluation is None:
        return None
    estimates = {}
    for assessment in evaluation.assessments:
        estimates.update(assessment.metrics)
    return {
        "sulfur_upper": _estimate_value(estimates.get("sulfur"), "upper"),
        "t95_upper": _estimate_value(estimates.get("t95"), "upper"),
        "cetane_lower": _estimate_value(estimates.get("cetane_number"), "lower"),
    }


def _estimate_value(estimate: Any, field: str) -> float | None:
    value = None if estimate is None else getattr(estimate, field)
    return None if value is None else float(value)


def _effective_selected_recipe(
    result: Recommendation, scenario: Any
) -> tuple[dict[str, float] | None, float | None]:
    if result.selected is None:
        return None, None
    candidate = result.selected.candidate
    if candidate.blend_mass_fractions:
        return dict(candidate.blend_mass_fractions), candidate.additive_mass_fraction
    return (
        dict(scenario.current_blend_mass_fractions),
        scenario.current_additive_mass_fraction,
    )


def _quality_expectation(value: object) -> dict[str, float | None]:
    if not isinstance(value, Mapping):
        raise ValueError("quality expectation must be an object")
    expected_keys = {"sulfur_upper", "t95_upper", "cetane_lower"}
    if set(value) != expected_keys:
        raise ValueError(f"quality expectation needs keys: {sorted(expected_keys)}")
    return {str(key): _optional_float(item) for key, item in value.items()}


def _check_quality(
    actual: dict[str, float | None] | None,
    expected: dict[str, float | None],
    label: str,
) -> None:
    if actual is None:
        raise ValueError(f"{label} is unavailable")
    for key, expected_value in expected.items():
        _check_optional_number(actual[key], expected_value, f"{label} {key}")


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError("expected a number or null")
    return float(value)


def _check_optional_number(actual: float | None, expected: float | None, label: str) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f"{label} availability changed")
        return
    if abs(actual - expected) > 1e-9:
        raise ValueError(f"{label} changed: {actual} != {expected}")


def _require_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"checksum mismatch for {path}: {actual} != {expected}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_zip_bytes(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, content)


__all__ = [
    "EpisodeSpec",
    "export_journals",
    "load_episode_specs",
    "recommendation_fingerprint",
    "run_acceptance_suite",
    "verify_model_freeze",
]
