"""Validated loaders for runtime, tag and scenario configuration."""

from __future__ import annotations

import csv
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

from source.contracts import RuntimeConfig, ScenarioConfig, TagMeta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MATERIALS_DIR = PROJECT_ROOT / "materials"
CONFIG_DIR = PROJECT_ROOT / "config"


@dataclass(frozen=True)
class DataPaths:
    """Explicit raw-source paths; tests replace them with small fixtures."""

    telemetry: dict[str, Path]
    pak: Path
    lims: Path
    tags: Path


DEFAULT_PATHS = DataPaths(
    telemetry={
        "avt": DATA_DIR / "avt_tags.csv",
        "ht": DATA_DIR / "242000_tags.csv",
    },
    pak=MATERIALS_DIR / "Выгрузка ПАК 01.01.2023 - н.в_.xlsx",
    lims=MATERIALS_DIR / "ЛИМСы 01.01.2023 - н.в_ (2).xlsx",
    tags=CONFIG_DIR / "tags.csv",
)


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    """Load strict TOML settings; unknown keys are rejected by Pydantic."""
    with Path(path).open("rb") as stream:
        return RuntimeConfig.model_validate(tomllib.load(stream))


def load_scenario(path: str | Path) -> ScenarioConfig:
    """Load one strict JSON scenario."""
    return ScenarioConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_tag_dictionary(path: str | Path) -> dict[str, TagMeta]:
    """Load canonical mappings keyed by raw source name.

    Duplicate raw names are rejected instead of silently overwriting a row.
    Empty CSV cells are normalized only where the contract permits ``None``.
    """
    result: dict[str, TagMeta] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            row["raw_unit"] = row["raw_unit"] or None
            row["controllable"] = row["controllable"].strip().lower() == "true"
            tag = TagMeta.model_validate(row)
            if tag.raw_name in result:
                raise ValueError(f"duplicate raw_name in tag dictionary: {tag.raw_name}")
            result[tag.raw_name] = tag
    return result


def config_fingerprint(config: RuntimeConfig) -> str:
    """Stable JSON used by the preparation manifest hash."""
    return json.dumps(
        config.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


__all__ = [
    "CONFIG_DIR",
    "DATA_DIR",
    "DEFAULT_PATHS",
    "MATERIALS_DIR",
    "PROJECT_ROOT",
    "DataPaths",
    "config_fingerprint",
    "load_runtime_config",
    "load_scenario",
    "load_tag_dictionary",
]
