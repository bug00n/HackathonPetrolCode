"""Acceptance checks for stage-2 prepared-data CLI wiring."""

from __future__ import annotations

import json
import tarfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from source.contracts import DatasetManifest, ProcessState
from source.data.prepare import PreparedData, load_prepared_dataset, write_prepared_dataset
from source.main import main

FIXTURES = Path(__file__).parent / "fixtures"


def _write_minimal_materials(root: Path) -> None:
    raw_data = root / "raw" / "data"
    raw_data.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2026-01-15 11:40:00", "2026-01-15 11:50:00"],
            "T1": [130.0, 131.0],
        }
    ).to_csv(raw_data / "avt_tags.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2026-01-15 11:40:00", "2026-01-15 11:50:00"],
            "F1": [50.0, 51.0],
        }
    ).to_csv(raw_data / "242000_tags.csv", index=False)
    with tarfile.open(root / "data.rar", "w") as archive:
        archive.add(raw_data, arcname="data")

    pd.DataFrame(
        [
            ["24-2000:D15", None],
            ["кг/м3", None],
            [datetime(2026, 1, 15, 12), 830.0],
        ]
    ).to_excel(root / "Выгрузка ПАК 01.01.2023 - н.в_.xlsx", header=False, index=False)
    section = "Установка 'Гидроочистка'. Точка отбора '2'. Продукт 'ДТ'"
    pd.DataFrame(
        [
            [section, None],
            ["Mg.Sulfur", None],
            ["мг/кг", None],
            ["Количество значений:", 1],
            [datetime(2026, 1, 15, 12), 8.4],
        ]
    ).to_excel(root / "ЛИМСы 01.01.2023 - н.в_ (2).xlsx", header=False, index=False)


def test_prepared_dataset_roundtrip(tmp_path: Path) -> None:
    """Verify a written prepared dataset can be loaded back without contract drift."""
    data = PreparedData(
        telemetry=pd.read_csv(FIXTURES / "data/telemetry.csv"),
        quality=pd.read_csv(FIXTURES / "data/quality.csv"),
        issues=pd.read_csv(FIXTURES / "data/issues.csv"),
        manifest=DatasetManifest.model_validate_json(
            (FIXTURES / "data/manifest.json").read_text(encoding="utf-8")
        ),
        feature_order=tuple(
            json.loads((FIXTURES / "data/feature_order.json").read_text(encoding="utf-8"))
        ),
    )

    dataset_path = write_prepared_dataset(data, tmp_path)
    loaded = load_prepared_dataset(dataset_path)

    assert loaded.manifest == data.manifest
    assert loaded.feature_order == data.feature_order
    pd.testing.assert_frame_equal(loaded.telemetry, data.telemetry, check_dtype=False)
    pd.testing.assert_frame_equal(loaded.quality, data.quality, check_dtype=False)
    pd.testing.assert_frame_equal(loaded.issues, data.issues, check_dtype=False)


def test_prepare_and_build_state_cli_on_small_materials(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify CLI can prepare fixture-like materials and build a history state."""
    materials = tmp_path / "materials"
    materials.mkdir()
    _write_minimal_materials(materials)
    output = tmp_path / "processed"

    assert (
        main(
            [
                "prepare",
                "--materials",
                str(materials),
                "--config",
                "config/runtime.toml",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    prepare_payload = json.loads(capsys.readouterr().out)
    dataset_path = Path(prepare_payload["dataset_path"])
    assert dataset_path.is_dir()
    assert prepare_payload["row_counts"]["quality"] >= 2
    assert prepare_payload["issues_count"] >= 0

    assert (
        main(
            [
                "build-state",
                "--dataset",
                str(dataset_path),
                "--scenario",
                "history",
                "--as-of",
                "2026-01-15T15:10:00Z",
            ]
        )
        == 0
    )
    state = ProcessState.model_validate_json(capsys.readouterr().out)
    assert state.dataset_id == prepare_payload["dataset_id"]
    assert state.signals["ht:2:Mg.Sulfur"].selected is not None
    assert state.signals["ht:2:Mg.Sulfur"].selected.source.value == "lims"


def test_build_state_cli_reports_missing_dataset(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify a missing prepared dataset returns a readable CLI error."""
    assert (
        main(
            [
                "build-state",
                "--dataset",
                "missing/dataset",
                "--scenario",
                "history",
                "--as-of",
                "2026-01-15T14:10:00Z",
            ]
        )
        == 1
    )
    assert "missing prepared dataset files" in capsys.readouterr().err
