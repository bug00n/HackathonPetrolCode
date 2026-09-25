"""Acceptance checks for stage-2 prepared-data CLI wiring."""

from __future__ import annotations

import io
import json
import tarfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

import source.data.prepare as prepare_module
from source.contracts import DatasetManifest, ProcessState
from source.data.prepare import PreparedData, load_prepared_dataset, write_prepared_dataset
from source.main import main

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture_prepared_data() -> PreparedData:
    return PreparedData(
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


@pytest.mark.parametrize("external_tar", [False, True])
def test_archive_extraction_rejects_symlinked_data_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, external_tar: bool
) -> None:
    """Archive metadata must not redirect CSV extraction outside the temporary directory."""
    archive_path = tmp_path / "malicious.tar"
    with tarfile.open(archive_path, "w") as archive:
        link = tarfile.TarInfo("data")
        link.type = tarfile.SYMTYPE
        link.linkname = "../outside"
        archive.addfile(link)
        for name in ("avt_tags.csv", "242000_tags.csv"):
            payload = b"date,value\n2026-01-15,1\n"
            member = tarfile.TarInfo(f"data/{name}")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))

    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "extracted"
    destination.mkdir()
    if external_tar:

        def reject_tar_archive(path: Path) -> None:
            raise tarfile.ReadError(path)

        monkeypatch.setattr(prepare_module.tarfile, "open", reject_tar_archive)

    with pytest.raises(ValueError, match="unsafe archive member"):
        prepare_module._extract_telemetry(archive_path, destination)
    assert not list(outside.iterdir())


def test_prepared_dataset_roundtrip(tmp_path: Path) -> None:
    """Verify a written prepared dataset can be loaded back without contract drift."""
    data = _fixture_prepared_data()

    dataset_path = write_prepared_dataset(data, tmp_path)
    loaded = load_prepared_dataset(dataset_path)

    assert loaded.manifest.model_dump(exclude={"prepared_sha256"}) == data.manifest.model_dump(
        exclude={"prepared_sha256"}
    )
    assert loaded.manifest.prepared_sha256 is not None
    assert loaded.feature_order == data.feature_order
    pd.testing.assert_frame_equal(loaded.telemetry, data.telemetry, check_dtype=False)
    pd.testing.assert_frame_equal(loaded.quality, data.quality, check_dtype=False)
    pd.testing.assert_frame_equal(loaded.issues, data.issues, check_dtype=False)


def test_prepared_dataset_memory_cache_isolated_and_invalidated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = write_prepared_dataset(_fixture_prepared_data(), tmp_path)
    read_csv = pd.read_csv
    reads: list[str] = []

    def counted_read_csv(path: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        reads.append(Path(path).name)
        return read_csv(path, *args, **kwargs)

    monkeypatch.setattr(prepare_module.pd, "read_csv", counted_read_csv)
    first = load_prepared_dataset(dataset_path)
    original = first.quality.loc[0, "value"]
    first.quality.loc[0, "value"] = 999
    second = load_prepared_dataset(dataset_path)
    assert second.quality.loc[0, "value"] == original
    assert len(reads) == 3

    second.quality.loc[0, "value"] = 998
    second.quality.to_csv(dataset_path / "quality.csv.gz", index=False, compression="gzip")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_prepared_dataset(dataset_path)
    assert len(reads) == 6


def test_prepared_dataset_publish_retries_transient_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify Windows-style transient locks do not fail prepared dataset publication."""
    calls = 0
    original_replace = prepare_module._replace_path

    def flaky_replace(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError("temporary Windows directory lock")
        original_replace(source, target)

    monkeypatch.setattr(prepare_module, "_replace_path", flaky_replace)

    dataset_path = write_prepared_dataset(_fixture_prepared_data(), tmp_path)

    assert calls == 2
    assert (dataset_path / "manifest.json").is_file()


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


def test_cli_rejects_stale_prepared_dataset_before_ml_diagnostics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify stale manifest hashes fail before downstream ML diagnostics."""
    dataset_path = write_prepared_dataset(_fixture_prepared_data(), tmp_path)

    assert main(["diagnose-ml", "--dataset", str(dataset_path)]) == 1

    error = capsys.readouterr().err
    assert "prepared dataset is not current" in error
    assert "config_sha256" in error
