"""Юнит-тесты DataProvider на синтетических файлах."""

from datetime import datetime

import pandas as pd
import pytest

from source.models import SourceKind, Unit
from source.storage import DataProvider


@pytest.fixture(scope="module")
def provider(tmp_path_factory):
    """DataProvider с полным набором синтетических мини-источников."""
    root = tmp_path_factory.mktemp("dp")

    # Телеметрия АВТ
    avt = root / "avt_tags.csv"
    pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2],
            "date": ["2023-01-01 00:00:00", "2023-01-01 00:10:00", "2023-01-01 00:20:00"],
            "T1": [130.37, 130.31, 130.25],
        }
    ).to_csv(avt, index=False)

    # ПАК
    pak = root / "pak.xlsx"
    pd.DataFrame(
        [
            ["24-2000:Mg.Sulfur", None, None, "24-2000:D15", None],
            ["ppm", None, None, "кг/м3", None],
            [datetime(2023, 1, 1), 6.57, None, datetime(2025, 3, 5, 10, 20), 831.5],
        ]
    ).to_excel(pak, header=False, index=False)

    # ЛИМС
    lims = root / "lims.xlsx"
    pd.DataFrame(
        [
            ["Установка 'Гидроочистка'.. Точка отбора '2'. Продукт 'ДТ'", None, None, None],
            ["Mg.Sulfur", None, "CetaneNumber", None],
            ["мг/кг", None, "ед.цет.ч.", None],
            ["Количество значений:", 1.0, None, 1.0],
            [datetime(2023, 2, 1), 6.5, datetime(2023, 3, 1), 52.0],
        ]
    ).to_excel(lims, header=False, index=False)

    # Справочник тегов
    tags = root / "tags.xlsx"
    with pd.ExcelWriter(tags, engine="openpyxl") as writer:
        pd.DataFrame(
            {
                "АВТ (описание)": ["Температура верха К1"],
                "АВТ": ["T1"],
                "24-2000 (описание)": [None],
                "24-2000": [None],
            }
        ).to_excel(writer, sheet_name="КИП", index=False)
        pd.DataFrame(
            {"24-2000": ["Поточный анализатор серы"], "Unnamed: 1": ["24-2000:Mg.Sulfur.Q"]}
        ).to_excel(writer, sheet_name="ПАК", index=False)

    return DataProvider(
        telemetry_paths={"АВТ": avt},
        pak_path=pak,
        lims_path=lims,
        tags_path=tags,
    )


class TestDataProvider:
    def test_telemetry(self, provider):
        df = provider.get_telemetry("АВТ")
        assert len(df) == 3
        assert {"date", "АВТ:T1"} == set(df.columns)

    def test_pak(self, provider):
        samples = provider.get_pak()
        assert len(samples) == 2
        assert all(s.source is SourceKind.PAK for s in samples)

    def test_lims(self, provider):
        samples = provider.get_lims()
        assert len(samples) == 2
        tags = {s.tag for s in samples}
        assert "ЛИМС:Гидроочистка.2:Mg.Sulfur" in tags
        assert "ЛИМС:Гидроочистка.2:CetaneNumber" in tags

    def test_tag_dictionary(self, provider):
        tags = provider.get_tag_dictionary()
        assert "АВТ:T1" in tags
        assert "24-2000:Mg.Sulfur.Q" in tags

    def test_cache_same_object(self, provider):
        assert provider.get_pak() is provider.get_pak()
        assert provider.get_lims() is provider.get_lims()
        assert provider.get_telemetry("АВТ") is provider.get_telemetry("АВТ")

    def test_clear_cache(self, provider):
        first = provider.get_pak()
        provider.clear_cache()
        second = provider.get_pak()
        assert first is not second
        assert first == second  # содержимое то же (воспроизводимость)

    def test_units_propagated(self, provider):
        sulfur = next(s for s in provider.get_lims() if s.parameter == "Mg.Sulfur")
        assert sulfur.unit is Unit.MG_KG
        pak_sulfur = next(s for s in provider.get_pak() if s.tag == "24-2000:Mg.Sulfur")
        assert pak_sulfur.unit is Unit.PPM
