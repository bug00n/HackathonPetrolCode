"""Юнит-тесты чтения ПАК (синтетический файл со структурой оригинала)."""

from datetime import datetime

import pandas as pd
import pytest

from source.models import ReliabilityFlag, SourceKind, Unit
from source.readers_pak import read_pak


@pytest.fixture(scope="module")
def pak_xlsx(tmp_path_factory):
    """Синтетическая выгрузка ПАК: строка 0 — теги, строка 1 — единицы, далее данные.

    Структура пар колонок как в оригинале: чётная колонка — timestamp
    (строка 0 — тег, строка 1 — единица), нечётная — значение.
    В данных: валидные значения, пропуск (NaN) и мусорные ячейки.
    """
    path = tmp_path_factory.mktemp("pak") / "pak.xlsx"

    rows = []
    rows.append(["24-2000:Mg.Sulfur", None, None, "24-2000:D15", None])  # теги
    rows.append(["ppm", None, None, "кг/м3", None])  # единицы
    rows.append([datetime(2023, 1, 1, 0, 0), 6.5729, None, datetime(2025, 3, 5, 10, 20), 831.5055])
    rows.append([datetime(2023, 1, 1, 0, 10), 6.7300, None, datetime(2025, 3, 5, 10, 30), 831.3279])
    rows.append([datetime(2023, 1, 1, 0, 20), None, None, None, 831.4442])  # пропуски

    df = pd.DataFrame(rows)
    df.to_excel(path, header=False, index=False)
    return path


class TestReadPak:
    def test_all_samples_parsed(self, pak_xlsx):
        samples = read_pak(pak_xlsx)
        # Сера: 2 валидных (3-я без значения), плотность: 2 валидных
        # (3-я без timestamp). Пары с пропуском отбрасываются:
        # значение без времени несинхронизируемо (ТЗ: синхронизация по времени).
        assert len(samples) == 4

    def test_sulfur_samples(self, pak_xlsx):
        samples = read_pak(pak_xlsx)
        sulfur = [s for s in samples if s.tag == "24-2000:Mg.Sulfur"]
        assert len(sulfur) == 2
        assert sulfur[0].value == pytest.approx(6.5729)
        assert sulfur[0].timestamp == datetime(2023, 1, 1, 0, 0)
        assert sulfur[0].unit is Unit.PPM

    def test_density_samples(self, pak_xlsx):
        samples = read_pak(pak_xlsx)
        density = [s for s in samples if s.tag == "24-2000:D15"]
        assert len(density) == 2
        assert density[0].unit is Unit.DENSITY
        assert density[0].value == pytest.approx(831.5055)
        assert density[0].timestamp == datetime(2025, 3, 5, 10, 20)

    def test_source_and_namespace(self, pak_xlsx):
        samples = read_pak(pak_xlsx)
        assert all(s.source is SourceKind.PAK for s in samples)
        assert all(s.namespace == "24-2000" for s in samples)
        assert all(s.reliability is ReliabilityFlag.RELIABLE for s in samples)

    def test_no_nan_values(self, pak_xlsx):
        samples = read_pak(pak_xlsx)
        assert all(s.value is not None for s in samples)
        assert all(s.timestamp is not None for s in samples)
        assert all(s.is_usable for s in samples)
