"""Юнит-тесты чтения ЛИМС (синтетический файл со структурой оригинала)."""

from datetime import datetime

import pandas as pd
import pytest

from source.contracts import ReliabilityFlag, SourceKind, Unit
from source.data.ingest import normalize_section, read_lims


@pytest.fixture(scope="module")
def lims_xlsx(tmp_path_factory):
    """Синтетический ЛИМС: 2 секции × 2 параметра × 3 строки данных.

    Структура строк как в оригинале:
    0 — секция (только в первой колонке группы), 1 — параметр,
    2 — единица, 3 — счётчик, 4+ — данные.
    """
    path = tmp_path_factory.mktemp("lims") / "lims.xlsx"

    sec_av = "Установка 'АВТ'. Точка отбора '2'. Продукт 'Дизельное топливо'"
    sec_go = "Установка 'Гидроочистка'.. Точка отбора '2'. Продукт 'Дизельное топливо'"

    rows = []
    # Строка 0: секции
    rows.append([sec_av, None, None, None, sec_go, None, None, None])
    # Строка 1: параметры
    rows.append(["D15", None, "90%.T", None, "Mg.Sulfur", None, "CetaneNumber", None])
    # Строка 2: единицы
    rows.append(["кг/м3", None, "°С", None, "мг/кг", None, "ед.цет.ч.", None])
    # Строка 3: счётчик (должен быть отброшен)
    rows.append(["Количество значений:", 3.0, None, 3.0, "Количество значений:", 2.0, None, 2.0])
    # Строки 4+: данные
    rows.append(
        [
            datetime(2023, 1, 2, 14, 0),
            833.1,
            datetime(2023, 1, 2, 14, 0),
            344.0,
            datetime(2023, 2, 1, 10, 0),
            6.5,
            datetime(2023, 3, 1, 9, 0),
            52.0,
        ]
    )
    rows.append(
        [
            datetime(2023, 1, 4, 14, 0),
            834.2,
            datetime(2023, 1, 4, 14, 0),
            346.0,
            datetime(2023, 2, 3, 10, 0),
            7.1,
            None,
            None,
        ]
    )  # пропуск цетана
    rows.append(
        [
            datetime(2023, 1, 6, 14, 0),
            835.3,
            datetime(2023, 1, 6, 14, 0),
            347.0,
            None,
            None,
            None,
            None,
        ]
    )

    pd.DataFrame(rows).to_excel(path, header=False, index=False)
    return path


class TestNormalizeSection:
    def test_avt(self):
        section = "Установка 'АВТ'. Точка отбора '2'. Продукт 'Дизельное топливо'"
        assert normalize_section(section) == "АВТ.2"

    def test_avt_point_with_dot(self):
        section = "Установка 'АВТ'. Точка отбора '2.1'. Продукт 'Дизельное топливо'"
        assert normalize_section(section) == "АВТ.2.1"

    def test_godt_double_dot_typo(self):
        section = "Установка 'Гидроочистка'.. Точка отбора '2'. Продукт 'Дизельное топливо'"
        assert normalize_section(section) == "Гидроочистка.2"

    def test_frac_diz_point(self):
        section = "Установка 'Гидроочистка'. Точка отбора '1'. Продукт 'ФРАКЦ_ДИЗ'."
        assert normalize_section(section) == "Гидроочистка.1"


class TestReadLims:
    def test_all_samples(self, lims_xlsx):
        samples = read_lims(lims_xlsx)
        # D15: 3, 90%.T: 3, Mg.Sulfur: 2, CetaneNumber: 1
        assert len(samples) == 9

    def test_tags_with_namespace(self, lims_xlsx):
        samples = read_lims(lims_xlsx)
        tags = {s.tag for s in samples}
        assert tags == {
            "ЛИМС:АВТ.2:D15",
            "ЛИМС:АВТ.2:90%.T",
            "ЛИМС:Гидроочистка.2:Mg.Sulfur",
            "ЛИМС:Гидроочистка.2:CetaneNumber",
        }

    def test_units(self, lims_xlsx):
        samples = read_lims(lims_xlsx)
        by_param = {s.parameter: s.unit for s in samples}
        assert by_param["D15"] is Unit.DENSITY
        assert by_param["90%.T"] is Unit.CELSIUS
        assert by_param["Mg.Sulfur"] is Unit.MG_KG
        assert by_param["CetaneNumber"] is Unit.CETANE

    def test_counts_row_skipped(self, lims_xlsx):
        samples = read_lims(lims_xlsx)
        # Строка «Количество значений:» не попала в данные
        assert all(s.value is not None and not isinstance(s.value, str) for s in samples)
        assert all(3.0 != s.value or s.parameter is None for s in samples)
        # Счётчик 3.0 не должен появиться как значение серы
        sulfur = [s for s in samples if s.parameter == "Mg.Sulfur"]
        assert {s.value for s in sulfur} == {6.5, 7.1}

    def test_source_and_reliability(self, lims_xlsx):
        samples = read_lims(lims_xlsx)
        assert all(s.source is SourceKind.LIMS for s in samples)
        assert all(s.reliability is ReliabilityFlag.RELIABLE for s in samples)
        assert all(s.is_usable for s in samples)

    def test_section_carried_between_group_columns(self, lims_xlsx):
        """Секция из колонки группы распространяется на соседние пары."""
        samples = read_lims(lims_xlsx)
        d15 = [s for s in samples if s.parameter == "D15"]
        assert all(s.namespace == "АВТ.2" for s in d15)
