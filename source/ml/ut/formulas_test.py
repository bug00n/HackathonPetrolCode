"""Юнит-тесты формул ВАК и лабораторных показателей (ml/formulas)."""

import pandas as pd
import pytest

from source.ml.formulas import load_lab_parameters, load_vak_formulas


@pytest.fixture(scope="module")
def ml_xlsx(tmp_path_factory):
    """Синтетический справочник с листами ВАК и ЛА (структура оригинала)."""
    path = tmp_path_factory.mktemp("ml") / "ml_tags.xlsx"

    # Лист ВАК: тег → формула
    vak = pd.DataFrame(
        {
            "24-2000. ГО ДТ": ["24-2000:GODT:T90", "24-2000:GODT:T50", None],
            "Unnamed: 1": ["162.998+0.12945*T12", "44.625+10.0224*P13", None],
        }
    )
    # Лист ЛА: показатель по точкам отбора (колонки выровнены по длине)
    la = pd.DataFrame(
        {
            "Установка 'АВТ'. Точка отбора '1'. Продукт 'Дизельное топливо'": [
                "ПТФ",
                "Т90%",
                "Т50%",
            ],
            "Установка 'Гидроочистка'. Точка отбора '2'. Продукт 'Дизельное топливо'": [
                "Температура помутнения",
                "Плотность при 15 °С",
                None,
            ],
        }
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        vak.to_excel(writer, sheet_name="ВАК", index=False)
        la.to_excel(writer, sheet_name="ЛА", index=False)
    return path


class TestLoadVakFormulas:
    def test_formulas_loaded(self, ml_xlsx):
        formulas = load_vak_formulas(ml_xlsx)
        assert "24-2000:GODT:T90" in formulas
        assert formulas["24-2000:GODT:T90"] == "162.998+0.12945*T12"

    def test_no_nan_formulas(self, ml_xlsx):
        formulas = load_vak_formulas(ml_xlsx)
        assert all(str(v).strip() for v in formulas.values())


class TestLoadLabParameters:
    def test_sections_parsed(self, ml_xlsx):
        params = load_lab_parameters(ml_xlsx)
        assert len(params) == 2
        av_section = next(k for k in params if "АВТ" in k)
        assert params[av_section] == ["ПТФ", "Т90%", "Т50%"]

    def test_go_section(self, ml_xlsx):
        params = load_lab_parameters(ml_xlsx)
        go_section = next(k for k in params if "Гидроочистка" in k)
        assert "Плотность при 15 °С" in params[go_section]
