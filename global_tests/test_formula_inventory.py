"""The stage-0 VAK/LA inventory is read as text and never evaluated."""

import pandas as pd
import pytest

from source.ml.formulas import load_lab_parameters, load_vak_formulas


@pytest.fixture(scope="module")
def ml_xlsx(tmp_path_factory):
    path = tmp_path_factory.mktemp("ml") / "ml_tags.xlsx"
    vak = pd.DataFrame(
        {
            "24-2000. ГО ДТ": ["24-2000:GODT:T90", "24-2000:GODT:T50", None],
            "Unnamed: 1": ["162.998+0.12945*T12", "44.625+10.0224*P13", None],
        }
    )
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


def test_vak_formulas_are_preserved_as_unexecuted_text(ml_xlsx) -> None:
    formulas = load_vak_formulas(ml_xlsx)
    assert formulas["24-2000:GODT:T90"] == "162.998+0.12945*T12"
    assert all(str(value).strip() for value in formulas.values())


def test_lab_parameter_inventory_keeps_sections(ml_xlsx) -> None:
    parameters = load_lab_parameters(ml_xlsx)
    avt_section = next(key for key in parameters if "АВТ" in key)
    ht_section = next(key for key in parameters if "Гидроочистка" in key)
    assert parameters[avt_section] == ["ПТФ", "Т90%", "Т50%"]
    assert "Плотность при 15 °С" in parameters[ht_section]
