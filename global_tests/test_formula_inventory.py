"""The stage-0 VAK/LA inventory is read as text and never evaluated."""

from pathlib import Path

import pandas as pd
import pytest

from source.ml.formulas import (
    EXPERT_VAK_CORRECTIONS,
    apply_expert_vak_corrections,
    load_lab_parameters,
    load_vak_formulas,
)


@pytest.fixture(scope="module")
def ml_xlsx(tmp_path_factory):
    """Build a small workbook fixture containing VAK formulas and LA sections."""
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
    """Verify VAK formulas are inventoried as text rather than executed."""
    formulas = load_vak_formulas(ml_xlsx)
    assert formulas["24-2000:GODT:T90"] == "162.998+0.12945*T12"
    assert all(str(value).strip() for value in formulas.values())


def test_expert_corrections_are_explicit_and_do_not_mutate_raw_inventory(ml_xlsx) -> None:
    """Keep the supplied workbook intact while exposing the dated corrections."""
    raw = load_vak_formulas(ml_xlsx)

    corrected = apply_expert_vak_corrections(raw)

    present_corrections = set(EXPERT_VAK_CORRECTIONS).intersection(raw)
    assert present_corrections
    assert {tag_id: corrected[tag_id] for tag_id in present_corrections} == {
        tag_id: EXPERT_VAK_CORRECTIONS[tag_id] for tag_id in present_corrections
    }
    assert raw["24-2000:GODT:T90"] == "162.998+0.12945*T12"
    assert "F15/2000" in corrected["24-2000:GODT:T90"]


def test_every_expert_correction_matches_the_supplied_workbook() -> None:
    """Guard against a correction key that no longer names an inventoried formula."""
    raw = load_vak_formulas(Path("materials/Теги_хакатон.xlsx"))

    assert set(EXPERT_VAK_CORRECTIONS) <= set(raw)


def test_lab_parameter_inventory_keeps_sections(ml_xlsx) -> None:
    """Verify laboratory parameters remain grouped by their source section."""
    parameters = load_lab_parameters(ml_xlsx)
    avt_section = next(key for key in parameters if "АВТ" in key)
    ht_section = next(key for key in parameters if "Гидроочистка" in key)
    assert parameters[avt_section] == ["ПТФ", "Т90%", "Т50%"]
    assert "Плотность при 15 °С" in parameters[ht_section]
