"""Юнит-тесты чтения справочника тегов.

Использует синтетический Excel-файл с той же структурой листов,
что и materials/Теги_хакатон.xlsx (КИП, ПАК, ВАК, ЛА).
"""

import pandas as pd
import pytest

from source.contracts import SourceKind, Unit
from source.data.prepare import NS_AV, NS_GODT, load_tag_dictionary
from source.ml.formulas import load_lab_parameters, load_vak_formulas

NS = "24-2000"


@pytest.fixture(scope="module")
def tags_xlsx(tmp_path_factory):
    """Создаёт синтетический справочник тегов, повторяющий структуру оригинала."""
    path = tmp_path_factory.mktemp("tags") / "tags.xlsx"

    # Лист КИП: пары колонок (описание, тег)
    kip = pd.DataFrame(
        {
            "АВТ (описание)": ["Температура верха К1", "Давление верха К1", None],
            "АВТ": ["T1", "P2", None],
            f"{NS} (описание)": ["Расход бензина", "Давление на входе", None],
            NS: ["F1", "P3", None],
        }
    )

    # Лист ПАК: описание → тег
    pak = pd.DataFrame(
        {
            NS: [
                "Блок стабилизации. Поточный анализатор содержания серы",
                "Плотность при 15C (расчетная)",
            ],
            "Unnamed: 1": ["24-2000:Mg.Sulfur.Q", "24-2000:D15"],
        }
    )

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
        kip.to_excel(writer, sheet_name="КИП", index=False)
        vak.to_excel(writer, sheet_name="ВАК", index=False)
        pak.to_excel(writer, sheet_name="ПАК", index=False)
        la.to_excel(writer, sheet_name="ЛА", index=False)

    return path


class TestLoadTagDictionary:
    def test_kip_tags_loaded(self, tags_xlsx):
        tags = load_tag_dictionary(tags_xlsx)
        assert "АВТ:T1" in tags
        assert "АВТ:P2" in tags
        assert "24-2000:F1" in tags
        assert "24-2000:P3" in tags

    def test_kip_descriptions(self, tags_xlsx):
        tags = load_tag_dictionary(tags_xlsx)
        assert tags["АВТ:T1"].description == "Температура верха К1"
        assert tags["24-2000:F1"].description == "Расход бензина"

    def test_namespaces_are_separated(self, tags_xlsx):
        tags = load_tag_dictionary(tags_xlsx)
        assert tags["АВТ:T1"].namespace == NS_AV
        assert tags["24-2000:F1"].namespace == NS_GODT

    def test_nan_rows_skipped(self, tags_xlsx):
        tags = load_tag_dictionary(tags_xlsx)
        # Проверяем, что нет пустых тегов из строк с NaN
        assert all(t.tag_id.strip() for t in tags.values())

    def test_pak_source_kind(self, tags_xlsx):
        tags = load_tag_dictionary(tags_xlsx)
        # Теги ПАК: ключ без двойного префикса, source_kind == PAK
        assert "24-2000:24-2000:Mg.Sulfur.Q" not in tags
        assert "24-2000:Mg.Sulfur.Q" in tags
        assert tags["24-2000:Mg.Sulfur.Q"].source_kind == SourceKind.PAK
        pak_tags = [t for t in tags.values() if t.source_kind == SourceKind.PAK]
        assert len(pak_tags) == 2


class TestLoadVakFormulas:
    def test_formulas_loaded(self, tags_xlsx):
        formulas = load_vak_formulas(tags_xlsx)
        assert "24-2000:GODT:T90" in formulas
        assert formulas["24-2000:GODT:T90"] == "162.998+0.12945*T12"

    def test_no_nan_formulas(self, tags_xlsx):
        formulas = load_vak_formulas(tags_xlsx)
        assert all(str(v).strip() for v in formulas.values())


class TestLoadLabParameters:
    def test_sections_parsed(self, tags_xlsx):
        params = load_lab_parameters(tags_xlsx)
        assert len(params) == 2
        av_section = next(k for k in params if "АВТ" in k)
        assert params[av_section] == ["ПТФ", "Т90%", "Т50%"]

    def test_go_section(self, tags_xlsx):
        params = load_lab_parameters(tags_xlsx)
        go_section = next(k for k in params if "Гидроочистка" in k)
        assert "Плотность при 15 °С" in params[go_section]


class TestRealUnits:
    def test_resolve_unit_known(self):
        from source.data.prepare import resolve_unit

        assert resolve_unit("°С") is Unit.CELSIUS
        assert resolve_unit("кг/м3") is Unit.DENSITY
        assert resolve_unit("мг/кг") is Unit.MG_KG

    def test_resolve_unit_unknown(self):
        from source.data.prepare import resolve_unit

        assert resolve_unit("фурлонги") is Unit.UNKNOWN
        assert resolve_unit(None) is Unit.UNKNOWN
