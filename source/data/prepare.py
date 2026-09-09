"""Нормализация и справочник тегов (DESIGN.md §3: data/prepare.py).

Читает справочник «Теги_хакатон.xlsx» (листы КИП и ПАК) и приводит
единицы измерения к enum. Смысл тега берётся только из справочника
(требование ТЗ), поэтому неподтверждённые теги не угадываются.

Формулы ВАК и перечень ЛА относятся к слою ML (DESIGN §142) и живут
в ``source/ml/formulas.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from source.contracts import SourceKind, TagMeta, Unit

# Названия листов справочника, относящихся к backend-нормализации
SHEET_KIP = "КИП"
SHEET_PAK = "ПАК"

# Пространства имён тегов (требование плана этапа 0)
NS_AV = "АВТ"
NS_GODT = "24-2000"


def _unit_from_raw(raw: object) -> Unit:
    """Приводит строку единицы из файла к enum, неизвестные — в UNKNOWN."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return Unit.UNKNOWN
    text = str(raw).strip()
    try:
        return Unit(text)
    except ValueError:
        return Unit.UNKNOWN


def resolve_unit(raw_unit: object) -> Unit:
    """Публичная функция приведения единицы для читателей данных."""
    return _unit_from_raw(raw_unit)


def _kip_column_mapping(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Извлекает тройки (namespace, tag_id, description) из листа КИП.

    Лист содержит пары колонок: '<NS> (описание)', '<NS>'.
    """
    result: list[tuple[str, str, str]] = []
    columns = list(df.columns)
    for desc_col, tag_col in zip(columns[0::2], columns[1::2], strict=False):
        desc_name = str(desc_col)
        namespace = desc_name.split("(")[0].strip()
        for _, row in df.iterrows():
            tag_id = row[tag_col]
            description = row[desc_col]
            if pd.isna(tag_id) or pd.isna(description):
                continue
            result.append((namespace, str(tag_id).strip(), str(description).strip()))
    return result


def _tag_key(namespace: str, tag_id: str) -> str:
    """Строит ключ тега с пространством имён без дублирования.

    Теги ПАК/ВАК уже содержат namespace в имени ('24-2000:Mg.Sulfur.Q'),
    теги КИП — короткие ('T1'). В обоих случаях ключ должен быть уникален
    и единообразен: '<namespace>:<остаток>'.
    """
    if tag_id.startswith(f"{namespace}:"):
        return tag_id
    return f"{namespace}:{tag_id}"


def load_tag_dictionary(path: str | Path) -> dict[str, TagMeta]:
    """Читает справочник тегов и возвращает словарь по tag_id с namespace.

    Ключ словаря — '<namespace>:<tag_id>', чтобы теги АВТ и 24-2000
    не пересекались (требование плана: пространства имён тегов).
    """
    xls = pd.ExcelFile(path)

    tags: dict[str, TagMeta] = {}

    if SHEET_KIP in xls.sheet_names:
        kip = pd.read_excel(xls, sheet_name=SHEET_KIP)
        for namespace, tag_id, description in _kip_column_mapping(kip):
            tags[_tag_key(namespace, tag_id)] = TagMeta(
                tag_id=tag_id,
                description=description,
                namespace=namespace,
            )

    if SHEET_PAK in xls.sheet_names:
        pak = pd.read_excel(xls, sheet_name=SHEET_PAK)
        for _, row in pak.iterrows():
            description = row.iloc[0]
            tag_id = row.iloc[1]
            if pd.isna(tag_id) or pd.isna(description):
                continue
            key = _tag_key(NS_GODT, str(tag_id).strip())
            tags[key] = TagMeta(
                tag_id=str(tag_id).strip(),
                description=str(description).strip(),
                namespace=NS_GODT,
                source_kind=SourceKind.PAK,
            )

    return tags


__all__ = [
    "NS_AV",
    "NS_GODT",
    "SHEET_KIP",
    "SHEET_PAK",
    "load_tag_dictionary",
    "resolve_unit",
]
