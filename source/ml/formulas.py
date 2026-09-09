"""Формулы виртуальных анализаторов и лабораторные показатели (ML).

По DESIGN §142 формулы — зона ответственности ML. Читаются из
справочника «Теги_хакатон.xlsx» (листы ВАК и ЛА). Формулы возвращаются
строками без вычисления: их проверка — отдельная задача
(план: «Формулы ВАК требуют проверки»).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Листы справочника, относящиеся к ML
SHEET_VAK = "ВАК"
SHEET_LA = "ЛА"


def load_vak_formulas(path: str | Path) -> dict[str, str]:
    """Читает лист ВАК: тег виртуального анализатора → формула.

    Формулы возвращаются как строки без вычисления: их проверка —
    отдельная задача (план: «Формулы ВАК требуют проверки»).
    """
    if SHEET_VAK not in pd.ExcelFile(path).sheet_names:
        return {}
    vak = pd.read_excel(path, sheet_name=SHEET_VAK)
    formulas: dict[str, str] = {}
    columns = list(vak.columns)
    for tag_col, formula_col in zip(columns[0::2], columns[1::2], strict=False):
        for _, row in vak.iterrows():
            tag_id = row[tag_col]
            formula = row[formula_col]
            if pd.isna(tag_id) or pd.isna(formula):
                continue
            formulas[str(tag_id).strip()] = str(formula).strip()
    return formulas


def load_lab_parameters(path: str | Path) -> dict[str, list[str]]:
    """Читает лист ЛА: точка отбора → список лабораторных показателей.

    Заголовки колонок вида «Установка 'АВТ'. Точка отбора '2'. Продукт '...'»
    служат ключами секций ЛИМС.
    """
    if SHEET_LA not in pd.ExcelFile(path).sheet_names:
        return {}
    la = pd.read_excel(path, sheet_name=SHEET_LA)
    result: dict[str, list[str]] = {}
    for column in la.columns:
        params = [str(v).strip() for v in la[column].dropna() if str(v).strip()]
        result[str(column).strip()] = params
    return result


__all__ = ["SHEET_LA", "SHEET_VAK", "load_lab_parameters", "load_vak_formulas"]
