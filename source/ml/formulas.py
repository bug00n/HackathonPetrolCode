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

# Corrections supplied by the task authors on 2026-09-10; see DESIGN §16.
EXPERT_VAK_CORRECTIONS: dict[str, str] = {
    "24-2000:GODT:T90": (
        "162.998+0.12945*T12+59.57*(F15/2000)+0.00036*W7+0.26366*T23-424.72638*F1/F26"
    ),
    "24-2000:GODT:T50": "44.625+10.0224*P13+0.06981*F9+0.471*T6",
    "24-2000:GODT:CloudPoint": (
        "0.0002*F22+0.0021*W7+0.00008*F25-0.30656*F1+0.12018*T6"
        "+0.01916*F9-48.254-0.05249*T16+0.00011"
    ),
    "24-2000:GODT:CFPP": ("0.22088*T23-102.375-47.75834*P8+0.03862*F9+43.60207*W7+43.81849*P24"),
    "24-2000:GODT:T95": ("0.03814*F9-9.201-0.00002*F2+0.50*T6+0.48321*LIMS:24-2000.Pipeline.95%.T"),
    "AVT6:240-350:CFPP": ("31.40363-0.06784*T33+17.411*P67-8.11544*P4-0.47309*(F65/F32+F30)"),
}


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


def apply_expert_vak_corrections(formulas: dict[str, str]) -> dict[str, str]:
    """Overlay only author-confirmed corrections on an inventoried formula set."""
    return {
        tag_id: EXPERT_VAK_CORRECTIONS.get(tag_id, formula) for tag_id, formula in formulas.items()
    }


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


__all__ = [
    "EXPERT_VAK_CORRECTIONS",
    "SHEET_LA",
    "SHEET_VAK",
    "apply_expert_vak_corrections",
    "load_lab_parameters",
    "load_vak_formulas",
]
