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


def normalize_formula_text(value: object) -> str:
    """Convert organiser formula typography to stable ASCII-like text."""
    import re

    text = str(value).strip().replace("−", "-").replace("×", "*")
    return re.sub(r"(?<=\d),(?=\d)", ".", text)


# Corrections supplied in the organisers' 2026-09-15 VAK workbook.  Keep the
# raw inventory untouched and overlay only these exact, auditable expressions.
FORMULA_SOURCE_VERSION = "organizer-vak-2026-09-15"
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
    "AVT6:240-350:D15": ("791.22872-5.30294*(F65/(F32+F30))+0.52755*T66-0.15629*T33"),
    "AVT6:240-350:CFPP": ("31.40363-0.06784*T33+17.411*P67-8.11544*P4-0.47309*F65/(F32+F30)"),
    "AVT6:350:T50": (
        "493.6798+1.281193*T42-0.955342*T48-0.018454*F31+0.265904*F57-0.082047*T66-0.545083*T33"
    ),
    "AVT6:350:I350": ("39.562-1.62865*L43+0.76664*T6-0.22361*T18+0.00031*F64*(T15-T11)"),
}


def load_vak_formulas(path: str | Path) -> dict[str, str]:
    """Читает лист ВАК: тег виртуального анализатора → формула.

    Формулы возвращаются как строки без вычисления: их проверка —
    отдельная задача (план: «Формулы ВАК требуют проверки»).
    """
    workbook = pd.ExcelFile(path)
    if SHEET_VAK in workbook.sheet_names:
        sheets: tuple[str, ...] = (SHEET_VAK,)
    else:
        sheets = tuple(
            str(name)
            for name in workbook.sheet_names
            if "АВТ" in str(name).upper() or "24-2000" in str(name).upper()
        )
    if not sheets:
        return {}
    formulas: dict[str, str] = {}
    for sheet in sheets:
        vak = pd.read_excel(workbook, sheet_name=sheet)
        columns = [str(column) for column in vak.columns]
        if {"Модель", "Формула"}.issubset({str(column) for column in columns}):
            pairs: tuple[tuple[str, str], ...] = (("Модель", "Формула"),)
        else:
            pairs = tuple(zip(columns[0::2], columns[1::2], strict=False))
        for tag_col, formula_col in pairs:
            for _, row in vak.iterrows():
                tag_id = row[tag_col]
                formula = row[formula_col]
                if pd.isna(tag_id) or pd.isna(formula):
                    continue
                formulas[str(tag_id).strip()] = normalize_formula_text(formula)
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
    "FORMULA_SOURCE_VERSION",
    "SHEET_LA",
    "SHEET_VAK",
    "apply_expert_vak_corrections",
    "load_lab_parameters",
    "load_vak_formulas",
    "normalize_formula_text",
]
