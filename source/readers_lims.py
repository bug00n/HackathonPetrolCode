"""Чтение лабораторных анализов ЛИМС.

Структура файла «ЛИМСы 01.01.2023 - н.в_ (2).xlsx»:
- один лист, 108 колонок = 54 пары (timestamp, value);
- строка 0: имя секции («Установка 'АВТ'. Точка отбора '1'. Продукт '...'»)
  в первой колонке группы, остальные — NaN (растянутый заголовок);
- строка 1: имя параметра (CFPP, D15, Mg.Sulfur, CetanNumber, ...);
- строка 2: единица измерения (°С, кг/м3, мг/кг, ...);
- строка 3: служебный счётчик («Количество значений:», N) — отбрасывается;
- строки 4+: данные.

Каждая пара разбирается отдельно; секция становится пространством имён,
что даёт уникальные теги вида «ЛИМС:АВТ.2:D15» (требование плана:
пространства имён АВТ и гидроочистки).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from source.models import ReliabilityFlag, Sample, SourceKind
from source.tags import resolve_unit

# Префикс пространства имён лабораторных данных
NS_LIMS = "ЛИМС"

_ROW_SECTION = 0
_ROW_PARAMETER = 1
_ROW_UNIT = 2
_ROW_COUNT = 3
_ROW_DATA_START = 4


def normalize_section(section: str) -> str:
    """Нормализует имя секции к короткому пространству имён.

    «Установка 'АВТ'. Точка отбора '2.1'. Продукт '...'» → «АВТ.2.1»,
    «Установка 'Гидроочистка'. Точка отбора '2'...» → «Гидроочистка.2».
    Двойные точки в имени секции (опечатка источника) сглаживаются.
    """
    section = section.strip().rstrip(".").replace("..", ".")
    unit_match = re.search(r"Установка\s+'([^']+)'", section)
    point_match = re.search(r"Точка отбора\s+'([^']+)'", section)
    if not unit_match:
        return section
    namespace = unit_match.group(1)
    if point_match:
        namespace = f"{namespace}.{point_match.group(1)}"
    return namespace


def read_lims(path: str | Path) -> list[Sample]:
    """Читает выгрузку ЛИМС и возвращает плоский список измерений.

    Пустые ячейки пропускаются: у каждого параметра своя шкала времени,
    отсутствие анализа — не нуль, а отсутствие измерения.
    """
    raw = pd.read_excel(path, header=None)

    samples: list[Sample] = []
    n_cols = raw.shape[1]

    col = 0
    current_section = ""
    while col < n_cols - 1:
        # Обновляем текущую секцию, если в этой колонке задано новое имя
        section_raw = raw.iloc[_ROW_SECTION, col]
        if pd.notna(section_raw):
            current_section = str(section_raw)

        param_raw = raw.iloc[_ROW_PARAMETER, col]
        if pd.isna(param_raw):
            col += 1
            continue

        parameter = str(param_raw).strip()
        namespace = normalize_section(current_section)
        tag = f"{NS_LIMS}:{namespace}:{parameter}"
        unit = resolve_unit(raw.iloc[_ROW_UNIT, col])

        ts_col, val_col = col, col + 1
        data = raw.iloc[_ROW_DATA_START:, [ts_col, val_col]].copy()
        data.columns = ["timestamp", "value"]
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
        data["value"] = pd.to_numeric(data["value"], errors="coerce")
        data = data.dropna(subset=["timestamp", "value"])

        for ts, value in data.itertuples(index=False):
            samples.append(
                Sample(
                    tag=tag,
                    source=SourceKind.LIMS,
                    unit=unit,
                    timestamp=ts.to_pydatetime(),
                    value=float(value),
                    namespace=namespace,
                    parameter=parameter,
                    reliability=ReliabilityFlag.RELIABLE,
                )
            )

        col += 2

    return samples


__all__ = ["NS_LIMS", "normalize_section", "read_lims"]
