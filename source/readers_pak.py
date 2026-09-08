"""Чтение выгрузки ПАК (поточные анализаторы).

Структура файла «Выгрузка ПАК 01.01.2023 - н.в_.xlsx»:
- один лист, колонки идут парами: '<тег>' (timestamp) + 'Unnamed: N' (значение);
- строка 0 содержит единицы измерения ('ppm', 'кг/м3');
- строки 1+ содержат данные (timestamp в чётных колонках, значение в нечётных).

Каждая пара разбирается отдельно (требование плана: «Разбирать каждую пару
„дата — значение" ЛИМС/ПАК отдельно»), единицы сохраняются явно.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from source.models import ReliabilityFlag, Sample, SourceKind
from source.tags import NS_GODT, resolve_unit


def read_pak(path: str | Path) -> list[Sample]:
    """Читает выгрузку ПАК и возвращает плоский список измерений.

    Непустые значения становятся Sample с reliability=RELIABLE;
    пустые ячейки пропускаются (у ПАК разные шкалы времени у колонок,
    поэтому отсутствие значения — не «нуль», а отсутствие измерения).
    """
    raw = pd.read_excel(path, header=None)

    samples: list[Sample] = []
    n_cols = raw.shape[1]

    # Пары определяем по факту: тег-колонка — это колонка, у которой
    # в строке 0 непустой тег; её значения — в следующей колонке.
    # (Пары не обязаны быть строго чётными: между ними бывают
    # пустые колонки-разделители.)
    col = 0
    while col < n_cols - 1:
        tag_raw = raw.iloc[0, col]
        if pd.isna(tag_raw):
            col += 1
            continue

        tag = str(tag_raw).strip()
        ts_col, val_col = col, col + 1
        unit = resolve_unit(raw.iloc[1, ts_col])

        # Данные начинаются со строки 2 (строки 0-1 — теги/единицы)
        data = raw.iloc[2:, [ts_col, val_col]].copy()
        data.columns = ["timestamp", "value"]
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
        data["value"] = pd.to_numeric(data["value"], errors="coerce")
        data = data.dropna(subset=["timestamp", "value"])

        for ts, value in data.itertuples(index=False):
            samples.append(
                Sample(
                    tag=tag,
                    source=SourceKind.PAK,
                    unit=unit,
                    timestamp=ts.to_pydatetime(),
                    value=float(value),
                    namespace=NS_GODT,
                    reliability=ReliabilityFlag.RELIABLE,
                )
            )

        col += 2

    return samples


__all__ = ["read_pak"]
